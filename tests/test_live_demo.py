"""
BotShield – Live demo page, explanations and pipeline-faithful ML retraining tests
"""

import math
import time

import pytest

from app import app
from backend.database.db import get_connection, get_requests, insert_request, upsert_session
from backend.detection.fsm import BehavioralFSM, FSMEvent, map_path_to_symbol
from backend.ml.features import FEATURE_NAMES
from backend.ml.model import IsolationForestModel, load_model, train_unsupervised_model
from backend.ml.realistic import (
    BOT_KINDS,
    build_training_vectors,
    evaluate_model_realistic,
    generate_bot,
    generate_human,
)
from backend.scoring.service import HybridConfig
from backend.simulation import live

DEMO_PAGES = ["/admin/demo", "/admin/api/demo/scenarios", "/admin/api/demo/me"]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["ADMIN_TOKEN"] = "live-demo-test-token"
    with app.test_client() as c:
        yield c


@pytest.fixture
def demo_env(monkeypatch):
    monkeypatch.setenv("BOTSHIELD_DEMO_MODE", "1")


@pytest.fixture(autouse=True)
def _stop_runs():
    """Never let a scenario thread outlive its test (and its temp database)."""
    yield
    for run in list(live._RUNS.values()):
        run.stop_event.set()
    deadline = time.time() + 10
    while any(r.status == "running" for r in live._RUNS.values()) and time.time() < deadline:
        time.sleep(0.05)
    live._RUNS.clear()


@pytest.fixture
def realistic_model(tmp_path):
    path = tmp_path / "realistic.joblib"
    train_unsupervised_model(build_training_vectors(60, seed=5), model_path=path, contamination=0.02,
                             log_transform=True, n_estimators=100, dataset_type="realistic_pipeline_60")
    return str(path)


def _wait_for(run_id, client, timeout=45):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/admin/api/demo/runs/{run_id}").get_json()["run"]
        if run["status"] != "running":
            return run
        time.sleep(0.25)
    raise AssertionError("scenario did not finish in time")


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", DEMO_PAGES)
def test_demo_page_does_not_exist_in_production(client, path):
    assert client.get(path).status_code == 404
    # even with a valid admin token: the feature is demo-only
    assert client.get(path, headers={"X-Admin-Token": "live-demo-test-token"}).status_code == 404


def test_demo_run_api_does_not_exist_in_production(client):
    resp = client.post("/admin/api/demo/runs", json={"scenario": "scraper"})
    assert resp.status_code == 404
    assert live._RUNS == {}


def test_demo_page_rejects_remote_clients(client, demo_env):
    assert client.get("/admin/demo", environ_base={"REMOTE_ADDR": "10.0.0.7"}).status_code == 404


def test_demo_page_renders_in_demo_mode(client, demo_env):
    resp = client.get("/admin/demo")
    assert resp.status_code == 200
    assert "Live detection scenarios" in resp.get_data(as_text=True)
    data = client.get("/admin/api/demo/scenarios").get_json()
    assert {s["id"] for s in data["scenarios"]} == set(live.SCENARIOS)
    assert "/admin/demo" in client.get("/admin/dashboard").get_data(as_text=True)


@pytest.mark.parametrize("kwargs", [
    {"data": {"scenario": "scraper"}},                                     # form post, not JSON
    {"json": {"scenario": "scraper"}, "headers": {"Origin": "https://evil.example"}},
])
def test_start_run_rejects_cross_site_requests(client, demo_env, kwargs):
    assert client.post("/admin/api/demo/runs", **kwargs).status_code == 400
    assert live._RUNS == {}


def test_start_run_rejects_unknown_scenario(client, demo_env):
    assert client.post("/admin/api/demo/runs", json={"scenario": "nope"}).status_code == 400


def test_concurrent_run_limit(client, demo_env):
    for _ in range(live.MAX_CONCURRENT_RUNS):
        assert client.post("/admin/api/demo/runs", json={"scenario": "stealth"}).status_code == 201
    assert client.post("/admin/api/demo/runs", json={"scenario": "stealth"}).status_code == 429


# ---------------------------------------------------------------------------
# End-to-end scenario through the real app
# ---------------------------------------------------------------------------

def test_credential_scenario_runs_live_and_explains_changes(client, demo_env, _use_temp_db):
    resp = client.post("/admin/api/demo/runs", json={"scenario": "credential"})
    assert resp.status_code == 201
    run = _wait_for(resp.get_json()["run"]["run_id"], client)

    assert run["status"] == "finished"
    latest = run["latest"]
    codes = {r["code"] for r in latest["fsm"]["rules"]}
    assert "ANOMALY_HIGH_REQUEST_LOW_TELEMETRY" in codes
    assert latest["fsm"]["score"] > 0
    # No model in tests: FSM-only mode, hybrid equals FSM, ML reported as unavailable
    assert latest["hybrid"]["mode"] == "FSM_ONLY"
    assert latest["hybrid"]["score"] == pytest.approx(latest["fsm"]["score"])
    assert latest["ml"]["score"] is None
    assert len(run["timeline"]) >= 2
    texts = [e["text"] for e in run["log"]]
    assert any(t.startswith("FSM rule fired") for t in texts)
    assert any(e["kind"] == "stage" for e in run["log"])

    conn = get_connection(_use_temp_db)
    try:
        label = conn.execute("SELECT source, is_bot FROM session_labels WHERE session_id = ?",
                             (run["session_id"],)).fetchone()
        requests = get_requests(conn, run["session_id"], limit=500)
        history = conn.execute("SELECT COUNT(*) FROM session_risk_history WHERE session_id = ?",
                               (run["session_id"],)).fetchone()[0]
    finally:
        conn.close()
    assert tuple(label) == ("simulation:live-credential", 1)
    # Every request of the run is logged (1 GET + 18 POST /login)
    assert len(requests) == 19
    assert history == len(run["timeline"])


def test_stop_run(client, demo_env):
    run_id = client.post("/admin/api/demo/runs", json={"scenario": "stealth"}).get_json()["run"]["run_id"]
    time.sleep(0.5)
    assert client.post(f"/admin/api/demo/runs/{run_id}/stop", json={}).status_code == 200
    assert _wait_for(run_id, client, timeout=10)["status"] == "stopped"


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def _http(t, path):
    return FSMEvent(map_path_to_symbol(path), t, f"t{t}", "HTTP_REQUEST", path, int(t * 1000), "HTTP_GET")


def test_fsm_breakdown_adds_up_to_score():
    result = BehavioralFSM().process_events([_http(k * 0.03, f"/product/{k % 6 + 1}") for k in range(12)])
    breakdown = live.fsm_breakdown(result)
    assert {r["code"] for r in breakdown["rules"]} == set(result.anomaly_codes)
    assert min(1.0, breakdown["raw_total"]) == pytest.approx(result.risk_score)


def test_ml_breakdown_lists_unusual_inputs(realistic_model):
    import random
    bot = generate_bot("scraper", random.Random(3))
    from backend.ml.features import extract_session_features
    res = IsolationForestModel(model_path=realistic_model).predict(
        extract_session_features(events=bot.events, requests=bot.requests))
    out = live.ml_breakdown(res, realistic_model)
    assert out["score"] == res.anomaly_score
    assert len(out["unusual_inputs"]) == 3
    assert all(u["feature"] in FEATURE_NAMES and math.isfinite(u["z"]) for u in out["unusual_inputs"])


def test_describe_changes_reports_rule_and_risk_changes():
    prev = {"fsm": {"score": 0.0, "state": "BROWSING", "rules": []},
            "ml": {"score": 0.5, "is_anomaly": False, "threshold": 0.55, "unusual_inputs": []},
            "hybrid": {"score": 0.2, "risk_level": "LOW", "arithmetic": "x"}}
    cur = {"fsm": {"score": 0.6, "state": "SUSPICIOUS", "rules": [
               {"code": "ANOMALY_RAPID_REQUEST_BURST", "title": "Request burst", "penalty": 0.25, "explanation": "e"}]},
           "ml": {"score": 0.6, "is_anomaly": True, "threshold": 0.55,
                  "unusual_inputs": [{"text": "request rate = 9/s"}]},
           "hybrid": {"score": 0.6, "risk_level": "MEDIUM", "arithmetic": "0.60 × 0.600 + 0.40 × 0.600 = 0.600"}}
    notes = live.describe_changes(prev, cur)
    assert any("FSM rule fired — Request burst (+0.25)" in n for n in notes)
    assert any("Behavioral state BROWSING → SUSPICIOUS" in n for n in notes)
    assert any("ML anomaly score 0.50 → 0.60" in n for n in notes)
    assert any("above the model's threshold" in n for n in notes)
    assert any("Risk level LOW → MEDIUM" in n for n in notes)


def test_my_session_endpoint(client, demo_env, _use_temp_db, realistic_model):
    assert client.get("/admin/api/demo/me").get_json()["available"] is False
    sid = "abcdef01-2345-4678-9abc-def012345678"
    conn = get_connection(_use_temp_db)
    upsert_session(conn, sid)
    for i in range(8):
        insert_request(conn, sid, "GET", f"/product/{i % 6 + 1}", 200, timestamp=f"2020-01-01T10:00:00.{i:02d}0000Z")
    conn.commit()
    conn.close()
    app.config["BOTSHIELD_MODEL_PATH"] = realistic_model
    client.set_cookie("botshield_session", sid)
    x = client.get("/admin/api/demo/me").get_json()["explanation"]
    cfg = HybridConfig()
    assert x["hybrid"]["mode"] == "HYBRID"
    assert x["hybrid"]["score"] == pytest.approx(cfg.fsm_weight * x["fsm"]["score"] + cfg.ml_weight * x["ml"]["score"], abs=1e-4)
    assert x["fsm"]["rules"]
    assert len(x["ml"]["unusual_inputs"]) == 3


# ---------------------------------------------------------------------------
# Pipeline-faithful ML retraining
# ---------------------------------------------------------------------------

def test_training_vectors_are_deterministic_and_finite():
    a = build_training_vectors(20, seed=9)
    b = build_training_vectors(20, seed=9)
    assert a == b
    assert len(a) >= 40  # full sessions + in-progress snapshots
    assert all(len(v) == len(FEATURE_NAMES) and all(math.isfinite(x) for x in v) for v in a)


def test_generated_human_matches_tracker_mechanics():
    import random
    s = generate_human(random.Random(1))
    paths = [r["path"] for r in s.requests]
    assert "/api/events" in paths                     # telemetry flushes are logged requests
    types = {e["event_type"] for e in s.events}
    assert {"PAGE_VIEW", "PAGE_EXIT", "CLICK"} <= types
    assert types <= {"PAGE_VIEW", "CLICK", "SCROLL", "MOUSE_ACTIVITY", "KEYBOARD_ACTIVITY", "PAGE_EXIT"}


def test_log_transform_is_stored_in_pipeline(realistic_model):
    bundle, status, _ = load_model(realistic_model)
    assert status.value == "available"
    assert list(bundle["pipeline"].named_steps)[:2] == ["log1p", "scaler"]
    assert bundle["metadata"]["feature_transform"] == "log1p"


def test_default_training_is_unchanged(tmp_path):
    path = tmp_path / "plain.joblib"
    train_unsupervised_model(build_training_vectors(10, seed=2), model_path=path)
    bundle, _, _ = load_model(path)
    assert list(bundle["pipeline"].named_steps) == ["scaler", "forest"]


def test_realistic_evaluation_separates_humans_and_bots(realistic_model):
    report = evaluate_model_realistic(realistic_model)
    per = report["results"]["per_scenario"]
    assert set(per) == {"human", *BOT_KINDS}
    assert report["real_world_accuracy_claim"] is False
    m = report["results"]["metrics"]
    assert m["false_positive_rate"] <= 0.10
    assert m["recall"] >= 0.80


def test_analytics_uses_matching_offline_evaluation(client, realistic_model):
    app.config["BOTSHIELD_MODEL_PATH"] = realistic_model
    data = client.get("/admin/api/analytics", headers={"X-Admin-Token": "live-demo-test-token"}).get_json()
    assert data["offline_evaluation"]["evaluation_name"].startswith("Simulated pipeline sessions")
