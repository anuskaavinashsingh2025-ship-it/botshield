"""
BotShield – Local demo mode, FSM fixes and demo-scenario tests

- Demo mode opens the admin dashboard without a token, for loopback requests only
- Production (no demo flag, BOTSHIELD_ENV=production, remote or proxied clients)
  keeps administrator authentication
- FSM: bursts inside longer sessions are detected; one page load is not counted twice
- Demo scenario: labeled SIMULATED sessions scored with the 60/40 hybrid formula
"""

import pytest

from app import app
from backend.database.db import get_connection
from backend.detection.fsm import BehavioralFSM, FSMEvent, FSMSymbol, map_path_to_symbol
from backend.ml.model import train_unsupervised_model
from backend.ml.train import generate_synthetic_baseline
from backend.scoring.service import HybridConfig

ADMIN_TOKEN = "demo-test-token-456"
ADMIN_PAGES = ["/admin/dashboard", "/admin/"]
ADMIN_APIS = ["/admin/api/stats", "/admin/api/sessions", "/admin/api/analytics"]


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["ADMIN_TOKEN"] = ADMIN_TOKEN
    with app.test_client() as c:
        yield c


@pytest.fixture
def demo_env(monkeypatch):
    monkeypatch.setenv("BOTSHIELD_DEMO_MODE", "1")


# ---------------------------------------------------------------------------
# Demo mode: direct access from this machine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ADMIN_PAGES)
def test_demo_dashboard_opens_without_token(client, demo_env, path):
    resp = client.get(path)
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)
    assert "Security Operations Center" in html
    assert "LOCAL DEMO MODE" in html
    assert "Administrator Token" not in html  # no token screen
    assert "/admin/logout" not in html


@pytest.mark.parametrize("path", ADMIN_APIS)
def test_demo_apis_work_without_token(client, demo_env, path):
    resp = client.get(path)
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


@pytest.mark.parametrize("method", ["get", "post"])
def test_demo_login_route_redirects_to_dashboard(client, demo_env, method):
    resp = getattr(client, method)("/admin/login")
    assert resp.status_code == 302
    assert resp.headers["Location"].endswith("/admin/dashboard")


def test_demo_logout_is_a_no_op_redirect(client, demo_env):
    resp = client.get("/admin/logout")
    assert resp.status_code == 302
    assert "botshield_admin_token=;" not in resp.headers.get("Set-Cookie", "")


def test_demo_works_even_without_configured_token(client, demo_env):
    app.config["ADMIN_TOKEN"] = None
    assert client.get("/admin/api/stats").status_code == 200


# ---------------------------------------------------------------------------
# Production: authentication is retained
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ADMIN_APIS)
def test_production_apis_require_token(client, path):
    assert client.get(path).status_code == 401
    assert client.get(path, headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get(path, headers={"X-Admin-Token": ADMIN_TOKEN}).status_code == 200


def test_production_dashboard_shows_token_prompt(client):
    resp = client.get("/admin/dashboard")
    assert resp.status_code == 401
    assert "Administrator Token" in resp.get_data(as_text=True)


def test_production_token_login_still_sets_cookie(client):
    resp = client.post("/admin/login", data={"token": ADMIN_TOKEN})
    assert resp.status_code == 302
    assert "botshield_admin_token=" in resp.headers.get("Set-Cookie", "")
    page = client.get("/admin/dashboard")
    assert page.status_code == 200
    assert "LOCAL DEMO MODE" not in page.get_data(as_text=True)
    assert "/admin/logout" in page.get_data(as_text=True)


def test_production_get_login_redirects_to_prompt(client):
    resp = client.get("/admin/login")
    assert resp.status_code == 302
    assert client.get("/admin/dashboard").status_code == 401


def test_production_env_overrides_demo_flag(client, demo_env, monkeypatch):
    monkeypatch.setenv("BOTSHIELD_ENV", "production")
    assert client.get("/admin/api/stats").status_code == 401
    assert client.get("/admin/dashboard").status_code == 401


def test_demo_mode_rejects_non_local_clients(client, demo_env):
    remote = {"REMOTE_ADDR": "192.168.1.50"}
    assert client.get("/admin/api/stats", environ_base=remote).status_code == 401
    assert client.get("/admin/dashboard", environ_base=remote).status_code == 401


@pytest.mark.parametrize("header", ["X-Forwarded-For", "X-Real-IP", "Forwarded"])
def test_demo_mode_rejects_proxied_requests(client, demo_env, header):
    resp = client.get("/admin/api/stats", headers={header: "203.0.113.9"})
    assert resp.status_code == 401


def test_app_config_can_disable_demo_mode(client, demo_env):
    app.config["BOTSHIELD_DEMO_MODE"] = False
    assert client.get("/admin/api/stats").status_code == 401


# ---------------------------------------------------------------------------
# FSM transitions and scoring fixes
# ---------------------------------------------------------------------------

def _http(t, path):
    return FSMEvent(map_path_to_symbol(path), t, f"t{t}", "HTTP_REQUEST", path, int(t * 1000), "HTTP_GET")


def _page_view(t, path):
    return FSMEvent(map_path_to_symbol(path), t, f"t{t}", "TELEMETRY", path, int(t * 1000) + 1, "PAGE_VIEW")


def _scroll(t):
    return FSMEvent(FSMSymbol.USER_INTERACTION, t, f"t{t}", "TELEMETRY", "/", int(t * 1000) + 2, "SCROLL")


def _human_browsing():
    events = []
    for i, path in enumerate(["/", "/products", "/product/1", "/about"]):
        t = i * 15
        events += [_http(t, path), _page_view(t + 0.3, path), _scroll(t + 4), _scroll(t + 8)]
    return events


def _run(events):
    return BehavioralFSM().process_events(sorted(events, key=lambda e: e.epoch_timestamp))


def test_fsm_normal_human_browsing_scores_zero():
    result = _run(_human_browsing())
    assert result.risk_score == 0.0
    assert result.anomaly_codes == []
    states = [step.to_state.value for step in result.transition_history]
    assert "PRODUCT_VIEW" in states and result.final_state.value == "BROWSING"


def test_fsm_detects_burst_after_normal_browsing():
    burst = [_http(60 + k * 0.1, f"/product/{k % 6 + 1}") for k in range(20)]
    result = _run(_human_browsing() + burst)
    assert "ANOMALY_RAPID_REQUEST_BURST" in result.anomaly_codes
    assert result.final_state.value == "SUSPICIOUS"
    assert result.risk_score > 0.35  # previously the burst was averaged away


def test_fsm_burst_needs_superhuman_rate():
    # Five pages in ~4 s is fast but humanly possible: no burst anomaly
    paced = [_http(k * 1.0, f"/product/{k + 1}") for k in range(5)]
    assert "ANOMALY_RAPID_REQUEST_BURST" not in _run(paced).anomaly_codes


def test_fsm_does_not_count_one_page_load_twice():
    # Two reloads = two server requests + two PAGE_VIEW mirrors, not four visits
    events = [_http(0, "/"), _page_view(0.3, "/"), _http(5, "/"), _page_view(5.3, "/")]
    result = _run(events)
    assert result.risk_score == 0.0
    assert "ANOMALY_PATH_CYCLING" not in result.anomaly_codes


def test_fsm_still_detects_repeated_reloading():
    events = [e for k in range(5) for e in (_http(k * 1.0, "/"), _page_view(k * 1.0 + 0.3, "/"))]
    assert "ANOMALY_PATH_CYCLING" in _run(events).anomaly_codes


def test_fsm_page_view_only_sessions_still_count_cycles():
    # Without server-side requests (telemetry only), PAGE_VIEWs are the page loads
    events = [_page_view(k * 1.0, "/") for k in range(5)]
    assert "ANOMALY_PATH_CYCLING" in _run(events).anomaly_codes


# ---------------------------------------------------------------------------
# Demo scenario and hybrid scoring
# ---------------------------------------------------------------------------

@pytest.fixture
def demo_rows(_use_temp_db, tmp_path):
    from backend.simulation.demo import run_demo

    model = tmp_path / "model.joblib"
    train_unsupervised_model(generate_synthetic_baseline(60, random_state=3), model_path=model)
    # Demo default pacing; at dwell_scale<1 the "human" loads pages faster than
    # a person can (<150 ms apart), which the FSM rightly flags.
    return run_demo(db_path=_use_temp_db, model_path=str(model), dwell_scale=1.0,
                    sessions_per_profile=1, mode="in-process"), _use_temp_db


def test_demo_scenario_uses_hybrid_formula(demo_rows):
    rows, _ = demo_rows
    cfg = HybridConfig()
    assert {r["profile"] for r in rows} == {"human", "scraper", "credential", "stealth"}
    for r in rows:
        assert r["mode"] == "HYBRID"
        assert r["hybrid"] == pytest.approx(cfg.fsm_weight * r["fsm"] + cfg.ml_weight * r["ml"], abs=1e-4)


def test_demo_scenario_bots_trigger_fsm_rules(demo_rows):
    rows, _ = demo_rows
    by_profile = {r["profile"]: r for r in rows}
    assert by_profile["human"]["fsm"] == 0.0
    for bot in ("scraper", "credential"):
        assert by_profile[bot]["fsm"] > 0.0
        assert by_profile[bot]["reasons"]
        assert by_profile[bot]["fsm"] > by_profile["human"]["fsm"]


def test_demo_scenario_is_labeled_and_persisted_as_simulated(demo_rows, client):
    rows, db = demo_rows
    conn = get_connection(db)
    try:
        labels = {r["session_id"]: r["source"] for r in conn.execute("SELECT * FROM session_labels")}
        stored = {r["session_id"] for r in conn.execute("SELECT session_id FROM session_risks")}
    finally:
        conn.close()
    for r in rows:
        assert labels[r["session_id"]] == f"simulation:{r['profile']}"
        assert r["session_id"] in stored

    headers = {"X-Admin-Token": ADMIN_TOKEN}
    sim = client.get("/admin/api/sessions?traffic=SIMULATED", headers=headers).get_json()
    assert sim["total"] == len(rows)
    assert all(s["traffic_source"]["simulated"] for s in sim["sessions"])
    real = client.get("/admin/api/sessions?traffic=REAL", headers=headers).get_json()
    assert all(not s["traffic_source"]["simulated"] for s in real["sessions"])


def test_simulated_human_telemetry_matches_tracker_format(_use_temp_db):
    """Simulator telemetry is accepted by /api/events and lands after its page request."""
    from backend.database.db import get_events, get_requests
    from backend.detection.fsm import parse_iso_to_epoch
    from backend.simulation.profiles import ProfileType
    from backend.simulation.runner import SimulationConfig, SimulationRunner

    session = SimulationRunner(SimulationConfig(
        mode="in-process", profiles=[ProfileType.HUMAN], sessions_per_profile=1,
        evaluate_detection=False, db_path=_use_temp_db, dwell_scale=1.0,
    )).run().sessions[0]

    conn = get_connection(_use_temp_db)
    try:
        api = [r for r in get_requests(conn, session.session_id, limit=500) if r["path"] == "/api/events"]
        pages = {r["path"]: parse_iso_to_epoch(r["timestamp"])
                 for r in get_requests(conn, session.session_id, limit=500) if r["path"] != "/api/events"}
        events = get_events(conn, session.session_id)
    finally:
        conn.close()

    assert api and all(r["status_code"] == 200 for r in api)  # previously all 400
    assert {e["event_type"] for e in events} >= {"PAGE_VIEW", "CLICK", "SCROLL", "MOUSE_ACTIVITY"}
    for e in events:
        if e["event_type"] == "PAGE_VIEW":
            assert parse_iso_to_epoch(e["timestamp"]) >= pages[e["page"]]
