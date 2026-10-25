"""
BotShield – Hybrid FSM + ML pipeline tests

Covers the live integration added on top of the Phase 4/5 scoring services:
- No placeholder ML scores (invalid features / non-finite output -> ERROR, None)
- Persisted scoring metadata and per-session score history
- Additive migration of older session_risks tables
- /api/risk/current: FSM, ML and hybrid scores for the caller's own session
- Monitor polling is never logged as requests or fed to the automata
- Admin analytics: auth, score comparison, labeled precision/recall/F1/FPR
- Simulator ground-truth labels
"""

import math
import sqlite3

import pytest

import app as app_module
from app import app
from backend.database.db import (
    get_connection,
    get_requests,
    get_risk_history,
    init_db,
    insert_events,
    insert_request,
    upsert_session,
    upsert_session_label,
    upsert_session_risk,
)
from backend.database.stats import get_admin_analytics, get_admin_session_detail
from backend.ml.features import FEATURE_NAMES, ExtractedFeatures
from backend.ml.model import IsolationForestModel, MLModelStatus, train_unsupervised_model
from backend.ml.train import generate_synthetic_baseline
from backend.scoring.service import HybridConfig, ScoringMode, evaluate_session_risk

SESSION = "abcdef01-2345-4678-9abc-def012345678"
ADMIN_TOKEN = "test-admin-token-123"


@pytest.fixture
def db_path(_use_temp_db):
    return _use_temp_db


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "model.joblib"
    train_unsupervised_model(generate_synthetic_baseline(60, random_state=7), model_path=path)
    return str(path)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app_module._risk_cache.clear()
    with app.test_client() as c:
        yield c
    app_module._risk_cache.clear()


def _seed_activity(db, session_id=SESSION):
    # Timestamps are in the past so later "now" activity always sorts after them
    conn = get_connection(db)
    try:
        upsert_session(conn, session_id)
        insert_events(conn, session_id, [
            {"event_type": "PAGE_VIEW", "timestamp": "2020-01-01T10:00:00.000000Z", "page": "/"},
            {"event_type": "SCROLL", "timestamp": "2020-01-01T10:00:03.000000Z", "page": "/", "scroll_depth": 40},
            {"event_type": "CLICK", "timestamp": "2020-01-01T10:00:06.000000Z", "page": "/", "element_type": "a"},
            {"event_type": "MOUSE_ACTIVITY", "timestamp": "2020-01-01T10:00:08.000000Z", "page": "/", "activity_count": 30},
        ])
        for i, path in enumerate(["/", "/products", "/product/1"]):
            insert_request(conn, session_id, "GET", path, 200, timestamp=f"2020-01-01T10:00:0{i * 3}.000000Z")
        conn.commit()
    finally:
        conn.close()


def _features(vector):
    return ExtractedFeatures(
        feature_vector=vector,
        feature_names=FEATURE_NAMES,
        feature_dict={n: v for n, v in zip(FEATURE_NAMES, vector)},
    )


# ---------------------------------------------------------------------------
# ML inference never fabricates a score
# ---------------------------------------------------------------------------

def test_ml_invalid_features_report_error_without_score(model_path):
    vector = generate_synthetic_baseline(1, random_state=1)[0]
    vector[3] = float("nan")
    result = IsolationForestModel(model_path=model_path).predict(_features(vector))

    assert result.status == MLModelStatus.ERROR
    assert result.anomaly_score is None
    assert result.is_anomaly is None
    assert any("INVALID_FEATURES" in w for w in result.warnings)


def test_ml_non_finite_model_output_reports_error(model_path, monkeypatch):
    model = IsolationForestModel(model_path=model_path)
    monkeypatch.setattr(model.bundle["pipeline"], "score_samples", lambda X: [float("nan")])
    result = model.predict(_features(generate_synthetic_baseline(1, random_state=1)[0]))

    assert result.status == MLModelStatus.ERROR
    assert result.anomaly_score is None


def test_ml_available_reports_decision_threshold(model_path):
    result = IsolationForestModel(model_path=model_path).predict(
        _features(generate_synthetic_baseline(1, random_state=1)[0])
    )
    assert result.status == MLModelStatus.AVAILABLE
    threshold = result.model_metadata["decision_threshold"]
    assert 0.0 < threshold < 1.0
    # is_anomaly is the model's own decision on the same s(x) scale
    assert result.is_anomaly == (result.anomaly_score > threshold)


# ---------------------------------------------------------------------------
# Persistence: metadata, history, migration
# ---------------------------------------------------------------------------

def test_persisted_hybrid_result_stores_mode_status_and_history(db_path, model_path):
    _seed_activity(db_path)
    result = evaluate_session_risk(SESSION, db_path=db_path, model_path=model_path, persist=True)

    cfg = HybridConfig()
    assert result.scoring_mode == ScoringMode.HYBRID
    expected = cfg.fsm_weight * result.fsm_score + cfg.ml_weight * result.ml_score
    assert result.normalized_score == pytest.approx(round(expected, 4))

    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM session_risks WHERE session_id = ?", (SESSION,)).fetchone()
        assert row["scoring_mode"] == "HYBRID"
        assert row["ml_status"] == "available"
        assert row["fsm_state"] == result.fsm_result.final_state.value
        assert bool(row["ml_is_anomaly"]) == result.ml_result.is_anomaly

        history = get_risk_history(conn, SESSION)
        assert len(history) == 1
        assert history[0]["hybrid_score"] == pytest.approx(result.normalized_score)
        assert history[0]["ml_score"] == pytest.approx(result.ml_score)
    finally:
        conn.close()


def test_fsm_only_result_is_not_reported_as_ml_zero(db_path, tmp_path):
    _seed_activity(db_path)
    missing = str(tmp_path / "missing.joblib")
    result = evaluate_session_risk(SESSION, db_path=db_path, model_path=missing, persist=True)
    assert result.scoring_mode == ScoringMode.FSM_ONLY
    assert result.ml_score is None

    conn = get_connection(db_path)
    try:
        detail = get_admin_session_detail(conn, SESSION)
        history = get_risk_history(conn, SESSION)
    finally:
        conn.close()

    ra = detail["risk_assessment"]
    assert ra["scoring_mode"] == "FSM_ONLY"
    assert ra["ml_status"] == "untrained"
    assert ra["ml_score"] is None  # stored column holds 0.0, but it is not an ML score
    assert history[0]["ml_score"] is None
    assert detail["risk_history"][0]["scoring_mode"] == "FSM_ONLY"


def test_migration_adds_columns_to_existing_session_risks(tmp_path):
    legacy = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(legacy)
    conn.executescript(
        """
        CREATE TABLE sessions (session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, last_seen TEXT NOT NULL);
        CREATE TABLE session_risks (
            session_id TEXT PRIMARY KEY,
            risk_score REAL NOT NULL DEFAULT 0.0,
            fsm_score REAL NOT NULL DEFAULT 0.0,
            ml_score REAL NOT NULL DEFAULT 0.0,
            reasons TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL DEFAULT ''
        );
        INSERT INTO sessions VALUES ('s1', 't', 't');
        INSERT INTO session_risks (session_id, risk_score, fsm_score, ml_score) VALUES ('s1', 0.5, 0.4, 0.6);
        """
    )
    conn.commit()
    conn.close()

    init_db(legacy)
    init_db(legacy)  # idempotent

    conn = get_connection(legacy)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(session_risks)")}
        assert {"scoring_mode", "ml_status", "fsm_state", "ml_is_anomaly"} <= cols
        row = conn.execute("SELECT * FROM session_risks WHERE session_id = 's1'").fetchone()
        assert (row["risk_score"], row["fsm_score"], row["ml_score"]) == (0.5, 0.4, 0.6)
        assert row["scoring_mode"] is None
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"session_risk_history", "session_labels"} <= tables
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# /api/risk/current (Monitor panel)
# ---------------------------------------------------------------------------

def test_current_risk_without_session(client):
    data = client.get("/api/risk/current").get_json()
    assert data["available"] is False
    assert data["session_id"] is None


def test_current_risk_unknown_session_is_not_evaluated(client):
    client.set_cookie("botshield_session_id", SESSION)
    data = client.get("/api/risk/current").get_json()
    assert data["available"] is False
    assert data["session_id"] == SESSION


def test_current_risk_matches_backend_hybrid_calculation(client, db_path, model_path):
    app.config["BOTSHIELD_MODEL_PATH"] = model_path
    _seed_activity(db_path)
    client.set_cookie("botshield_session_id", SESSION)

    data = client.get("/api/risk/current").get_json()
    backend = evaluate_session_risk(SESSION, db_path=db_path, model_path=model_path)

    assert data["available"] is True
    assert data["ml"]["active"] is True
    assert data["ml"]["status"] == "available"
    assert data["hybrid"]["scoring_mode"] == "HYBRID"
    assert data["fsm"]["score"] == backend.fsm_score
    assert data["ml"]["score"] == backend.ml_score
    assert data["hybrid"]["score"] == backend.normalized_score
    w = data["hybrid"]["weights"]
    assert data["hybrid"]["score"] == pytest.approx(
        w["fsm"] * data["fsm"]["score"] + w["ml"] * data["ml"]["score"], abs=1e-4
    )
    assert data["fsm"]["state"] == backend.fsm_result.final_state.value
    # Server paths in model warnings are never exposed publicly
    assert "warnings" not in data


def test_current_risk_reports_ml_unavailable_without_placeholder(client, db_path, tmp_path):
    app.config["BOTSHIELD_MODEL_PATH"] = str(tmp_path / "missing.joblib")
    _seed_activity(db_path)
    client.set_cookie("botshield_session_id", SESSION)

    data = client.get("/api/risk/current").get_json()
    assert data["available"] is True
    assert data["ml"]["active"] is False
    assert data["ml"]["status"] == "untrained"
    assert data["ml"]["score"] is None
    assert data["hybrid"]["scoring_mode"] == "FSM_ONLY"
    assert data["hybrid"]["score"] == data["fsm"]["score"]


def test_monitor_polling_not_logged_or_fed_to_automata(client, db_path):
    from automata.history import clear_session_context, find_session_context

    clear_session_context(SESSION)
    _seed_activity(db_path)
    conn = get_connection(db_path)
    before = len(get_requests(conn, session_id=None, limit=5000))
    conn.close()

    client.set_cookie("botshield_session_id", SESSION)
    client.set_cookie("botshield_session", SESSION)
    for _ in range(5):
        app_module._risk_cache.clear()
        assert client.get("/api/risk/current").status_code == 200

    conn = get_connection(db_path)
    after = len(get_requests(conn, session_id=None, limit=5000))
    conn.close()
    assert after == before
    assert find_session_context(SESSION) is None


def test_current_risk_evaluation_failure_is_safe(client, db_path, monkeypatch):
    _seed_activity(db_path)
    client.set_cookie("botshield_session_id", SESSION)

    def boom(*args, **kwargs):
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr(app_module, "evaluate_session_risk", boom)
    resp = client.get("/api/risk/current")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["available"] is False
    assert "exploded" not in data["message"]


def test_request_only_activity_requeues_session_for_scoring(db_path):
    from backend.scoring.worker import get_pending_sessions

    _seed_activity(db_path)
    evaluate_session_risk(SESSION, db_path=db_path, persist=True)
    conn = get_connection(db_path)
    try:
        assert SESSION not in get_pending_sessions(conn)
        # A new page request without any browser telemetry (last_seen unchanged)
        insert_request(conn, SESSION, "GET", "/products", 200)
        conn.commit()
        assert SESSION in get_pending_sessions(conn)
    finally:
        conn.close()


def test_admin_api_polling_not_logged_against_visitor_session(client, db_path):
    app.config["ADMIN_TOKEN"] = ADMIN_TOKEN
    _seed_activity(db_path)
    conn = get_connection(db_path)
    before = len(get_requests(conn, session_id=SESSION, limit=5000))
    conn.close()

    client.set_cookie("botshield_session", SESSION)
    for endpoint in ("/admin/api/stats", "/admin/api/sessions", "/admin/api/analytics"):
        assert client.get(endpoint, headers={"X-Admin-Token": ADMIN_TOKEN}).status_code == 200
    client.get("/about")  # storefront pages are still logged

    conn = get_connection(db_path)
    paths = [r["path"] for r in get_requests(conn, session_id=SESSION, limit=5000)]
    conn.close()
    assert len(paths) == before + 1
    assert paths[-1] == "/about"


# ---------------------------------------------------------------------------
# Admin analytics
# ---------------------------------------------------------------------------

def test_analytics_requires_admin_auth(client):
    app.config["ADMIN_TOKEN"] = ADMIN_TOKEN
    assert client.get("/admin/api/analytics").status_code == 401
    resp = client.get("/admin/api/analytics", headers={"X-Admin-Token": ADMIN_TOKEN})
    assert resp.status_code == 200


def test_analytics_reports_model_status_and_offline_metrics(client, model_path):
    app.config["ADMIN_TOKEN"] = ADMIN_TOKEN
    app.config["BOTSHIELD_MODEL_PATH"] = model_path
    data = client.get("/admin/api/analytics", headers={"X-Admin-Token": ADMIN_TOKEN}).get_json()

    assert data["model"]["active"] is True
    assert data["model"]["status"] == "available"
    assert 0.0 < data["model"]["decision_threshold"] < 1.0
    metrics = data["offline_evaluation"]["results"]["metrics"]
    assert set(metrics) >= {"precision", "recall", "f1_score", "false_positive_rate"}
    assert data["offline_evaluation"]["real_world_accuracy_claim"] is False
    assert "0.60" in data["formula"] and "0.40" in data["formula"]


def test_analytics_without_model_says_ml_inactive(client, tmp_path):
    app.config["ADMIN_TOKEN"] = ADMIN_TOKEN
    app.config["BOTSHIELD_MODEL_PATH"] = str(tmp_path / "missing.joblib")
    data = client.get("/admin/api/analytics", headers={"X-Admin-Token": ADMIN_TOKEN}).get_json()
    assert data["model"]["active"] is False
    assert data["model"]["status"] == "untrained"
    assert data["offline_evaluation"] is None
    assert data["analytics"]["averages"]["ml"] is None


def test_labeled_metrics_and_detection_counts(db_path):
    # (session, is_bot, fsm, ml, hybrid, ml_is_anomaly)
    cases = [
        ("bot-1", 1, 0.90, 0.70, 0.82, True),   # all detectors: TP
        ("bot-2", 1, 0.20, 0.70, 0.40, True),   # FSM FN, ML TP, hybrid FN
        ("human-1", 0, 0.10, 0.40, 0.22, False),  # all TN
        ("human-2", 0, 0.80, 0.65, 0.74, True),   # all FP
    ]
    conn = get_connection(db_path)
    try:
        for sid, is_bot, fsm, ml, hyb, anomaly in cases:
            upsert_session(conn, sid)
            upsert_session_risk(conn, sid, risk_score=hyb, fsm_score=fsm, ml_score=ml,
                                scoring_mode="HYBRID", ml_status="available", ml_is_anomaly=anomaly)
            upsert_session_label(conn, sid, is_bot=bool(is_bot), source="test")
        # Unlabeled and evidence-free sessions are excluded where documented
        upsert_session(conn, "empty")
        upsert_session_risk(conn, "empty", risk_score=0.0, fsm_score=0.0,
                            scoring_mode="INSUFFICIENT_EVIDENCE", ml_status="available")
        conn.commit()
        a = get_admin_analytics(conn)
    finally:
        conn.close()

    assert a["evaluated_with_evidence"] == 4
    assert a["detections"] == {"fsm": 2, "ml": 3, "hybrid": 2, "fsm_and_ml": 2}
    assert a["averages"]["fsm"] == pytest.approx(0.5)
    assert sum(a["histograms"]["hybrid"]) == 4

    fsm = a["labeled_metrics"]["fsm"]
    assert (fsm["tp"], fsm["fp"], fsm["tn"], fsm["fn"]) == (1, 1, 1, 1)
    assert fsm["precision"] == 0.5 and fsm["recall"] == 0.5 and fsm["false_positive_rate"] == 0.5

    ml = a["labeled_metrics"]["ml"]
    assert (ml["tp"], ml["fp"], ml["tn"], ml["fn"]) == (2, 1, 1, 0)
    assert ml["recall"] == 1.0
    assert ml["f1_score"] == pytest.approx(round(2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0), 4))
    assert a["labels"] == {"total": 4, "bots": 2, "humans": 2, "evaluated": 4}


def test_simulator_records_ground_truth_labels(db_path, model_path):
    from backend.simulation.profiles import ProfileType
    from backend.simulation.runner import SimulationConfig, SimulationRunner

    summary = SimulationRunner(SimulationConfig(
        mode="in-process",
        profiles=[ProfileType.HUMAN, ProfileType.RAPID_SCRAPER],
        sessions_per_profile=1,
        evaluate_detection=False,
        db_path=db_path,
        model_path=model_path,
        dwell_scale=0.0,
    )).run()

    conn = get_connection(db_path)
    try:
        labels = {r["session_id"]: (r["is_bot"], r["source"])
                  for r in conn.execute("SELECT * FROM session_labels")}
    finally:
        conn.close()

    by_profile = {s.profile_type: s.session_id for s in summary.sessions}
    assert labels[by_profile["human"]] == (0, "simulation:human")
    assert labels[by_profile["scraper"]] == (1, "simulation:scraper")
