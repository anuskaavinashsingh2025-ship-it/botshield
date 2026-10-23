"""
BotShield – Phase 9 End-to-End Integration and System Validation Tests

Validates the complete operational lifecycle across all subsystems:
Browser Telemetry -> Flask Request Logging -> SQLite Persistence ->
FSM Evaluation + ML Evaluation -> Hybrid Risk Scoring -> Risk Persistence ->
Enforcement Decisions (Monitor & Enforce) -> Security Admin Dashboard & APIs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from flask import Flask, g

from app import app
from backend.database.db import (
    get_connection,
    get_events,
    get_requests,
    get_session,
    get_session_risk,
    init_db,
    insert_events,
    insert_request,
    upsert_session,
    upsert_session_risk,
)
from backend.detection.service import evaluate_session_fsm
from backend.enforcement import (
    EnforcementAction,
    EnforcementConfig,
    EnforcementDecision,
    EnforcementMode,
    EnforcementPolicy,
    EnforcementService,
)
from backend.ml.evaluation import evaluate_model_pipeline
from backend.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from backend.ml.model import (
    IsolationForestModel,
    MLModelStatus,
    get_model_path,
    train_unsupervised_model,
)
from backend.ml.service import evaluate_session_ml
from backend.ml.train import generate_synthetic_baseline
from backend.scoring import (
    HybridConfig,
    HybridRiskResult,
    RiskLevel,
    ScoringMode,
    evaluate_session_risk,
)

ADMIN_TEST_TOKEN = "e2e-secret-admin-token-xyz789"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provide an isolated, freshly initialized SQLite database."""
    db_file = str(tmp_path / "botshield_e2e.db")
    monkeypatch.setenv("BOTSHIELD_DB", db_file)
    init_db(db_file)
    return db_file


@pytest.fixture
def trained_model_path(tmp_path):
    """Train a reproducible synthetic model bundle in an isolated temp path."""
    model_file = str(tmp_path / "models" / "e2e_isolation_forest.joblib")
    vectors = generate_synthetic_baseline(sample_count=80, random_state=42)
    train_unsupervised_model(
        training_data=vectors,
        model_path=model_file,
        contamination=0.10,
        random_state=42,
        dataset_type="e2e_synthetic_test",
    )
    return model_file


@pytest.fixture
def client(temp_db, monkeypatch):
    """Configure Flask application client for end-to-end integration tests."""
    app.config["TESTING"] = True
    app.config["BOTSHIELD_DB"] = temp_db
    app.config["ADMIN_TOKEN"] = ADMIN_TEST_TOKEN
    app.config["REQUEST_LOGGING_ENABLED"] = True
    app.config["REQUEST_LOGGING_EXCLUDE_STATIC"] = True
    app.config["REQUEST_LOGGING_EXCLUDE_PATHS"] = {"/favicon.ico", "/api/events"}
    app.config["ENFORCEMENT_MODE"] = "monitor"
    app.config["ENFORCEMENT_FAIL_OPEN"] = True
    app.config["ENFORCEMENT_SENSITIVE_ROUTES"] = ("/login",)
    app.config["ENFORCEMENT_EXCLUDED_ROUTES"] = ("/static", "/favicon.ico", "/api/events")
    app.config["ENFORCEMENT_RATE_LIMIT_REQUESTS"] = 2
    app.config["ENFORCEMENT_RATE_LIMIT_WINDOW"] = 10.0
    app.config["ENFORCEMENT_CACHE_TTL"] = 60.0

    # Ensure active enforcement service uses temp_db and monitor mode
    enf_service = app.extensions.get("botshield_enforcement")
    if enf_service is not None:
        enf_service.config = EnforcementConfig(
            mode=EnforcementMode.MONITOR,
            fail_open=True,
            sensitive_routes=("/login",),
            excluded_routes=("/static", "/favicon.ico", "/api/events"),
            rate_limit_requests=2,
            rate_limit_window_seconds=10.0,
            risk_cache_ttl_seconds=60.0,
        )
        enf_service.limiter.reset()

    with app.test_client() as test_client:
        yield test_client

    # Reset enforcement limiter after test
    if enf_service is not None:
        enf_service.limiter.reset()


# ==============================================================================
# Scenario A: Normal Human Session Lifecycle
# ==============================================================================

def test_e2e_normal_human_session_lifecycle(client, temp_db, trained_model_path):
    """
    Test the full end-to-end lifecycle of a legitimate human browsing session:
    1. Browser telemetry submitted via POST /api/events
    2. HTTP requests issued and correlated via session cookie
    3. Events and requests verified in SQLite
    4. FSM evaluation detects normal navigation (score 0.0)
    5. ML evaluation classifies activity as normal baseline
    6. Hybrid scoring evaluates to LOW risk (HYBRID mode) and persists
    7. Enforcement allows requests to sensitive routes
    8. Authenticated admin API reports session and risk diagnostics
    """
    session_id = "aaaaaaaa-1111-4444-8888-000000000001"

    # 1. Client sends realistic browser telemetry batch
    telemetry_payload = {
        "session_id": session_id,
        "events": [
            {
                "event_type": "PAGE_VIEW",
                "timestamp": "2026-10-24T00:00:00Z",
                "url": "http://localhost/products",
                "metadata": {"path": "/products"},
            },
            {
                "event_type": "MOUSE_ACTIVITY",
                "timestamp": "2026-10-24T00:00:02Z",
                "url": "http://localhost/products",
                "metadata": {"moves": 24},
            },
            {
                "event_type": "SCROLL",
                "timestamp": "2026-10-24T00:00:04Z",
                "url": "http://localhost/products",
                "metadata": {"depth_px": 450},
            },
            {
                "event_type": "CLICK",
                "timestamp": "2026-10-24T00:00:06Z",
                "url": "http://localhost/products",
                "metadata": {"target": "a.product-card", "text": "Wireless Mouse"},
            },
            {
                "event_type": "KEYBOARD_ACTIVITY",
                "timestamp": "2026-10-24T00:00:10Z",
                "url": "http://localhost/product/1",
                "metadata": {"key_count": 5},
            },
        ],
    }
    resp = client.post("/api/events", json=telemetry_payload)
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
    assert resp.get_json()["stored"] == 5

    # 2. Client issues normal HTTP browsing requests using session cookie
    client.set_cookie("botshield_session", session_id)
    r1 = client.get("/products")
    assert r1.status_code == 200
    time.sleep(0.20)  # Human browsing pause (>150ms rapid traversal threshold)
    r2 = client.get("/product/1")
    assert r2.status_code == 200

    # 3. Verify SQLite persistence
    conn = get_connection(temp_db)
    sess_row = get_session(conn, session_id)
    events = get_events(conn, session_id)
    requests = get_requests(conn, session_id)
    conn.close()

    assert sess_row is not None
    assert len(events) == 5
    assert len(requests) == 2
    assert requests[0]["path"] == "/products"
    assert requests[1]["path"] == "/product/1"

    # 4. FSM evaluation
    fsm_res = evaluate_session_fsm(session_id=session_id, db_path=temp_db)
    assert fsm_res.risk_score == 0.0
    assert len(fsm_res.anomaly_codes) == 0

    # 5. ML evaluation
    ml_res = evaluate_session_ml(
        session_id=session_id,
        db_path=temp_db,
        model_path=trained_model_path,
    )
    assert ml_res.status == MLModelStatus.AVAILABLE
    assert ml_res.anomaly_score is not None
    assert 0.0 <= ml_res.anomaly_score <= 1.0

    # 6. Hybrid risk scoring with persistence
    hybrid_res = evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path=trained_model_path,
        persist=True,
    )
    assert hybrid_res.scoring_mode == ScoringMode.HYBRID
    assert hybrid_res.risk_level == RiskLevel.LOW
    assert hybrid_res.composite_score < 35.0
    assert hybrid_res.fsm_score == 0.0
    assert hybrid_res.ml_score is not None

    # Verify risk row in database
    conn = get_connection(temp_db)
    risk_row = get_session_risk(conn, session_id)
    conn.close()
    assert risk_row is not None
    assert risk_row["risk_score"] == pytest.approx(hybrid_res.normalized_score, abs=1e-3)
    assert risk_row["fsm_score"] == 0.0

    # 7. Enforcement check on sensitive route (/login)
    login_resp = client.get("/login")
    assert login_resp.status_code == 200

    # 8. Admin API verification
    admin_headers = {"Authorization": f"Bearer {ADMIN_TEST_TOKEN}"}
    sess_api_resp = client.get(f"/admin/api/sessions/{session_id}", headers=admin_headers)
    assert sess_api_resp.status_code == 200
    data = sess_api_resp.get_json()
    assert data["session_id"] == session_id
    assert data["risk_level"] == "LOW"
    assert data["evaluation_mode"] == "HYBRID"
    assert data["event_count"] == 5
    assert data["request_count"] == 3

    stats_resp = client.get("/admin/api/stats", headers=admin_headers)
    assert stats_resp.status_code == 200
    stats = stats_resp.get_json()
    assert stats["total_sessions"] >= 1
    assert stats["risk_distribution"]["low"] >= 1
    assert stats["scoring_modes"]["HYBRID"] >= 1


# ==============================================================================
# Scenario B: Suspicious Automated Session
# ==============================================================================

def test_e2e_suspicious_automated_bot_session(client, temp_db, trained_model_path):
    """
    Test a suspicious automated scraper session:
    1. Register session in database
    2. Directly issue high-velocity requests in sub-100ms intervals with 0 telemetry
    3. FSM identifies RAPID_TRAVERSAL and/or HIGH_SCRAPER_RATIO
    4. Hybrid evaluation scores session as HIGH risk
    5. Persisted in session_risks
    6. Admin API reports session as HIGH risk with explanatory factors
    """
    session_id = "bbbbbbbb-2222-4444-8888-000000000002"

    conn = get_connection(temp_db)
    upsert_session(conn, session_id)

    # Insert 10 rapid server requests occurring within 0.5 seconds
    base_epoch = 1700000000.0
    rapid_paths = ["/products", "/product/1", "/products", "/product/2", "/login", "/about", "/products", "/product/3", "/login", "/products"]
    for i, path in enumerate(rapid_paths):
        ts = datetime.fromtimestamp(base_epoch + (i * 0.05), tz=timezone.utc).isoformat()
        insert_request(
            conn,
            session_id=session_id,
            method="GET",
            path=path,
            status_code=200,
            timestamp=ts,
            user_agent="BotShieldHeadlessCrawler/2.0",
        )
    conn.commit()
    conn.close()

    # FSM Evaluation
    fsm_res = evaluate_session_fsm(session_id=session_id, db_path=temp_db)
    assert fsm_res.risk_score >= 0.70
    reasons_str = " ".join(fsm_res.anomaly_codes)
    assert "ANOMALY_RAPID_TRAVERSAL" in reasons_str or "ANOMALY_HIGH_REQ_LOW_TELEMETRY" in reasons_str or "ANOMALY_PATH_CYCLING" in reasons_str

    # Hybrid Risk Evaluation
    hybrid_res = evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path=trained_model_path,
        persist=True,
    )
    assert hybrid_res.risk_level == RiskLevel.HIGH
    assert hybrid_res.composite_score >= 70.0

    # In default monitor mode, requests still succeed without blocking
    client.set_cookie("botshield_session", session_id)
    login_resp = client.get("/login")
    assert login_resp.status_code == 200

    # Admin API reports HIGH risk and reasons
    admin_headers = {"Authorization": f"Bearer {ADMIN_TEST_TOKEN}"}
    sess_resp = client.get(f"/admin/api/sessions/{session_id}", headers=admin_headers)
    assert sess_resp.status_code == 200
    data = sess_resp.get_json()["session"]
    assert data["risk_assessment"]["risk_level"] == "HIGH"
    assert data["risk_assessment"]["composite_score"] >= 70.0
    assert len(data["risk_assessment"]["reasons"]) > 0

    stats_resp = client.get("/admin/api/stats", headers=admin_headers)
    assert stats_resp.status_code == 200
    assert stats_resp.get_json()["risk_distribution"]["high"] >= 1


# ==============================================================================
# Scenario C: Insufficient Evidence
# ==============================================================================

def test_e2e_insufficient_evidence_session(client, temp_db, trained_model_path):
    """
    Test a session with no recorded telemetry or requests:
    - Expected mode: INSUFFICIENT_EVIDENCE
    - Score: 0.0, RiskLevel: LOW
    - Warning: INSUFFICIENT_SESSION_EVIDENCE
    - Enforcement safe ALLOW behavior
    """
    session_id = "cccccccc-3333-4444-8888-000000000003"

    conn = get_connection(temp_db)
    upsert_session(conn, session_id)
    conn.commit()
    conn.close()

    res = evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path=trained_model_path,
        persist=True,
    )
    assert res.scoring_mode == ScoringMode.INSUFFICIENT_EVIDENCE
    assert res.composite_score == 0.0
    assert res.normalized_score == 0.0
    assert res.risk_level == RiskLevel.LOW
    assert "INSUFFICIENT_SESSION_EVIDENCE" in " ".join(res.warnings)

    # Enforcement allows requests safely
    client.set_cookie("botshield_session", session_id)
    resp = client.get("/login")
    assert resp.status_code == 200

    # Admin API shows session with zero counts
    admin_headers = {"Authorization": f"Bearer {ADMIN_TEST_TOKEN}"}
    api_resp = client.get(f"/admin/api/sessions/{session_id}", headers=admin_headers)
    assert api_resp.status_code == 200
    sess = api_resp.get_json()["session"]
    assert sess["risk_assessment"]["composite_score"] == 0.0
    assert sess["risk_assessment"]["risk_level"] == "LOW"


# ==============================================================================
# Scenario D: ML Unavailable or Untrained Fallback
# ==============================================================================

def test_e2e_ml_unavailable_or_untrained_fallback(client, temp_db, tmp_path):
    """
    Test behavior when ML model file is missing or untrained:
    - Application remains fully operational
    - Safe fallback to FSM_ONLY scoring mode
    - ML score is None (not falsely reported as zero or available)
    - Diagnostics explain fallback mode
    - Admin dashboard correctly tallies FSM_ONLY session
    """
    session_id = "dddddddd-4444-4444-8888-000000000004"
    missing_model_file = str(tmp_path / "models" / "nonexistent_model.joblib")

    # Record legitimate browsing requests
    conn = get_connection(temp_db)
    upsert_session(conn, session_id)
    insert_request(conn, session_id, "GET", "/products", 200, "2026-10-24T00:00:00Z")
    insert_events(conn, session_id, [
        {"event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:00:02Z", "url": "http://localhost/products"},
        {"event_type": "CLICK", "timestamp": "2026-10-24T00:00:06Z", "url": "http://localhost/products"},
    ])
    insert_request(conn, session_id, "GET", "/product/2", 200, "2026-10-24T00:00:15Z")
    conn.commit()
    conn.close()

    # ML service directly confirms UNTRAINED
    ml_res = evaluate_session_ml(session_id=session_id, db_path=temp_db, model_path=missing_model_file)
    assert ml_res.status == MLModelStatus.UNTRAINED
    assert ml_res.anomaly_score is None

    # Hybrid evaluation safely falls back to FSM_ONLY
    hybrid_res = evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path=missing_model_file,
        persist=True,
    )
    assert hybrid_res.scoring_mode == ScoringMode.FSM_ONLY
    assert hybrid_res.ml_score is None
    assert hybrid_res.fsm_score == 0.0
    assert hybrid_res.risk_level == RiskLevel.LOW
    assert "FSM-only fallback mode" in " ".join(hybrid_res.warnings)

    # Admin API reports session evaluated under FSM_ONLY
    admin_headers = {"Authorization": f"Bearer {ADMIN_TEST_TOKEN}"}
    stats_resp = client.get("/admin/api/stats", headers=admin_headers)
    assert stats_resp.status_code == 200
    assert stats_resp.get_json()["scoring_modes"]["FSM_ONLY"] >= 1


# ==============================================================================
# Scenario E: Stale and Missing Risk Policy
# ==============================================================================

def test_e2e_stale_and_missing_risk_policy(temp_db):
    """
    Verify enforcement decision service safely handles missing and stale risk records:
    - Missing risk record -> default ALLOW
    - Stale risk record (older than TTL) -> default ALLOW
    """
    enf_service = EnforcementService(
        config=EnforcementConfig(
            mode=EnforcementMode.ENFORCE,
            risk_cache_ttl_seconds=60.0,
            fail_open=True,
        )
    )

    # 1. Missing risk record: session exists in DB but no row in session_risks
    session_id_missing = "eeeeeeee-5555-4444-8888-000000000005"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id_missing)
    conn.commit()
    conn.close()

    decision_missing = enf_service.evaluate_request(
        session_id=session_id_missing,
        client_ip="192.168.1.10",
        path="/login",
        db_path=temp_db,
    )
    assert decision_missing.action == EnforcementAction.ALLOW
    assert "No existing risk record found" in decision_missing.reason

    # 2. Stale risk record: persisted 2 hours ago (TTL is 60s)
    session_id_stale = "ffffffff-6666-4444-8888-000000000006"
    stale_ts = datetime.fromtimestamp(time.time() - 7200.0, tz=timezone.utc).isoformat()
    conn = get_connection(temp_db)
    upsert_session(conn, session_id_stale)
    upsert_session_risk(
        conn,
        session_id=session_id_stale,
        risk_score=0.95,
        fsm_score=0.90,
        ml_score=0.95,
        reasons=["RAPID_TRAVERSAL"],
        updated_at=stale_ts,
    )
    conn.commit()
    conn.close()

    decision_stale = enf_service.evaluate_request(
        session_id=session_id_stale,
        client_ip="192.168.1.11",
        path="/login",
        db_path=temp_db,
    )
    assert decision_stale.action == EnforcementAction.ALLOW
    assert "stale" in decision_stale.reason.lower()


# ==============================================================================
# Scenario F: Enforcement Modes (Monitor vs Active Enforce)
# ==============================================================================

def test_e2e_enforcement_monitor_mode_never_blocks(client, temp_db):
    """
    In MONITOR mode (default), even HIGH-risk sessions are observed and logged,
    never blocked or restricted.
    """
    session_id = "11111111-7777-4444-8888-000000000007"

    conn = get_connection(temp_db)
    upsert_session(conn, session_id)
    upsert_session_risk(conn, session_id=session_id, risk_score=0.95, fsm_score=0.95, ml_score=0.95, reasons=["RAPID_TRAVERSAL"])
    conn.commit()
    conn.close()

    client.set_cookie("botshield_session", session_id)
    resp = client.get("/login")
    assert resp.status_code == 200
    assert "Login" in resp.get_data(as_text=True)


def test_e2e_active_enforcement_mode_isolated(client, temp_db):
    """
    Verify active ENFORCE mode in isolated test execution:
    - HIGH risk -> HTTP 403 Forbidden on sensitive route (/login)
    - LOW risk -> HTTP 200 OK on sensitive route (/login)
    - Excluded routes (/static, /api/events) always allowed even for HIGH risk
    - MEDIUM risk -> Rate limited upon quota exhaustion (HTTP 429 with Retry-After)
    """
    enf_service = app.extensions["botshield_enforcement"]
    original_config = enf_service.config
    try:
        # Set active ENFORCE mode
        enf_service.config = EnforcementConfig(
            mode=EnforcementMode.ENFORCE,
            fail_open=True,
            sensitive_routes=("/login",),
            excluded_routes=("/static", "/favicon.ico", "/api/events"),
            rate_limit_requests=2,
            rate_limit_window_seconds=10.0,
            risk_cache_ttl_seconds=60.0,
        )
        enf_service.limiter.reset()

        # 1. HIGH risk session
        high_sess = "22222222-8888-4444-8888-000000000008"
        conn = get_connection(temp_db)
        upsert_session(conn, high_sess)
        upsert_session_risk(conn, session_id=high_sess, risk_score=0.90, fsm_score=0.90, reasons=["CONFIRMED_SCRAPER"])
        conn.commit()
        conn.close()

        client.set_cookie("botshield_session", high_sess)
        r_high = client.get("/login")
        assert r_high.status_code == 403
        assert "Access Restricted" in r_high.get_data(as_text=True)

        # Excluded route (/api/events) still accessible
        r_excl = client.post("/api/events", json={"session_id": high_sess, "events": [{"event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:00:00Z", "url": "http://localhost/"}]})
        assert r_excl.status_code == 200

        # 2. LOW risk session
        low_sess = "33333333-9999-4444-8888-000000000009"
        conn = get_connection(temp_db)
        upsert_session(conn, low_sess)
        upsert_session_risk(conn, session_id=low_sess, risk_score=0.10, fsm_score=0.10, reasons=[])
        conn.commit()
        conn.close()

        client.set_cookie("botshield_session", low_sess)
        r_low = client.get("/login")
        assert r_low.status_code == 200

        # 3. MEDIUM risk rate limiting
        med_sess = "44444444-aaaa-4444-8888-000000000010"
        conn = get_connection(temp_db)
        upsert_session(conn, med_sess)
        upsert_session_risk(conn, session_id=med_sess, risk_score=0.50, fsm_score=0.50, reasons=["SUSPICIOUS_TIMING"])
        conn.commit()
        conn.close()

        client.set_cookie("botshield_session", med_sess)
        # Limit is 2 requests per window
        req1 = client.get("/login")
        assert req1.status_code == 200
        req2 = client.get("/login")
        assert req2.status_code == 200
        req3 = client.get("/login")
        assert req3.status_code == 429
        assert "Retry-After" in req3.headers

    finally:
        # Always restore original configuration
        enf_service.config = original_config
        enf_service.limiter.reset()


# ==============================================================================
# Scenario G: Security Administrator Access Control
# ==============================================================================

def test_e2e_admin_access_control(client, temp_db):
    """
    Verify administrative authentication and fail-closed security:
    - Unauthenticated API call -> 401
    - Invalid token -> 401
    - Valid Bearer token -> 200
    - Form login with valid token sets cookie and redirects
    - Unconfigured server token fails closed (403)
    """
    # 1. Unauthenticated API access
    r_unauth = client.get("/admin/api/stats")
    assert r_unauth.status_code == 401
    assert r_unauth.get_json()["error"] == "Unauthorized"

    # 2. Invalid token
    r_invalid = client.get("/admin/api/stats", headers={"Authorization": "Bearer wrong-token"})
    assert r_invalid.status_code == 401

    # 3. Valid token via Authorization header
    r_valid = client.get("/admin/api/stats", headers={"Authorization": f"Bearer {ADMIN_TEST_TOKEN}"})
    assert r_valid.status_code == 200
    assert "total_sessions" in r_valid.get_json()

    # 4. Form login sets cookie and redirects
    r_login = client.post("/admin/login", data={"token": ADMIN_TEST_TOKEN}, follow_redirects=False)
    assert r_login.status_code == 302
    assert "botshield_admin_token" in r_login.headers.get("Set-Cookie", "")

    # Access dashboard with cookie
    r_dash = client.get("/admin/dashboard")
    assert r_dash.status_code == 200
    assert "Security Operations Center" in r_dash.get_data(as_text=True)

    # 5. Fail-closed when server token is unconfigured
    orig_token = app.config.get("ADMIN_TOKEN")
    try:
        app.config["ADMIN_TOKEN"] = None
        with patch.dict(os.environ, {}, clear=True):
            r_unconf = client.get("/admin/api/stats", headers={"Authorization": f"Bearer {ADMIN_TEST_TOKEN}"})
            assert r_unconf.status_code == 403
    finally:
        app.config["ADMIN_TOKEN"] = orig_token


# ==============================================================================
# Scenario H: Persistence and Database Restart Behavior
# ==============================================================================

def test_e2e_persistence_across_connection_restarts(temp_db):
    """
    Verify that sessions, events, requests, and risk assessments survive
    connection closure and restart without depending on in-memory state.
    """
    session_id = "55555555-bbbb-4444-8888-000000000011"

    # 1. Open connection, insert complete records, evaluate risk, close
    conn1 = get_connection(temp_db)
    upsert_session(conn1, session_id)
    insert_events(conn1, session_id, [
        {"event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:00:00Z", "url": "http://localhost/"},
        {"event_type": "CLICK", "timestamp": "2026-10-24T00:00:05Z", "url": "http://localhost/"},
    ])
    insert_request(conn1, session_id, "GET", "/", 200, "2026-10-24T00:00:00Z")
    upsert_session_risk(
        conn1,
        session_id=session_id,
        risk_score=0.25,
        fsm_score=0.20,
        ml_score=0.30,
        reasons=["NORMAL_BROWSING"],
    )
    conn1.commit()
    conn1.close()

    # 2. Open completely fresh, independent connection simulating process restart
    conn2 = get_connection(temp_db)
    sess = get_session(conn2, session_id)
    events = get_events(conn2, session_id)
    reqs = get_requests(conn2, session_id)
    risk = get_session_risk(conn2, session_id)
    conn2.close()

    assert sess is not None
    assert sess["session_id"] == session_id
    assert len(events) == 2
    assert len(reqs) == 1
    assert risk is not None
    assert risk["risk_score"] == 0.25
    assert risk["fsm_score"] == 0.20
    assert risk["ml_score"] == 0.30
    assert json.loads(risk["reasons"]) == ["NORMAL_BROWSING"]


# ==============================================================================
# Scenario I: ML Artifact & Evaluation Integrity
# ==============================================================================

def test_e2e_ml_artifact_and_evaluation_integrity(tmp_path, trained_model_path):
    """
    Verify Phase 8/9 artifact integrity requirements:
    1. Evaluating nonexistent file raises FileNotFoundError
    2. Evaluating corrupt file raises RuntimeError with CORRUPT status
    3. Evaluating valid trained model returns artifact path, metadata, and synthetic provenance
    """
    # 1. Nonexistent model file
    nonexistent = str(tmp_path / "models" / "ghost_model.joblib")
    with pytest.raises(FileNotFoundError) as exc_info:
        evaluate_model_pipeline(model_path=nonexistent)
    assert "does not exist" in str(exc_info.value)

    # 2. Corrupt model file
    corrupt_file = tmp_path / "models" / "corrupt_model.joblib"
    corrupt_file.parent.mkdir(parents=True, exist_ok=True)
    corrupt_file.write_bytes(b"garbage-corrupt-data-not-a-joblib-bundle")
    with pytest.raises(RuntimeError) as exc_info2:
        evaluate_model_pipeline(model_path=str(corrupt_file))
    assert "corrupt" in str(exc_info2.value).lower()

    # 3. Valid trained model evaluation
    report = evaluate_model_pipeline(model_path=trained_model_path, random_state=42)
    assert report["real_world_accuracy_claim"] is False
    assert "synthetic behavioral scenarios" in report["disclaimer"].lower()
    assert report["model_artifact"] == trained_model_path
    assert report["model_metadata"]["dataset_type"] == "e2e_synthetic_test"
    assert report["model_metadata"]["schema_version"] == FEATURE_SCHEMA_VERSION
    assert report["splits"]["validation_samples"] > 0
    assert report["splits"]["test_samples"] > 0
    assert 0.0 <= report["operating_threshold"] <= 1.0
    res = report["results"]
    assert "confusion_matrix" in res
    assert "metrics" in res
    assert 0.0 <= res["metrics"]["precision"] <= 1.0
    assert 0.0 <= res["metrics"]["recall"] <= 1.0
    assert 0.0 <= res["metrics"]["f1_score"] <= 1.0
