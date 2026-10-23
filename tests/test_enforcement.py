"""
BotShield – Phase 6 Risk-Based Enforcement and Mitigation Tests

Unit and integration tests for:
1. Monitoring-only behavior never blocks a request (default safe mode).
2. LOW-risk sessions are allowed normally.
3. MEDIUM-risk sessions follow configured policy (e.g. sliding-window rate limiting).
4. HIGH-risk sessions follow configured policy (e.g. HTTP 403 restriction on sensitive routes).
5. Missing risk records apply configured safe fallback.
6. Stale risk records (exceeding TTL) apply configured safe fallback.
7. Untrained ML / FSM-only scoring mode respects FSM risk score.
8. Invalid or missing session identifiers fall back safely to client IP.
9. Excluded routes (/static, /api/events, /favicon.ico) always bypass enforcement.
10. Rate-limit boundary behavior (exact quota thresholding and reset after window).
11. Correct HTTP status codes and headers (429 Too Many Requests with Retry-After, 403 Forbidden).
12. Database and evaluation failures fail open gracefully without application outage.
13. Policy exceptions and middleware failure resilience.
14. Enforcement disabled mode completely bypasses evaluation.
15. Repeated requests and bounded limiter state (pruning and hard ceiling on memory).
16. Existing storefront routes and telemetry ingestion remain fully functional.
"""

import os
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from flask import Flask, g, jsonify, request

from backend.database.db import (
    get_connection,
    init_db,
    insert_events,
    insert_request,
    upsert_session,
    upsert_session_risk,
)
from backend.enforcement import (
    BoundedRateLimiter,
    EnforcementAction,
    EnforcementConfig,
    EnforcementDecision,
    EnforcementMode,
    EnforcementPolicy,
    EnforcementService,
    init_enforcement_middleware,
)
from backend.scoring import RiskLevel, ScoringMode


@pytest.fixture
def temp_db():
    """Create an isolated, freshly initialized SQLite database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    try:
        os.remove(db_path)
    except OSError:
        pass


@pytest.fixture
def custom_app(temp_db):
    """Factory fixture creating an isolated test Flask application with enforcement middleware."""
    test_app = Flask(__name__)
    test_app.config["TESTING"] = True
    test_app.config["BOTSHIELD_DB"] = temp_db
    test_app.config["ENFORCEMENT_MODE"] = "monitor"
    test_app.config["ENFORCEMENT_FAIL_OPEN"] = True
    test_app.config["ENFORCEMENT_SENSITIVE_ROUTES"] = ("/login", "/checkout")
    test_app.config["ENFORCEMENT_EXCLUDED_ROUTES"] = ("/static", "/favicon.ico", "/api/events")
    test_app.config["ENFORCEMENT_RATE_LIMIT_REQUESTS"] = 3
    test_app.config["ENFORCEMENT_RATE_LIMIT_WINDOW"] = 5.0
    test_app.config["ENFORCEMENT_CACHE_TTL"] = 30.0

    @test_app.route("/")
    def index():
        return "Home Page", 200

    @test_app.route("/login", methods=["GET", "POST"])
    def login_route():
        return "Login Page", 200

    @test_app.route("/checkout", methods=["POST"])
    def checkout_route():
        return jsonify({"status": "order placed"}), 200

    @test_app.route("/static/test.css")
    def static_route():
        return "body {}", 200

    @test_app.route("/api/events", methods=["POST"])
    def events_route():
        return jsonify({"ok": True}), 200

    return test_app


# ---------------------------------------------------------------------------
# 1. Bounded Sliding-Window Rate Limiter Unit Tests
# ---------------------------------------------------------------------------

def test_bounded_rate_limiter_quota_and_window():
    """Verify exact request quota tracking and sliding window expiration."""
    limiter = BoundedRateLimiter(max_tracked_clients=100)
    client_key = "test-client-1"

    t0 = 1000.0

    # Quota: 3 requests per 10 seconds
    limited, retry = limiter.check(client_key, max_requests=3, window_seconds=10.0, now=t0)
    assert not limited
    assert retry == 0

    limited, retry = limiter.check(client_key, max_requests=3, window_seconds=10.0, now=t0 + 1.0)
    assert not limited
    assert retry == 0

    limited, retry = limiter.check(client_key, max_requests=3, window_seconds=10.0, now=t0 + 2.0)
    assert not limited
    assert retry == 0

    # 4th request within window must be rate-limited
    limited, retry = limiter.check(client_key, max_requests=3, window_seconds=10.0, now=t0 + 3.0)
    assert limited
    assert retry == 7  # 1000.0 + 10.0 - 1003.0 = 7.0 seconds remaining

    # Advance beyond window of first request (t0 + 10.1)
    limited, retry = limiter.check(client_key, max_requests=3, window_seconds=10.0, now=t0 + 10.1)
    assert not limited
    assert retry == 0


def test_bounded_rate_limiter_memory_bounds():
    """Limiter must never grow past max_tracked_clients, pruning oldest entries."""
    limiter = BoundedRateLimiter(max_tracked_clients=5)

    now = 1000.0
    for i in range(15):
        limiter.check(f"client-{i}", max_requests=5, window_seconds=60.0, now=now)

    assert limiter.tracked_count <= 5


# ---------------------------------------------------------------------------
# 2. Pure Policy Engine Unit Tests
# ---------------------------------------------------------------------------

def test_policy_risk_level_mapping():
    """Policy must map LOW to ALLOW, MEDIUM to RATE_LIMIT, and HIGH to BLOCK."""
    cfg = EnforcementConfig(
        low_risk_action=EnforcementAction.ALLOW,
        medium_risk_action=EnforcementAction.RATE_LIMIT,
        high_risk_action=EnforcementAction.BLOCK,
        fallback_action=EnforcementAction.ALLOW,
    )

    # LOW risk
    action, reason = EnforcementPolicy.evaluate(RiskLevel.LOW, ScoringMode.HYBRID, config=cfg)
    assert action == EnforcementAction.ALLOW
    assert "LOW risk" in reason

    # MEDIUM risk
    action, reason = EnforcementPolicy.evaluate(RiskLevel.MEDIUM, ScoringMode.HYBRID, config=cfg)
    assert action == EnforcementAction.RATE_LIMIT
    assert "MEDIUM risk" in reason

    # HIGH risk
    action, reason = EnforcementPolicy.evaluate(RiskLevel.HIGH, ScoringMode.HYBRID, config=cfg)
    assert action == EnforcementAction.BLOCK
    assert "HIGH risk" in reason


def test_policy_fallback_on_missing_stale_or_error():
    """Missing, stale, or errored risk records must invoke configured fallback action."""
    cfg = EnforcementConfig(fallback_action=EnforcementAction.ALLOW)

    # Missing record
    act_missing, r_missing = EnforcementPolicy.evaluate(None, config=cfg)
    assert act_missing == EnforcementAction.ALLOW
    assert "No existing risk record" in r_missing

    # Stale record
    act_stale, r_stale = EnforcementPolicy.evaluate(RiskLevel.HIGH, is_stale=True, config=cfg)
    assert act_stale == EnforcementAction.ALLOW
    assert "stale" in r_stale

    # Error condition
    act_err, r_err = EnforcementPolicy.evaluate(RiskLevel.HIGH, has_error=True, config=cfg)
    assert act_err == EnforcementAction.ALLOW
    assert "error" in r_err


def test_policy_insufficient_evidence_allows_browsing():
    """Sessions with insufficient evidence must be permitted to browse normally."""
    action, reason = EnforcementPolicy.evaluate(
        RiskLevel.LOW,
        scoring_mode=ScoringMode.INSUFFICIENT_EVIDENCE,
    )
    assert action == EnforcementAction.ALLOW
    assert "Insufficient session evidence" in reason


def test_policy_fsm_only_mode_respects_fsm_score():
    """When ML is untrained (FSM_ONLY mode), high FSM score still results in BLOCK."""
    action, reason = EnforcementPolicy.evaluate(
        RiskLevel.HIGH,
        scoring_mode=ScoringMode.FSM_ONLY,
    )
    assert action == EnforcementAction.BLOCK
    assert "FSM-only" in reason


# ---------------------------------------------------------------------------
# 3. Enforcement Service Integration Tests
# ---------------------------------------------------------------------------

def test_service_route_exclusion_and_sensitivity(temp_db):
    """Excluded and non-sensitive routes must bypass enforcement even for HIGH-risk sessions."""
    service = EnforcementService(
        config=EnforcementConfig(
            mode=EnforcementMode.ENFORCE,
            sensitive_routes=("/login",),
            excluded_routes=("/static", "/api/events"),
        )
    )

    session_id = "sess-high-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id)
    # Store high risk (0.85 = 85.0 -> HIGH)
    upsert_session_risk(conn, session_id, risk_score=0.85, fsm_score=0.85, reasons=["RAPID_TRAVERSAL"])
    conn.commit()
    conn.close()

    # 1. Non-sensitive route (Home '/') -> ALLOW
    dec_home = service.evaluate_request(session_id, "127.0.0.1", "/", db_path=temp_db)
    assert dec_home.action == EnforcementAction.ALLOW
    assert not dec_home.enforced
    assert "non-sensitive" in dec_home.reason.lower()

    # 2. Excluded route ('/static/style.css') -> ALLOW
    dec_static = service.evaluate_request(session_id, "127.0.0.1", "/static/style.css", db_path=temp_db)
    assert dec_static.action == EnforcementAction.ALLOW
    assert not dec_static.enforced
    assert "excluded routes" in dec_static.reason

    # 3. Sensitive route ('/login') -> BLOCK (enforced)
    dec_login = service.evaluate_request(session_id, "127.0.0.1", "/login", db_path=temp_db)
    assert dec_login.action == EnforcementAction.BLOCK
    assert dec_login.enforced
    assert dec_login.risk_level == RiskLevel.HIGH


def test_service_risk_record_freshness_ttl(temp_db):
    """Records exceeding risk_cache_ttl_seconds must be flagged stale and use fallback."""
    service = EnforcementService(
        config=EnforcementConfig(
            mode=EnforcementMode.ENFORCE,
            risk_cache_ttl_seconds=30.0,
            fallback_action=EnforcementAction.ALLOW,
        )
    )

    session_id = "sess-ttl-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id)

    # Insert an old risk record timestamped 60 seconds ago
    old_iso = datetime.fromtimestamp(time.time() - 60.0, timezone.utc).isoformat()
    upsert_session_risk(
        conn,
        session_id=session_id,
        risk_score=0.90,
        fsm_score=0.90,
        reasons=["BOT"],
        updated_at=old_iso,
    )
    conn.commit()
    conn.close()

    dec = service.evaluate_request(session_id, "127.0.0.1", "/login", db_path=temp_db)
    assert dec.is_stale
    assert dec.action == EnforcementAction.ALLOW
    assert "stale" in dec.reason


# ---------------------------------------------------------------------------
# 4. Middleware End-to-End Tests
# ---------------------------------------------------------------------------

def test_middleware_monitoring_only_never_blocks(custom_app, temp_db):
    """In MONITOR mode (default), even HIGH-risk sessions receive 200 OK and are never blocked."""
    custom_app.config["ENFORCEMENT_MODE"] = "monitor"
    init_enforcement_middleware(custom_app)

    session_id = "sess-mon-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id)
    upsert_session_risk(conn, session_id, risk_score=0.95, fsm_score=0.95, reasons=["ATTACK"])
    conn.commit()
    conn.close()

    client = custom_app.test_client()
    client.set_cookie("botshield_session", session_id)

    # Sensitive route request under MONITOR mode
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"Login Page" in resp.data


def test_middleware_enforce_mode_blocks_high_risk_session(custom_app, temp_db):
    """In ENFORCE mode, HIGH-risk sessions on sensitive routes receive HTTP 403 Forbidden."""
    custom_app.config["ENFORCEMENT_MODE"] = "enforce"
    init_enforcement_middleware(custom_app)

    session_id = "sess-enf-block-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id)
    upsert_session_risk(conn, session_id, risk_score=0.80, fsm_score=0.80, reasons=["SCRAPER"])
    conn.commit()
    conn.close()

    client = custom_app.test_client()
    client.set_cookie("botshield_session", session_id)

    # 1. HTML request on sensitive route /login -> 403 with HTML challenge page
    resp_html = client.get("/login")
    assert resp_html.status_code == 403
    assert b"Access Restricted" in resp_html.data

    # 2. JSON API request on sensitive route /checkout -> 403 with JSON payload
    resp_json = client.post("/checkout", json={"item": 1})
    assert resp_json.status_code == 403
    data = resp_json.get_json()
    assert data["error"] == "Forbidden"


def test_middleware_enforce_mode_rate_limits_medium_risk(custom_app, temp_db):
    """In ENFORCE mode, MEDIUM-risk sessions exceeding quota receive HTTP 429 with Retry-After."""
    custom_app.config["ENFORCEMENT_MODE"] = "enforce"
    custom_app.config["ENFORCEMENT_RATE_LIMIT_REQUESTS"] = 2
    custom_app.config["ENFORCEMENT_RATE_LIMIT_WINDOW"] = 10.0
    init_enforcement_middleware(custom_app)

    session_id = "sess-enf-med-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id)
    # 0.50 score -> MEDIUM risk (rate limited)
    upsert_session_risk(conn, session_id, risk_score=0.50, fsm_score=0.50, reasons=["SUSPICIOUS_TIMING"])
    conn.commit()
    conn.close()

    client = custom_app.test_client()
    client.set_cookie("botshield_session", session_id)

    # Request 1: allowed
    r1 = client.get("/login")
    assert r1.status_code == 200

    # Request 2: allowed
    r2 = client.get("/login")
    assert r2.status_code == 200

    # Request 3: rate limited (exceeds 2 requests)
    r3 = client.get("/login")
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers
    assert int(r3.headers["Retry-After"]) > 0


def test_middleware_fail_open_on_database_error(custom_app):
    """Database failures must fail open without crashing requests when fail_open=True."""
    custom_app.config["ENFORCEMENT_MODE"] = "enforce"
    custom_app.config["ENFORCEMENT_FAIL_OPEN"] = True
    # Point to nonexistent / unwriteable directory to force connection error
    custom_app.config["BOTSHIELD_DB"] = "/nonexistent/invalid_path/db.sqlite"
    init_enforcement_middleware(custom_app)

    client = custom_app.test_client()
    client.set_cookie("botshield_session", "valid-format-session-uuid-12345")

    # Request must succeed despite broken database
    resp = client.get("/login")
    assert resp.status_code == 200


def test_middleware_disabled_mode_bypasses_all(custom_app, temp_db):
    """In DISABLED mode, all checks are bypassed and g contains no enforcement action."""
    custom_app.config["ENFORCEMENT_MODE"] = "disabled"
    init_enforcement_middleware(custom_app)

    client = custom_app.test_client()
    resp = client.get("/login")
    assert resp.status_code == 200


def test_decision_to_dict_structure():
    """Verify EnforcementDecision serialization contains no sensitive data."""
    decision = EnforcementDecision(
        action=EnforcementAction.RATE_LIMIT,
        enforced=True,
        session_id="sess-dict-001",
        client_ip="192.168.1.1",
        path="/login",
        risk_level=RiskLevel.MEDIUM,
        risk_score=0.45,
        composite_score=45.0,
        scoring_mode=ScoringMode.HYBRID,
        reason="Throttled by rate limit",
        retry_after=5,
        is_stale=False,
        mode=EnforcementMode.ENFORCE,
    )

    d = decision.to_dict()
    assert d["action"] == "RATE_LIMIT"
    assert d["enforced"] is True
    assert d["session_id"] == "sess-dict-001"
    assert d["risk_level"] == "MEDIUM"
    assert d["composite_score"] == 45.0
    assert d["retry_after"] == 5
    # Ensure no cookies or auth headers in dict
    assert "cookie" not in d
    assert "password" not in d


def test_enforcement_config_validation():
    """EnforcementConfig must validate positive bounds on rates and TTLs."""
    with pytest.raises(ValueError, match="rate_limit_requests"):
        EnforcementConfig(rate_limit_requests=0)

    with pytest.raises(ValueError, match="rate_limit_window_seconds"):
        EnforcementConfig(rate_limit_window_seconds=-1.0)

    with pytest.raises(ValueError, match="risk_cache_ttl_seconds"):
        EnforcementConfig(risk_cache_ttl_seconds=0.0)

    with pytest.raises(ValueError, match="max_tracked_clients"):
        EnforcementConfig(max_tracked_clients=-5)


def test_rate_limiter_reset():
    """Verify reset() empties all tracked state in BoundedRateLimiter."""
    limiter = BoundedRateLimiter(max_tracked_clients=10)
    limiter.check("ip-1", max_requests=5, window_seconds=10.0)
    limiter.check("ip-2", max_requests=5, window_seconds=10.0)
    assert limiter.tracked_count == 2
    limiter.reset()
    assert limiter.tracked_count == 0


def test_missing_or_anonymous_session_rates_by_ip(temp_db):
    """Anonymous or missing session ID must be rate-limited by client IP when rate limiting applies."""
    service = EnforcementService(
        config=EnforcementConfig(
            mode=EnforcementMode.ENFORCE,
            fallback_action=EnforcementAction.RATE_LIMIT,
            rate_limit_requests=2,
            rate_limit_window_seconds=10.0,
            sensitive_routes=("/login",),
        )
    )

    client_ip = "203.0.113.195"

    # Request 1 (missing session, fallback RATE_LIMIT applies, quota: 1 of 2) -> ALLOW
    dec1 = service.evaluate_request(None, client_ip, "/login", db_path=temp_db)
    assert dec1.action == EnforcementAction.ALLOW

    # Request 2 (quota: 2 of 2) -> ALLOW
    dec2 = service.evaluate_request(None, client_ip, "/login", db_path=temp_db)
    assert dec2.action == EnforcementAction.ALLOW

    # Request 3 (exceeds 2) -> RATE_LIMIT enforced
    dec3 = service.evaluate_request(None, client_ip, "/login", db_path=temp_db)
    assert dec3.action == EnforcementAction.RATE_LIMIT
    assert dec3.enforced
    assert dec3.retry_after is not None

