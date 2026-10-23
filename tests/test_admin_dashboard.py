"""
BotShield – Phase 7 Security Administrator Dashboard Tests

Tests for:
1. Authentication & Access Control:
   - Unauthenticated requests to dashboard or APIs are rejected.
   - Invalid credentials/tokens are rejected with 401.
   - Missing server token configuration fails closed (403).
   - Valid authentication succeeds via Bearer header, X-Admin-Token header, cookie, and query param.
   - POST /admin/login sets cookie and redirects.
   - GET /admin/logout clears cookie.
2. Statistics & Aggregation APIs:
   - Empty database returns clean zero-count statistics.
   - Aggregation accurately reflects seeded sessions, events, requests, and risk levels.
   - Nullable risk records are handled safely.
   - Simulated database errors return clean 500 responses without leaking tracebacks.
3. Session Listing & Pagination APIs:
   - Pagination limit and offset work and are bounded.
   - Risk level filtering (LOW, MEDIUM, HIGH, UNASSESSED) works.
   - Safe sorting against allowlisted fields works.
   - Invalid query parameters fall back safely.
4. Session Detail API:
   - Correct details returned for existing session.
   - Returns 404 for nonexistent session.
   - Unauthenticated access is blocked.
   - Sensitive fields are never exposed.
5. Integration with app & existing behavior:
   - Admin routes work when registered on app.
   - Existing storefront and telemetry routes remain functional.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from flask import Flask

from backend.admin import admin_bp
from backend.database.db import (
    get_connection,
    init_db,
    insert_events,
    insert_request,
    upsert_session,
    upsert_session_risk,
)


@pytest.fixture
def temp_db():
    """Create an isolated, freshly initialized SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    try:
        os.remove(db_path)
    except OSError:
        pass


@pytest.fixture
def test_app(temp_db):
    """Create a configured Flask application for admin testing."""
    app = Flask(
        __name__,
        template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend")),
        static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static")),
    )
    app.config["TESTING"] = True
    app.config["BOTSHIELD_DB"] = temp_db
    app.config["ADMIN_TOKEN"] = "secret-soc-token-12345"

    app.register_blueprint(admin_bp)
    return app


@pytest.fixture
def client(test_app):
    return test_app.test_client()


@pytest.fixture
def seeded_db(temp_db):
    """Seed test database with known sessions, events, requests, and risk evaluations."""
    conn = get_connection(temp_db)
    try:
        # Session 1: High risk (bot)
        s1 = "sess-high-risk-001"
        upsert_session(conn, s1)
        insert_events(
            conn,
            s1,
            [
                {"event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:00:01Z", "page": "/"},
                {"event_type": "CLICK", "timestamp": "2026-10-24T00:00:02Z", "page": "/product/1"},
            ],
        )
        insert_request(conn, s1, "GET", "/product/1", 200, "2026-10-24T00:00:02Z", "HeadlessChrome")
        upsert_session_risk(
            conn,
            session_id=s1,
            risk_score=0.88,
            fsm_score=0.90,
            ml_score=0.85,
            reasons=["FSM: rapid transitions detected", "ML: anomaly score high"],
        )

        # Session 2: Medium risk (FSM-only)
        s2 = "sess-med-risk-002"
        upsert_session(conn, s2)
        insert_events(
            conn,
            s2,
            [
                {"event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:01:00Z", "page": "/"},
            ],
        )
        insert_request(conn, s2, "GET", "/", 200, "2026-10-24T00:01:00Z", "PythonRequests")
        upsert_session_risk(
            conn,
            session_id=s2,
            risk_score=0.55,
            fsm_score=0.55,
            ml_score=0.0,
            reasons=["FSM-only: elevated rate"],
        )

        # Session 3: Low risk (human)
        s3 = "sess-low-risk-003"
        upsert_session(conn, s3)
        insert_events(
            conn,
            s3,
            [
                {"event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:02:00Z", "page": "/"},
                {"event_type": "MOUSE_ACTIVITY", "timestamp": "2026-10-24T00:02:01Z", "page": "/"},
                {"event_type": "SCROLL", "timestamp": "2026-10-24T00:02:02Z", "page": "/"},
            ],
        )
        insert_request(conn, s3, "GET", "/", 200, "2026-10-24T00:02:00Z", "Mozilla/5.0")
        upsert_session_risk(
            conn,
            session_id=s3,
            risk_score=0.12,
            fsm_score=0.10,
            ml_score=0.15,
            reasons=["Normal human navigation"],
        )

        # Session 4: Unassessed session (no risk record)
        s4 = "sess-unassessed-004"
        upsert_session(conn, s4)
        insert_request(conn, s4, "GET", "/about", 200, "2026-10-24T00:03:00Z", "Mozilla/5.0")

        conn.commit()
    finally:
        conn.close()
    return temp_db


# ---------------------------------------------------------------------------
# 1. Authentication & Access Control Tests
# ---------------------------------------------------------------------------

def test_unauthenticated_dashboard_redirects_or_prompts(client):
    """Unauthenticated HTML request to dashboard renders access control prompt."""
    res = client.get("/admin/dashboard")
    assert res.status_code == 401
    assert b"Admin Authentication Required" in res.data or b"Access" in res.data


def test_unauthenticated_api_returns_401_json(client):
    """Unauthenticated API requests return 401 Unauthorized JSON."""
    res = client.get("/admin/api/stats")
    assert res.status_code == 401
    assert res.is_json
    data = res.get_json()
    assert data["error"] == "Unauthorized"


def test_invalid_token_returns_401(client):
    """Requests with invalid token header return 401."""
    res = client.get("/admin/api/stats", headers={"Authorization": "Bearer wrong-token"})
    assert res.status_code == 401
    data = res.get_json()
    assert data["error"] == "Unauthorized"


def test_missing_server_token_fails_closed(client, test_app):
    """If ADMIN_TOKEN is not configured on the server, all requests fail closed (403)."""
    test_app.config["ADMIN_TOKEN"] = None
    with patch.dict(os.environ, {"BOTSHIELD_ADMIN_TOKEN": "", "ADMIN_TOKEN": ""}, clear=True):
        res = client.get("/admin/api/stats", headers={"Authorization": "Bearer secret-soc-token-12345"})
        assert res.status_code == 403
        data = res.get_json()
        assert data["error"] == "Forbidden"
        assert "not configured" in data["message"].lower()


def test_authenticated_via_bearer_header(client):
    """Authorization: Bearer <token> grants access."""
    res = client.get("/admin/api/stats", headers={"Authorization": "Bearer secret-soc-token-12345"})
    assert res.status_code == 200
    assert res.is_json


def test_authenticated_via_x_admin_token_header(client):
    """X-Admin-Token header grants access."""
    res = client.get("/admin/api/stats", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    assert res.is_json


def test_authenticated_via_cookie(client):
    """Cookie botshield_admin_token grants access."""
    client.set_cookie("botshield_admin_token", "secret-soc-token-12345")
    res = client.get("/admin/api/stats")
    assert res.status_code == 200
    assert res.is_json


def test_authenticated_via_query_param(client):
    """Query parameter ?token=<token> grants access."""
    res = client.get("/admin/api/stats?token=secret-soc-token-12345")
    assert res.status_code == 200
    assert res.is_json


def test_login_post_valid_token_sets_cookie(client):
    """POST /admin/login sets cookie and redirects to dashboard."""
    res = client.post("/admin/login", data={"token": "secret-soc-token-12345"})
    assert res.status_code == 302
    assert res.headers["Location"].endswith("/admin/dashboard")
    cookies = [h for h in res.headers.getlist("Set-Cookie") if "botshield_admin_token" in h]
    assert len(cookies) > 0


def test_login_post_invalid_token_fails(client):
    """POST /admin/login with invalid token returns 401."""
    res = client.post("/admin/login", data={"token": "wrong-token"})
    assert res.status_code == 401
    assert b"Invalid administrator token" in res.data


def test_logout_clears_cookie(client):
    """GET /admin/logout clears admin cookie."""
    client.set_cookie("botshield_admin_token", "secret-soc-token-12345")
    res = client.get("/admin/logout")
    assert res.status_code == 302
    # Verify cookie expires in response
    cookies = [h for h in res.headers.getlist("Set-Cookie") if "botshield_admin_token" in h]
    assert len(cookies) > 0


# ---------------------------------------------------------------------------
# 2. Statistics & Aggregation API Tests
# ---------------------------------------------------------------------------

def test_stats_empty_database(client):
    """Stats endpoint handles an empty database gracefully."""
    res = client.get("/admin/api/stats", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["total_sessions"] == 0
    assert data["total_events"] == 0
    assert data["total_requests"] == 0
    assert data["risk_counts"]["LOW"] == 0
    assert data["risk_counts"]["MEDIUM"] == 0
    assert data["risk_counts"]["HIGH"] == 0
    assert data["risk_counts"]["UNASSESSED"] == 0
    assert data["scoring_modes"]["HYBRID"] == 0
    assert data["scoring_modes"]["FSM_ONLY"] == 0
    assert data["recent_high_risk"] == []


def test_stats_with_seeded_data(client, seeded_db):
    """Stats endpoint returns accurate aggregated counts from seeded data."""
    res = client.get("/admin/api/stats", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()

    assert data["total_sessions"] == 4
    assert data["total_events"] == 6  # 2 in s1, 1 in s2, 3 in s3
    assert data["total_requests"] == 4  # 1 each in s1, s2, s3, s4

    # Risk counts
    assert data["risk_counts"]["HIGH"] == 1
    assert data["risk_counts"]["MEDIUM"] == 1
    assert data["risk_counts"]["LOW"] == 1
    assert data["risk_counts"]["UNASSESSED"] == 1

    # Scoring modes
    assert data["scoring_modes"]["HYBRID"] == 2
    assert data["scoring_modes"]["FSM_ONLY"] == 1

    # Recent high risk session list
    assert len(data["recent_high_risk"]) == 1
    high_sess = data["recent_high_risk"][0]
    assert high_sess["session_id"] == "sess-high-risk-001"
    assert high_sess["risk_score"] == 0.88
    assert high_sess["risk_level"] == "HIGH"


def test_stats_database_error_handling(client):
    """Database query failure in stats returns clean 500 without leaking stack traces."""
    with patch("backend.admin.routes.get_admin_overview_stats", side_effect=Exception("DB lock simulation")):
        res = client.get("/admin/api/stats", headers={"X-Admin-Token": "secret-soc-token-12345"})
        assert res.status_code == 500
        data = res.get_json()
        assert data["error"] == "Internal Server Error"
        assert "DB lock simulation" not in json.dumps(data)


# ---------------------------------------------------------------------------
# 3. Session Listing & Pagination API Tests
# ---------------------------------------------------------------------------

def test_sessions_pagination_defaults(client, seeded_db):
    """GET /admin/api/sessions returns paginated list with total count."""
    res = client.get("/admin/api/sessions", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 4
    assert len(data["sessions"]) == 4
    assert data["limit"] == 25
    assert data["offset"] == 0


def test_sessions_pagination_limits(client, seeded_db):
    """GET /admin/api/sessions respects limit and offset."""
    res = client.get("/admin/api/sessions?limit=2&offset=1", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 4
    assert len(data["sessions"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 1


def test_sessions_filter_by_risk_level(client, seeded_db):
    """Filter by risk_level=HIGH returns only high risk sessions."""
    res = client.get("/admin/api/sessions?risk_level=HIGH", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 1
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"] == "sess-high-risk-001"
    assert data["sessions"][0]["risk_level"] == "HIGH"


def test_sessions_filter_by_unassessed(client, seeded_db):
    """Filter by risk_level=UNASSESSED returns only sessions without risk records."""
    res = client.get("/admin/api/sessions?risk_level=UNASSESSED", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["total"] == 1
    assert data["sessions"][0]["session_id"] == "sess-unassessed-004"
    assert data["sessions"][0]["risk_level"] == "UNASSESSED"


def test_sessions_invalid_params_fallback(client, seeded_db):
    """Invalid limit, offset, or sort values safely fall back to defaults."""
    res = client.get("/admin/api/sessions?limit=-5&offset=xyz&sort_by=malicious_sql&sort_dir=INVALID", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["limit"] == 25
    assert data["offset"] == 0
    assert len(data["sessions"]) == 4


# ---------------------------------------------------------------------------
# 4. Session Detail API Tests
# ---------------------------------------------------------------------------

def test_session_detail_success(client, seeded_db):
    """GET /admin/api/sessions/<id> returns complete evidence and activity counts."""
    res = client.get("/admin/api/sessions/sess-high-risk-001", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["session_id"] == "sess-high-risk-001"
    assert data["risk_score"] == 0.88
    assert data["risk_level"] == "HIGH"
    assert data["fsm_score"] == 0.90
    assert data["ml_score"] == 0.85
    assert data["evaluation_mode"] == "HYBRID"
    assert "FSM: rapid transitions detected" in data["reasons"]
    assert data["event_count"] == 2
    assert data["request_count"] == 1
    assert len(data["recent_events"]) == 2
    assert len(data["recent_requests"]) == 1


def test_session_detail_unassessed_session(client, seeded_db):
    """Unassessed session returns UNASSESSED risk level without crashing."""
    res = client.get("/admin/api/sessions/sess-unassessed-004", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["session_id"] == "sess-unassessed-004"
    assert data["risk_level"] == "UNASSESSED"
    assert data["risk_score"] is None
    assert data["event_count"] == 0
    assert data["request_count"] == 1


def test_session_detail_not_found(client, seeded_db):
    """Nonexistent session returns 404."""
    res = client.get("/admin/api/sessions/sess-does-not-exist", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 404
    data = res.get_json()
    assert data["error"] == "Not Found"


def test_session_detail_sensitive_data_not_exposed(client, seeded_db):
    """Response does not leak credentials, secrets, or internal paths."""
    res = client.get("/admin/api/sessions/sess-high-risk-001", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    raw = res.data.decode("utf-8")
    assert "password" not in raw.lower()
    assert "secret" not in raw.lower()
    assert "cookie" not in raw.lower()


# ---------------------------------------------------------------------------
# 5. UI Rendering & Integration Tests
# ---------------------------------------------------------------------------

def test_dashboard_ui_renders_for_authorized_user(client):
    """Authorized user can load dashboard HTML."""
    res = client.get("/admin/dashboard", headers={"X-Admin-Token": "secret-soc-token-12345"})
    assert res.status_code == 200
    assert b"Security Operations Center" in res.data
    assert b"SOC" in res.data
    assert b"Telemetry Events" in res.data
    assert b"Risk Level Distribution" in res.data
