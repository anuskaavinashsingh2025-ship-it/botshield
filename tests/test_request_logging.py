"""
BotShield – Phase 2B Request Logging & Session Correlation Tests

Tests for:
- Server-side request logging on normal routes (GET /, GET /products, etc.)
- Session correlation across page views and telemetry events
- Session cookie issuance, attributes, and validation
- Untrusted and malformed session ID rejection (preventing DB pollution)
- Sensitive data exclusion (query parameters, bodies, passwords, auth tokens)
- Error and non-2xx status code logging (3xx, 4xx, 5xx)
- Static asset and configured path exclusion
- Resilience against logger and database failures (fails open)
- Duplicate logging prevention
"""

import json
import os
import sqlite3
import sys

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from backend.database.db import (
    get_connection,
    get_requests,
    get_session,
    init_db,
    upsert_session,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provide an isolated, freshly initialized SQLite database for each test."""
    db_file = str(tmp_path / "test_logging.db")
    monkeypatch.setenv("BOTSHIELD_DB", db_file)
    init_db(db_file)
    return db_file


@pytest.fixture
def client(temp_db):
    """Configure Flask test client with clean isolated testing settings."""
    app.config["TESTING"] = True
    app.config["REQUEST_LOGGING_ENABLED"] = True
    app.config["REQUEST_LOGGING_EXCLUDE_STATIC"] = True
    app.config["REQUEST_LOGGING_EXCLUDE_PATHS"] = {"/favicon.ico"}
    with app.test_client() as c:
        yield c


VALID_SESS_1 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
VALID_SESS_2 = "11111111-2222-3333-4444-555555555555"


# ---------------------------------------------------------------------------
# 1. Normal Request Logging
# ---------------------------------------------------------------------------

def test_normal_request_logging(client, temp_db):
    """Normal page requests should be recorded with method, path, status, and user-agent."""
    resp = client.get(
        "/products",
        headers={"User-Agent": "Mozilla/5.0 BotShieldTester/1.0"},
    )
    assert resp.status_code == 200

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    assert len(reqs) == 1
    r = reqs[0]
    assert r["method"] == "GET"
    assert r["path"] == "/products"
    assert r["status_code"] == 200
    assert "BotShieldTester" in r["user_agent"]
    assert "T" in r["timestamp"]


def test_session_cookie_issued_on_first_request(client, temp_db):
    """Initial request without cookie receives a valid botshield_session cookie."""
    resp = client.get("/")
    assert resp.status_code == 200

    # Cookie check
    cookie_header = resp.headers.get("Set-Cookie", "")
    assert "botshield_session=" in cookie_header
    assert "Path=/" in cookie_header
    assert "SameSite=Lax" in cookie_header

    # Request record should have session_id=None because session is not yet established in DB
    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()
    assert len(reqs) == 1
    assert reqs[0]["session_id"] is None


# ---------------------------------------------------------------------------
# 2. Session Correlation
# ---------------------------------------------------------------------------

def test_session_correlation_with_known_session(client, temp_db):
    """When a verified session sends a request with cookie, the request is correlated."""
    conn = get_connection(temp_db)
    upsert_session(conn, VALID_SESS_1)
    conn.commit()
    conn.close()

    client.set_cookie("botshield_session", VALID_SESS_1)
    resp = client.get("/about")
    assert resp.status_code == 200

    conn = get_connection(temp_db)
    reqs = get_requests(conn, session_id=VALID_SESS_1)
    conn.close()

    assert len(reqs) == 1
    assert reqs[0]["session_id"] == VALID_SESS_1
    assert reqs[0]["path"] == "/about"


def test_session_correlation_lifecycle_with_telemetry(client, temp_db):
    """
    Test full correlation flow:
    1. First hit GET / receives session cookie.
    2. Telemetry POST /api/events establishes session in DB.
    3. Next page view GET /products is correlated to that session.
    """
    # 1. First hit
    resp1 = client.get("/")
    assert resp1.status_code == 200
    cookie_val = client.get_cookie("botshield_session").value
    assert cookie_val is not None

    # 2. Telemetry events establish session
    resp_evt = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": cookie_val,
            "events": [{"event_type": "PAGE_VIEW", "timestamp": "2026-10-23T12:00:00Z", "page": "/"}],
        }),
        content_type="application/json",
    )
    assert resp_evt.status_code == 200

    # 3. Subsequent page request carries cookie
    resp2 = client.get("/products")
    assert resp2.status_code == 200

    # Check requests table: both /api/events and /products should be correlated
    conn = get_connection(temp_db)
    correlated = get_requests(conn, session_id=cookie_val)
    conn.close()

    paths = [r["path"] for r in correlated]
    assert "/api/events" in paths
    assert "/products" in paths


# ---------------------------------------------------------------------------
# 3. Untrusted & Malformed Session Handling (DB Pollution Prevention)
# ---------------------------------------------------------------------------

def test_untrusted_client_session_not_created_in_sessions_table(client, temp_db):
    """
    If a client sends an unverified session ID (even in valid UUID format),
    request logging MUST NOT create a new session row in the sessions table.
    """
    untrusted_id = "99999999-9999-9999-9999-999999999999"
    client.set_cookie("botshield_session", untrusted_id)

    resp = client.get("/products")
    assert resp.status_code == 200

    conn = get_connection(temp_db)
    # 1. sessions table must NOT have this untrusted ID
    session_row = get_session(conn, untrusted_id)
    assert session_row is None

    # 2. request must be logged with session_id=None
    reqs = get_requests(conn)
    conn.close()
    assert len(reqs) == 1
    assert reqs[0]["session_id"] is None

    # 3. Response should issue a fresh session cookie
    new_cookie = client.get_cookie("botshield_session").value
    assert new_cookie != untrusted_id


def test_malformed_session_cookie_handled_safely(client, temp_db):
    """Malformed or injection-attempt session cookies must be rejected safely."""
    client.set_cookie("botshield_session", "../../etc/passwd!@#$")

    resp = client.get("/login")
    assert resp.status_code == 200

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    assert len(reqs) == 1
    assert reqs[0]["session_id"] is None

    # A valid fresh UUID cookie is set
    cookie = client.get_cookie("botshield_session").value
    assert len(cookie) >= 20


# ---------------------------------------------------------------------------
# 4. Sensitive Data Exclusion & Path Normalization
# ---------------------------------------------------------------------------

def test_sensitive_data_and_query_parameters_excluded(client, temp_db):
    """Query strings, request bodies, and passwords must never be stored."""
    # GET with sensitive query parameters
    resp = client.get(
        "/products?search=my_password&token=secret_auth_token_12345#ref_code",
        headers={"Authorization": "Bearer sensitive_token_abc"},
    )
    assert resp.status_code == 200

    # POST with sensitive body
    resp_post = client.post(
        "/login",
        data={"username": "alice", "password": "SuperSecretPassword99!"},
    )
    assert resp_post.status_code == 200

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    assert len(reqs) == 2
    r_get = reqs[0]
    assert r_get["path"] == "/products"  # query & fragment completely stripped
    assert "token" not in r_get["path"]
    assert "secret" not in r_get["path"]

    r_post = reqs[1]
    assert r_post["path"] == "/login"

    # Search entire database for any occurrence of the password or token string
    conn = get_connection(temp_db)
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()
    for row in rows:
        row_str = " ".join(str(v) for v in row)
        assert "SuperSecretPassword99!" not in row_str
        assert "secret_auth_token_12345" not in row_str
        assert "sensitive_token_abc" not in row_str


# ---------------------------------------------------------------------------
# 5. Non-2xx Responses & Exceptions (404, 500, Redirects)
# ---------------------------------------------------------------------------

def test_404_not_found_logged(client, temp_db):
    """404 responses should be logged with status 404."""
    resp = client.get("/product/9999")
    assert resp.status_code == 404

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    assert len(reqs) == 1
    assert reqs[0]["path"] == "/product/9999"
    assert reqs[0]["status_code"] == 404


def test_500_internal_server_error_logged(client, temp_db):
    """Unhandled exceptions resulting in 500 should be captured via teardown hook."""
    try:
        client.get("/_test/crash")
    except RuntimeError:
        pass  # In testing mode with exceptions, Flask re-raises

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    crash_reqs = [r for r in reqs if r["path"] == "/_test/crash"]
    assert len(crash_reqs) == 1
    assert crash_reqs[0]["status_code"] == 500


def test_redirect_logged(client, temp_db):
    """Redirect responses (3xx) should be logged with redirect status code."""
    resp = client.get("/_test/redirect", follow_redirects=False)
    assert resp.status_code == 302

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    redir = [r for r in reqs if r["path"] == "/_test/redirect"]
    assert len(redir) == 1
    assert redir[0]["status_code"] == 302



# ---------------------------------------------------------------------------
# 6. Route Exclusion (Static Assets & Configurable Paths)
# ---------------------------------------------------------------------------

def test_static_asset_exclusion(client, temp_db):
    """Static assets under /static/ must be excluded from requests table."""
    resp = client.get("/static/css/style.css")
    assert resp.status_code == 200

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    # Zero requests logged for static asset
    assert len(reqs) == 0


def test_custom_excluded_paths(client, temp_db):
    """Paths listed in REQUEST_LOGGING_EXCLUDE_PATHS should not be logged."""
    app.config["REQUEST_LOGGING_EXCLUDE_PATHS"] = {"/about", "/favicon.ico"}

    resp = client.get("/about")
    assert resp.status_code == 200

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    assert len(reqs) == 0


def test_logging_disabled_flag(client, temp_db):
    """When REQUEST_LOGGING_ENABLED is False, no requests are recorded."""
    app.config["REQUEST_LOGGING_ENABLED"] = False

    resp = client.get("/products")
    assert resp.status_code == 200

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    assert len(reqs) == 0


# ---------------------------------------------------------------------------
# 7. Resilience Against Failures & Duplicate Prevention
# ---------------------------------------------------------------------------

def test_logger_failure_does_not_crash_application(client, temp_db, monkeypatch):
    """If database insert fails, the user's request still succeeds (fails open)."""
    def _failing_insert(*args, **kwargs):
        raise sqlite3.OperationalError("Simulated SQLite lock error")

    monkeypatch.setattr("backend.middleware.request_logger.insert_request", _failing_insert)

    resp = client.get("/products")
    # Response must still return 200 OK without crashing
    assert resp.status_code == 200
    assert b"Products" in resp.data


def test_duplicate_logging_prevention(client, temp_db):
    """Ensure each request only creates exactly 1 row in the requests table."""
    client.get("/")
    client.get("/about")

    conn = get_connection(temp_db)
    reqs = get_requests(conn)
    conn.close()

    assert len(reqs) == 2
    assert reqs[0]["path"] == "/"
    assert reqs[1]["path"] == "/about"
