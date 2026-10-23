"""
BotShield – Phase 2A Database Foundation Tests

Tests for:
- Schema initialization (tables, indexes, idempotence)
- Backward compatibility and non-destructive schema migration
- Request metadata logging (insert_request, get_requests, normalization, sanitization)
- Session risk record persistence (upsert_session_risk, get_session_risk)
- Foreign key constraints, missing sessions, and input validation
- Configurable BOTSHIELD_DB environment variable handling
"""

import json
import os
import sqlite3
import sys

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.db import (
    get_connection,
    get_requests,
    get_session,
    get_session_risk,
    init_db,
    insert_request,
    normalize_path,
    sanitize_user_agent,
    upsert_session,
    upsert_session_risk,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provide an isolated, freshly initialized SQLite database for each test."""
    db_file = str(tmp_path / "test_phase2a.db")
    monkeypatch.setenv("BOTSHIELD_DB", db_file)
    init_db(db_file)
    return db_file


VALID_SESS_1 = "11111111-2222-3333-4444-555555555555"
VALID_SESS_2 = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# ---------------------------------------------------------------------------
# 1. Schema Initialization & Idempotence
# ---------------------------------------------------------------------------

def test_schema_creates_all_tables_and_indexes(temp_db):
    """Verify that sessions, events, requests, and session_risks exist with their indexes."""
    conn = get_connection(temp_db)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur.fetchall()}
    assert {"sessions", "events", "requests", "session_risks"}.issubset(tables)

    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = {row["name"] for row in cur.fetchall()}
    expected_indexes = {
        "idx_events_session",
        "idx_events_timestamp",
        "idx_events_type",
        "idx_requests_session",
        "idx_requests_timestamp",
        "idx_requests_path",
        "idx_session_risks_score",
    }
    assert expected_indexes.issubset(indexes)
    conn.close()


def test_schema_initialization_is_idempotent(temp_db):
    """Calling init_db multiple times must not fail or corrupt the database."""
    init_db(temp_db)
    init_db(temp_db)
    conn = get_connection(temp_db)
    cur = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
    count = cur.fetchone()[0]
    assert count >= 4
    conn.close()


def test_safe_upgrade_from_legacy_database(tmp_path):
    """An existing database with only sessions/events should be safely upgraded with new tables."""
    legacy_db = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(legacy_db)
    # Simulate legacy Step 3 database schema
    conn.executescript("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            last_seen  TEXT NOT NULL
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            page TEXT NOT NULL,
            metadata TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        INSERT INTO sessions (session_id, created_at, last_seen)
        VALUES ('pre-existing-session-12345', '2026-09-01T00:00:00Z', '2026-09-01T00:00:00Z');
        INSERT INTO events (session_id, event_type, timestamp, page, metadata)
        VALUES ('pre-existing-session-12345', 'PAGE_VIEW', '2026-09-01T00:00:00Z', '/', '{}');
    """)
    conn.commit()
    conn.close()

    # Run modern init_db against legacy database
    init_db(legacy_db)

    # Verify existing data is completely intact
    conn = get_connection(legacy_db)
    session = get_session(conn, "pre-existing-session-12345")
    assert session is not None
    assert session["session_id"] == "pre-existing-session-12345"

    cur = conn.execute("SELECT COUNT(*) FROM events")
    assert cur.fetchone()[0] == 1

    # Verify new tables are now available and writable
    req_id = insert_request(
        conn,
        session_id="pre-existing-session-12345",
        method="GET",
        path="/products",
        status_code=200,
    )
    conn.commit()
    assert req_id > 0
    conn.close()


# ---------------------------------------------------------------------------
# 2. Path Normalization & User-Agent Sanitization Helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("/", "/"),
        ("/products", "/products"),
        ("/products/", "/products"),
        ("/products?page=2&sort=asc", "/products"),
        ("/products#featured", "/products"),
        ("/product/1?ref=home#reviews", "/product/1"),
        ("products", "/products"),
        ("///products///detail///1", "/products/detail/1"),
        ("  /login  ", "/login"),
    ],
)
def test_normalize_path(raw, expected):
    assert normalize_path(raw) == expected


def test_normalize_path_invalid():
    with pytest.raises(ValueError):
        normalize_path("")
    with pytest.raises(ValueError):
        normalize_path("   ")
    with pytest.raises(ValueError):
        normalize_path(None)  # type: ignore


def test_sanitize_user_agent():
    assert sanitize_user_agent(None) == ""
    assert sanitize_user_agent("") == ""

    # Normal user agent
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    assert sanitize_user_agent(ua) == ua

    # Strips control characters like CRLF (injection prevention)
    malicious_ua = "Mozilla/5.0\r\nInjected-Header: evil\n\tSubverted"
    sanitized = sanitize_user_agent(malicious_ua)
    assert "\r" not in sanitized
    assert "\n" not in sanitized
    assert "\t" not in sanitized
    assert "Injected-Header: evil" in sanitized

    # Length truncation
    long_ua = "A" * 600
    assert len(sanitize_user_agent(long_ua, max_len=100)) == 100


# ---------------------------------------------------------------------------
# 3. Request Insertion & Querying
# ---------------------------------------------------------------------------

def test_insert_request_with_valid_session(temp_db):
    """Inserting a request with an existing session stores all fields properly."""
    conn = get_connection(temp_db)
    upsert_session(conn, VALID_SESS_1)
    conn.commit()

    req_id = insert_request(
        conn,
        session_id=VALID_SESS_1,
        method="GET",
        path="/products?category=Desk",
        status_code=200,
        timestamp="2026-10-23T12:00:00.000000Z",
        user_agent="Mozilla/5.0 TestBrowser",
    )
    conn.commit()
    assert req_id == 1

    requests = get_requests(conn, session_id=VALID_SESS_1)
    assert len(requests) == 1
    r = requests[0]
    assert r["id"] == req_id
    assert r["session_id"] == VALID_SESS_1
    assert r["method"] == "GET"
    assert r["path"] == "/products"  # query stripped
    assert r["status_code"] == 200
    assert r["timestamp"] == "2026-10-23T12:00:00.000000Z"
    assert r["user_agent"] == "Mozilla/5.0 TestBrowser"
    conn.close()


def test_insert_request_nullable_session(temp_db):
    """Requests without a session (e.g. unassociated/first-hit) should store NULL session_id."""
    conn = get_connection(temp_db)
    req_id_none = insert_request(
        conn,
        session_id=None,
        method="GET",
        path="/",
        status_code=200,
    )
    req_id_empty = insert_request(
        conn,
        session_id="   ",
        method="POST",
        path="/login",
        status_code=401,
    )
    conn.commit()

    all_reqs = get_requests(conn)
    assert len(all_reqs) == 2
    assert all_reqs[0]["session_id"] is None
    assert all_reqs[1]["session_id"] is None
    conn.close()


def test_insert_request_default_timestamp_generated(temp_db):
    """When timestamp is None, insert_request generates a valid ISO UTC timestamp."""
    conn = get_connection(temp_db)
    req_id = insert_request(
        conn,
        session_id=None,
        method="HEAD",
        path="/about",
        status_code=200,
        timestamp=None,
    )
    conn.commit()

    reqs = get_requests(conn)
    assert len(reqs) == 1
    assert "T" in reqs[0]["timestamp"]
    assert "Z" in reqs[0]["timestamp"]
    conn.close()


def test_insert_request_foreign_key_violation(temp_db):
    """Inserting a request with a non-existent session_id must trigger foreign key constraint failure."""
    conn = get_connection(temp_db)
    with pytest.raises(sqlite3.IntegrityError):
        insert_request(
            conn,
            session_id="non-existent-session-uuid",
            method="GET",
            path="/",
            status_code=200,
        )
    conn.close()


def test_insert_request_validation_errors(temp_db):
    """insert_request must reject invalid methods, paths, and status codes."""
    conn = get_connection(temp_db)

    # Invalid method
    with pytest.raises(ValueError, match="method"):
        insert_request(conn, None, "", "/", 200)
    with pytest.raises(ValueError, match="method"):
        insert_request(conn, None, "INVALID_VERB", "/", 200)

    # Invalid path
    with pytest.raises(ValueError, match="path"):
        insert_request(conn, None, "GET", "", 200)

    # Invalid status code
    with pytest.raises(ValueError, match="status_code"):
        insert_request(conn, None, "GET", "/", "200")  # type: ignore
    with pytest.raises(ValueError, match="status_code"):
        insert_request(conn, None, "GET", "/", True)  # type: ignore (bool is subclass of int)
    with pytest.raises(ValueError, match="range"):
        insert_request(conn, None, "GET", "/", 99)
    with pytest.raises(ValueError, match="range"):
        insert_request(conn, None, "GET", "/", 600)

    conn.close()


def test_get_requests_pagination_and_ordering(temp_db):
    """get_requests should return rows ordered chronologically and respect limit."""
    conn = get_connection(temp_db)
    upsert_session(conn, VALID_SESS_1)
    upsert_session(conn, VALID_SESS_2)

    for i in range(5):
        insert_request(
            conn,
            session_id=VALID_SESS_1,
            method="GET",
            path=f"/page/{i}",
            status_code=200,
            timestamp=f"2026-10-23T10:0{i}:00Z",
        )
    insert_request(
        conn,
        session_id=VALID_SESS_2,
        method="GET",
        path="/other",
        status_code=200,
        timestamp="2026-10-23T11:00:00Z",
    )
    conn.commit()

    # Filtered by session
    sess1_reqs = get_requests(conn, session_id=VALID_SESS_1)
    assert len(sess1_reqs) == 5
    assert [r["path"] for r in sess1_reqs] == [f"/page/{i}" for i in range(5)]

    # Limit check
    limited = get_requests(conn, limit=3)
    assert len(limited) == 3

    conn.close()


# ---------------------------------------------------------------------------
# 4. Session Risk Records (Phase 2A)
# ---------------------------------------------------------------------------

def test_upsert_and_get_session_risk(temp_db):
    """Verify writing, reading, and updating session risk records."""
    conn = get_connection(temp_db)
    upsert_session(conn, VALID_SESS_1)
    conn.commit()

    # Initial risk insert
    upsert_session_risk(
        conn,
        session_id=VALID_SESS_1,
        risk_score=0.45,
        fsm_score=0.30,
        ml_score=0.60,
        reasons=["rapid_traversal", "low_mouse_movement"],
        updated_at="2026-10-23T12:00:00Z",
    )
    conn.commit()

    risk = get_session_risk(conn, VALID_SESS_1)
    assert risk is not None
    assert risk["session_id"] == VALID_SESS_1
    assert pytest.approx(risk["risk_score"], 0.001) == 0.45
    assert pytest.approx(risk["fsm_score"], 0.001) == 0.30
    assert pytest.approx(risk["ml_score"], 0.001) == 0.60
    reasons = json.loads(risk["reasons"])
    assert reasons == ["rapid_traversal", "low_mouse_movement"]
    assert risk["updated_at"] == "2026-10-23T12:00:00Z"

    # Update existing risk record (ON CONFLICT)
    upsert_session_risk(
        conn,
        session_id=VALID_SESS_1,
        risk_score=0.88,
        fsm_score=0.90,
        ml_score=0.85,
        reasons=["bot_user_agent_pattern"],
        updated_at="2026-10-23T12:05:00Z",
    )
    conn.commit()

    updated = get_session_risk(conn, VALID_SESS_1)
    assert pytest.approx(updated["risk_score"], 0.001) == 0.88
    assert pytest.approx(updated["fsm_score"], 0.001) == 0.90
    assert json.loads(updated["reasons"]) == ["bot_user_agent_pattern"]
    assert updated["updated_at"] == "2026-10-23T12:05:00Z"

    # Still only one risk row for this session
    cur = conn.execute("SELECT COUNT(*) FROM session_risks")
    assert cur.fetchone()[0] == 1

    conn.close()


def test_session_risk_default_values(temp_db):
    """Default values for fsm_score, ml_score, and reasons should be applied cleanly."""
    conn = get_connection(temp_db)
    upsert_session(conn, VALID_SESS_1)
    conn.commit()

    upsert_session_risk(conn, session_id=VALID_SESS_1, risk_score=0.15)
    conn.commit()

    risk = get_session_risk(conn, VALID_SESS_1)
    assert risk is not None
    assert pytest.approx(risk["risk_score"], 0.001) == 0.15
    assert pytest.approx(risk["fsm_score"], 0.001) == 0.0
    assert pytest.approx(risk["ml_score"], 0.001) == 0.0
    assert json.loads(risk["reasons"]) == []
    assert "T" in risk["updated_at"]
    conn.close()


def test_session_risk_reasons_json_string_input(temp_db):
    """Reasons passed as an already-serialized JSON string should be validated and preserved."""
    conn = get_connection(temp_db)
    upsert_session(conn, VALID_SESS_1)
    conn.commit()

    raw_json = json.dumps(["factor_1", "factor_2"])
    upsert_session_risk(conn, VALID_SESS_1, 0.5, reasons=raw_json)
    conn.commit()

    risk = get_session_risk(conn, VALID_SESS_1)
    assert json.loads(risk["reasons"]) == ["factor_1", "factor_2"]
    conn.close()


def test_get_session_risk_missing_session(temp_db):
    """Querying risk for an unrecorded session returns None."""
    conn = get_connection(temp_db)
    assert get_session_risk(conn, "missing-session-uuid") is None
    assert get_session_risk(conn, "") is None
    conn.close()


def test_session_risk_foreign_key_violation(temp_db):
    """Upserting risk for a session that does not exist in sessions table must fail with IntegrityError."""
    conn = get_connection(temp_db)
    with pytest.raises(sqlite3.IntegrityError):
        upsert_session_risk(conn, "unregistered-session-id", 0.5)
    conn.close()


def test_session_risk_validation_errors(temp_db):
    """Score bounds (0.0 to 1.0) and parameter types must be strictly enforced."""
    conn = get_connection(temp_db)
    upsert_session(conn, VALID_SESS_1)
    conn.commit()

    # Invalid session_id
    with pytest.raises(ValueError):
        upsert_session_risk(conn, "", 0.5)

    # Score out of range
    with pytest.raises(ValueError):
        upsert_session_risk(conn, VALID_SESS_1, -0.1)
    with pytest.raises(ValueError):
        upsert_session_risk(conn, VALID_SESS_1, 1.01)
    with pytest.raises(ValueError):
        upsert_session_risk(conn, VALID_SESS_1, 0.5, fsm_score=1.5)
    with pytest.raises(ValueError):
        upsert_session_risk(conn, VALID_SESS_1, 0.5, ml_score=-0.05)

    # Non-numeric scores
    with pytest.raises(ValueError):
        upsert_session_risk(conn, VALID_SESS_1, "high")  # type: ignore
    with pytest.raises(ValueError):
        upsert_session_risk(conn, VALID_SESS_1, True)  # type: ignore

    # Invalid reasons type
    with pytest.raises(ValueError):
        upsert_session_risk(conn, VALID_SESS_1, 0.5, reasons=123)  # type: ignore

    conn.close()
