"""
BotShield – Step 3 telemetry tests.

Tests for the /api/events endpoint, session handling, and event storage.
"""

import json
import os
import sys
import tempfile

import pytest

# Ensure the project root is on the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.db import get_connection, get_events, get_session, init_db


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Each test gets its own temporary SQLite database."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("BOTSHIELD_DB", db_file)
    init_db(db_file)
    yield db_file


@pytest.fixture
def client():
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# A valid session ID (UUID v4 format).
SESS = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# -----------------------------------------------------------------------
# Session creation
# -----------------------------------------------------------------------
def test_first_event_creates_session(client, _use_temp_db):
    """Sending the first event should create a session."""
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": SESS,
            "events": [
                {"event_type": "PAGE_VIEW", "timestamp": "2026-09-15T10:00:00Z", "page": "/"}
            ],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["stored"] == 1

    conn = get_connection()
    session = get_session(conn, SESS)
    conn.close()
    assert session is not None
    assert session["session_id"] == SESS


# -----------------------------------------------------------------------
# Existing session update
# -----------------------------------------------------------------------
def test_existing_session_updates_last_seen(client, _use_temp_db):
    """A second event should update last_seen rather than create a new session."""
    payload = {
        "session_id": SESS,
        "events": [
            {"event_type": "PAGE_VIEW", "timestamp": "2026-09-15T10:00:00Z", "page": "/"}
        ],
    }
    client.post("/api/events", data=json.dumps(payload), content_type="application/json")

    # Second request
    payload["events"][0]["timestamp"] = "2026-09-15T10:05:00Z"
    client.post("/api/events", data=json.dumps(payload), content_type="application/json")

    conn = get_connection()
    session = get_session(conn, SESS)
    # Should still be only one session
    count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    assert count == 1
    assert session is not None


# -----------------------------------------------------------------------
# Event insertion
# -----------------------------------------------------------------------
def test_events_stored_correctly(client, _use_temp_db):
    """Events should be correctly stored with metadata."""
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": SESS,
            "events": [
                {
                    "event_type": "CLICK",
                    "timestamp": "2026-09-15T10:01:00Z",
                    "page": "/products",
                    "element_type": "button",
                    "element_id": "add-to-cart-btn",
                }
            ],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200

    conn = get_connection()
    events = get_events(conn, SESS, event_type="CLICK")
    conn.close()
    assert len(events) == 1
    meta = json.loads(events[0]["metadata"])
    assert meta["element_type"] == "button"
    assert meta["element_id"] == "add-to-cart-btn"


# -----------------------------------------------------------------------
# Multiple events in a batch
# -----------------------------------------------------------------------
def test_batch_multiple_events(client, _use_temp_db):
    """A batch with multiple events should all be accepted."""
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": SESS,
            "events": [
                {"event_type": "PAGE_VIEW", "timestamp": "2026-09-15T10:00:00Z", "page": "/"},
                {"event_type": "SCROLL", "timestamp": "2026-09-15T10:00:02Z", "page": "/", "scroll_depth": 25},
                {"event_type": "CLICK", "timestamp": "2026-09-15T10:00:03Z", "page": "/",
                 "element_type": "a", "element_id": "nav-products"},
                {"event_type": "MOUSE_ACTIVITY", "timestamp": "2026-09-15T10:00:04Z", "page": "/",
                 "activity_count": 42},
                {"event_type": "KEYBOARD_ACTIVITY", "timestamp": "2026-09-15T10:00:05Z", "page": "/login",
                 "activity_count": 7},
            ],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["stored"] == 5

    conn = get_connection()
    events = get_events(conn, SESS)
    conn.close()
    assert len(events) == 5


# -----------------------------------------------------------------------
# Invalid requests
# -----------------------------------------------------------------------
def test_invalid_json(client):
    resp = client.post("/api/events", data="not json", content_type="application/json")
    assert resp.status_code == 400


def test_missing_session_id(client):
    resp = client.post(
        "/api/events",
        data=json.dumps({"events": [{"event_type": "PAGE_VIEW", "timestamp": "t", "page": "/"}]}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "session_id" in resp.get_json()["error"]


def test_invalid_session_id_format(client):
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": "too-short",
            "events": [{"event_type": "PAGE_VIEW", "timestamp": "t", "page": "/"}],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_empty_events_list(client):
    resp = client.post(
        "/api/events",
        data=json.dumps({"session_id": SESS, "events": []}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_invalid_event_type(client):
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": SESS,
            "events": [{"event_type": "HACKING", "timestamp": "t", "page": "/"}],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "invalid event_type" in resp.get_json()["error"]


def test_missing_timestamp(client):
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": SESS,
            "events": [{"event_type": "PAGE_VIEW", "page": "/"}],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 400


# -----------------------------------------------------------------------
# Sensitive data safety
# -----------------------------------------------------------------------
def test_password_not_in_click_metadata(client, _use_temp_db):
    """Even if the tracker somehow sent a password field, it should be
    stored in metadata but the test verifies the tracker design should
    NOT send such fields. This test ensures the API doesn't crash and
    the stored metadata does not contain 'password' as a key we look for."""
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": SESS,
            "events": [
                {
                    "event_type": "CLICK",
                    "timestamp": "2026-09-15T10:01:00Z",
                    "page": "/login",
                    "element_type": "button",
                    "element_id": "login-btn",
                }
            ],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    conn = get_connection()
    events = get_events(conn, SESS, event_type="CLICK")
    conn.close()
    meta = json.loads(events[0]["metadata"])
    # No password or value field should be present
    assert "password" not in meta
    assert "value" not in meta


# -----------------------------------------------------------------------
# PAGE_EXIT event
# -----------------------------------------------------------------------
def test_page_exit_event(client, _use_temp_db):
    resp = client.post(
        "/api/events",
        data=json.dumps({
            "session_id": SESS,
            "events": [
                {"event_type": "PAGE_EXIT", "timestamp": "2026-09-15T10:10:00Z", "page": "/products"}
            ],
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    conn = get_connection()
    events = get_events(conn, SESS, event_type="PAGE_EXIT")
    conn.close()
    assert len(events) == 1
    assert events[0]["page"] == "/products"


def test_tracker_script_deduplicates_exit_handlers(client):
    """Verify static tracker.js contains exitHandled guard to prevent duplicate PAGE_EXIT beacons."""
    resp = client.get("/static/js/tracker.js")
    assert resp.status_code == 200
    js_text = resp.get_data(as_text=True)
    assert "exitHandled" in js_text
    assert "visibilitychange" in js_text
    assert "pagehide" in js_text
    assert "pageshow" in js_text
