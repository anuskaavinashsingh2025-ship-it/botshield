import pytest
from flask import json
from automata.history import get_session_context, clear_session_context
from automata.states import State
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def reset_sessions():
    # Helper to clear in-memory sessions between tests
    clear_session_context("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    clear_session_context("11111111-2222-3333-4444-555555555555")

def test_initial_session_state(client):
    client.set_cookie("botshield_session_id", "11111111-2222-3333-4444-555555555555")
    response = client.get("/api/automata/status")
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert data["session_id"] == "11111111-2222-3333-4444-555555555555"
    assert data["state"] == "START"
    assert data["score"] == 0.0
    assert data["risk_level"] == "LOW"
    assert "No behavioral activity recorded yet" in data["reasons"][0]

def test_no_session_state(client):
    # No cookie set
    response = client.get("/api/automata/status")
    assert response.status_code == 200
    data = response.get_json()
    
    assert data["session_id"] is None
    assert data["state"] == "START"
    assert data["score"] == 0.0

def test_score_changes_after_behavioral_events(client):
    # Setup database with a session
    from backend.database.db import get_connection, upsert_session
    conn = get_connection()
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    upsert_session(conn, sid)
    conn.commit()
    conn.close()

    client.set_cookie("botshield_session_id", sid)
    
    # Send some interaction events to /api/events
    events = [
        {"event_type": "PAGE_VIEW", "timestamp": "2026-09-21T10:00:00Z"},
        {"event_type": "SCROLL", "timestamp": "2026-09-21T10:00:05Z"},
        {"event_type": "CLICK", "timestamp": "2026-09-21T10:00:10Z"}
    ]
    resp = client.post("/api/events", json={"session_id": sid, "events": events})
    assert resp.status_code == 200
    
    # Check status
    status_resp = client.get("/api/automata/status")
    assert status_resp.status_code == 200
    data = status_resp.get_json()
    
    assert data["session_id"] == sid
    assert data["state"] in ("INTERACTIVE_BROWSING", "HUMAN_LIKE")
    assert 0.0 <= data["score"] <= 1.0
    assert len(data["reasons"]) > 0

def test_score_changes_after_rapid_requests(client):
    # Create the session in the DB
    from backend.database.db import get_connection, upsert_session
    conn = get_connection()
    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    upsert_session(conn, sid)
    conn.commit()
    conn.close()

    client.set_cookie("botshield_session_id", sid)
    
    # Manually trigger requests in rapid succession
    client.get("/about")
    client.get("/about")
    client.get("/about")
    client.get("/about")
    
    status_resp = client.get("/api/automata/status")
    data = status_resp.get_json()
    
    # Because we hit the same endpoint 4 times instantly, we should trigger both rapid and repeated
    assert data["state"] in ("RAPID_REQUESTS", "REPEATED_REQUESTS", "SUSPICIOUS_BEHAVIOR")
    assert data["score"] > 0.3
    assert data["risk_level"] in ("MEDIUM", "HIGH")

def test_product_pages_grouped_for_automata():
    # Moved from the retired Step 4 logger: product pages share one FSM endpoint.
    from app import _automata_endpoint
    assert _automata_endpoint("/product/1") == "/product/<id>"
    assert _automata_endpoint("/product/42") == "/product/<id>"
    assert _automata_endpoint("/product/abc") == "/product/abc"
    assert _automata_endpoint("/products") == "/products"
