import os
import sqlite3
import pytest
from app import app
from backend.database.db import get_connection

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Each test runs against a temporary database (see conftest.py).
    with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def setup_teardown_db():
    conn = get_connection()
    # Create tables just in case
    with open("backend/database/schema.sql") as f:
        conn.executescript(f.read())
    conn.execute("DELETE FROM requests")
    conn.commit()
    conn.close()
    yield
    conn = get_connection()
    conn.execute("DELETE FROM requests")
    conn.commit()
    conn.close()

def test_basic_request_logging(client):
    response = client.get("/")
    assert response.status_code == 200

    conn = get_connection()
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 1
    req = rows[0]
    assert req["session_id"] is None
    assert req["method"] == "GET"
    assert req["endpoint"] == "/"
    assert req["status_code"] == 200
    assert req["response_time_ms"] >= 0

def test_session_association(client):
    conn = get_connection()
    from backend.database.db import upsert_session
    upsert_session(conn, "test-session-123")
    conn.commit()
    conn.close()

    client.set_cookie("botshield_session_id", "test-session-123")
    response = client.get("/about")
    assert response.status_code == 200

    conn = get_connection()
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 1
    req = rows[0]
    assert req["session_id"] == "test-session-123"
    assert req["endpoint"] == "/about"

def test_static_asset_exclusion(client):
    # Make a dummy request that simulates a static file fetch
    # In testing, the file might not exist but the middleware should still ignore it
    response = client.get("/static/css/style.css")

    conn = get_connection()
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 0

def test_api_event_exclusion(client):
    response = client.post("/api/events", json={
        "session_id": "test-session-123",
        "events": [{"event_type": "PAGE_VIEW", "timestamp": "2026-09-21T10:30:15Z", "page": "/"}]
    })

    conn = get_connection()
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 0

def test_endpoint_normalization(client):
    # Assuming we have a product detail route that uses a number
    # Let's request an existing product or a 404 one
    response = client.get("/product/1")

    conn = get_connection()
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/product/<id>"

def test_failed_request_logging(client):
    # This should yield a 404
    response = client.get("/non-existent-page")
    assert response.status_code == 404

    conn = get_connection()
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0]["status_code"] == 404

def test_burst_request_observation(client):
    for _ in range(5):
        client.get("/products")

    conn = get_connection()
    cur = conn.execute("SELECT * FROM requests")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 5
