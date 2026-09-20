"""
BotShield – Database helper (Step 3)

Thin wrapper around SQLite for session and event storage.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("botshield.db")

# Resolve paths relative to the project root (one level above backend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
_DEFAULT_DB_PATH = _PROJECT_ROOT / "botshield.db"

# Allowed event types – reject anything outside this set
VALID_EVENT_TYPES = {
    "PAGE_VIEW",
    "CLICK",
    "SCROLL",
    "MOUSE_ACTIVITY",
    "KEYBOARD_ACTIVITY",
    "PAGE_EXIT",
}


def _get_db_path() -> str:
    """Return the database file path, honouring BOTSHIELD_DB env var for testing."""
    return os.environ.get("BOTSHIELD_DB", str(_DEFAULT_DB_PATH))


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection with WAL mode and foreign-key enforcement."""
    path = db_path or _get_db_path()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | None = None) -> None:
    """Create tables and indexes from schema.sql if they don't exist."""
    conn = get_connection(db_path)
    try:
        with open(_SCHEMA_PATH) as f:
            conn.executescript(f.read())
        conn.commit()
        logger.info("Database initialised at %s", db_path or _get_db_path())
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def upsert_session(conn: sqlite3.Connection, session_id: str) -> None:
    """Create a new session or bump its last_seen timestamp."""
    now = _now_iso()
    conn.execute(
        """
        INSERT INTO sessions (session_id, created_at, last_seen)
        VALUES (?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET last_seen = excluded.last_seen
        """,
        (session_id, now, now),
    )


def get_session(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    cur = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
    return cur.fetchone()


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def insert_events(
    conn: sqlite3.Connection,
    session_id: str,
    events: list[dict],
) -> int:
    """
    Validate and insert a batch of events.

    Returns the number of events inserted.
    Raises ValueError for invalid payloads.
    """
    if not events:
        raise ValueError("events list is empty")

    rows = []
    for evt in events:
        event_type = evt.get("event_type", "")
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"invalid event_type: {event_type!r}")

        timestamp = evt.get("timestamp", "")
        if not timestamp:
            raise ValueError("event missing timestamp")

        page = evt.get("page", "")

        # Build metadata dict from all extra keys
        meta = {
            k: v
            for k, v in evt.items()
            if k not in ("event_type", "timestamp", "page")
        }
        rows.append((session_id, event_type, timestamp, page, json.dumps(meta)))

    conn.executemany(
        "INSERT INTO events (session_id, event_type, timestamp, page, metadata) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def get_events(
    conn: sqlite3.Connection,
    session_id: str,
    event_type: str | None = None,
) -> list[sqlite3.Row]:
    """Fetch events for a session, optionally filtered by type."""
    if event_type:
        cur = conn.execute(
            "SELECT * FROM events WHERE session_id = ? AND event_type = ? "
            "ORDER BY timestamp",
            (session_id, event_type),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
    return cur.fetchall()
