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

# Standard HTTP methods supported for request logging
VALID_HTTP_METHODS = {
    "GET",
    "POST",
    "PUT",
    "DELETE",
    "PATCH",
    "HEAD",
    "OPTIONS",
    "TRACE",
    "CONNECT",
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


# Columns added to session_risks after Phase 2A. CREATE TABLE IF NOT EXISTS does
# not alter existing tables, so older databases get them via ALTER TABLE ADD COLUMN.
_SESSION_RISKS_ADDED_COLUMNS = {
    "scoring_mode": "TEXT",
    "ml_status": "TEXT",
    "fsm_state": "TEXT",
    "ml_is_anomaly": "INTEGER",
}


def _migrate_session_risks(conn: sqlite3.Connection) -> None:
    """Add missing nullable hybrid-metadata columns; never rewrites existing rows."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(session_risks)")}
    for name, col_type in _SESSION_RISKS_ADDED_COLUMNS.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE session_risks ADD COLUMN {name} {col_type}")
            logger.info("Migrated session_risks: added column %s", name)


def init_db(db_path: str | None = None) -> None:
    """Create tables and indexes from schema.sql if they don't exist."""
    conn = get_connection(db_path)
    try:
        with open(_SCHEMA_PATH) as f:
            conn.executescript(f.read())
        _migrate_session_risks(conn)
        conn.commit()
        logger.info("Database initialised at %s", db_path or _get_db_path())
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Path & Header sanitization helpers
# ---------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    """
    Normalize URL path for request logging:
    - Strips query string and fragment
    - Ensures leading slash
    - Collapses duplicate slashes
    - Strips trailing slash (except for root '/')
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")

    # Strip query string and fragment
    cleaned = path.split("?", 1)[0].split("#", 1)[0].strip()
    if not cleaned:
        cleaned = "/"

    # Ensure leading slash
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned

    # Collapse duplicate slashes
    import re
    cleaned = re.sub(r"/+", "/", cleaned)

    # Strip trailing slash if longer than 1 character
    if len(cleaned) > 1 and cleaned.endswith("/"):
        cleaned = cleaned.rstrip("/")

    return cleaned[:500]


def sanitize_user_agent(user_agent: str | None, max_len: int = 500) -> str:
    """
    Sanitize User-Agent header:
    - Strips non-printable and control characters (e.g. CRLF) to prevent injection
    - Truncates to max_len
    - Returns empty string if None
    """
    if not user_agent:
        return ""
    # Filter printable characters (excluding carriage returns, line feeds, tabs)
    sanitized = "".join(
        ch for ch in str(user_agent)
        if ch.isprintable() and ch not in ("\r", "\n", "\t")
    )
    return sanitized.strip()[:max_len]


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


# ---------------------------------------------------------------------------
# Request helpers (Phase 2A)
# ---------------------------------------------------------------------------

def insert_request(
    conn: sqlite3.Connection,
    session_id: str | None,
    method: str,
    path: str,
    status_code: int,
    timestamp: str | None = None,
    user_agent: str | None = None,
) -> int:
    """
    Insert a server-observed HTTP request record.

    session_id: session UUID or None if unassociated
    method: HTTP method (e.g. GET, POST)
    path: request path (will be normalized without query string)
    status_code: HTTP response status code (100 - 599)
    timestamp: ISO8601 timestamp string or None (will generate current UTC timestamp)
    user_agent: raw User-Agent header (will be sanitized)

    Returns the inserted row ID.
    Raises ValueError for invalid parameters.
    """
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")
    method_upper = method.strip().upper()
    if method_upper not in VALID_HTTP_METHODS:
        raise ValueError(f"invalid or unsupported HTTP method: {method!r}")

    normalized_path = normalize_path(path)

    if not isinstance(status_code, int) or isinstance(status_code, bool):
        raise ValueError("status_code must be an integer")
    if not (100 <= status_code <= 599):
        raise ValueError(f"status_code out of valid HTTP range (100-599): {status_code}")

    sess_id = session_id.strip() if isinstance(session_id, str) and session_id.strip() else None
    ts = timestamp if (isinstance(timestamp, str) and timestamp.strip()) else _now_iso()
    ua = sanitize_user_agent(user_agent)

    cur = conn.execute(
        """
        INSERT INTO requests (session_id, method, path, status_code, timestamp, user_agent)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (sess_id, method_upper, normalized_path, status_code, ts, ua),
    )
    return cur.lastrowid


def get_requests(
    conn: sqlite3.Connection,
    session_id: str | None = None,
    limit: int = 100,
) -> list[sqlite3.Row]:
    """Fetch request records, optionally filtered by session_id, ordered chronologically."""
    if session_id:
        cur = conn.execute(
            "SELECT * FROM requests WHERE session_id = ? ORDER BY timestamp ASC, id ASC LIMIT ?",
            (session_id, limit),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM requests ORDER BY timestamp ASC, id ASC LIMIT ?",
            (limit,),
        )
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Session Risk helpers (Phase 2A)
# ---------------------------------------------------------------------------

def upsert_session_risk(
    conn: sqlite3.Connection,
    session_id: str,
    risk_score: float | None = None,
    fsm_score: float | None = None,
    ml_score: float | None = None,
    reasons: list[str] | str | None = None,
    updated_at: str | None = None,
    *,
    scoring_mode: str | None = None,
    ml_status: str | None = None,
    fsm_state: str | None = None,
    ml_is_anomaly: bool | None = None,
) -> None:
    """
    Insert or update the persistent risk score and explainability factors for a session.

    Scores must be floats between 0.0 and 1.0 (or None to preserve existing values).
    reasons may be a list of string factors, a JSON string array, or None.
    The keyword-only hybrid metadata fields are written only when provided.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    sess_id = session_id.strip()

    # Check for existing record to preserve fields if performing a partial update
    existing = get_session_risk(conn, sess_id)

    # Validate non-None scores
    for name, val in [("risk_score", risk_score), ("fsm_score", fsm_score), ("ml_score", ml_score)]:
        if val is not None:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ValueError(f"{name} must be a float between 0.0 and 1.0")
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0 (got {val})")

    if existing is not None:
        effective_fsm = float(fsm_score) if fsm_score is not None else float(existing["fsm_score"])
        effective_ml = float(ml_score) if ml_score is not None else float(existing["ml_score"])
        if risk_score is not None:
            effective_risk = float(risk_score)
        else:
            effective_risk = float(existing["risk_score"])
        if reasons is None:
            reasons_json = existing["reasons"]
        elif isinstance(reasons, list):
            reasons_json = json.dumps([str(r) for r in reasons])
        elif isinstance(reasons, str):
            try:
                parsed = json.loads(reasons)
                reasons_json = json.dumps(parsed if isinstance(parsed, list) else [str(parsed)])
            except (json.JSONDecodeError, ValueError):
                reasons_json = json.dumps([reasons])
        else:
            raise ValueError("reasons must be a list of strings, a JSON string, or None")
    else:
        effective_fsm = float(fsm_score) if fsm_score is not None else 0.0
        effective_ml = float(ml_score) if ml_score is not None else 0.0
        effective_risk = float(risk_score) if risk_score is not None else effective_fsm
        if reasons is None:
            reasons_json = "[]"
        elif isinstance(reasons, list):
            reasons_json = json.dumps([str(r) for r in reasons])
        elif isinstance(reasons, str):
            try:
                parsed = json.loads(reasons)
                reasons_json = json.dumps(parsed if isinstance(parsed, list) else [str(parsed)])
            except (json.JSONDecodeError, ValueError):
                reasons_json = json.dumps([reasons])
        else:
            raise ValueError("reasons must be a list of strings, a JSON string, or None")

    ts = updated_at if (isinstance(updated_at, str) and updated_at.strip()) else _now_iso()

    conn.execute(
        """
        INSERT INTO session_risks (session_id, risk_score, fsm_score, ml_score, reasons, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            risk_score = excluded.risk_score,
            fsm_score  = excluded.fsm_score,
            ml_score   = excluded.ml_score,
            reasons    = excluded.reasons,
            updated_at = excluded.updated_at
        """,
        (sess_id, effective_risk, effective_fsm, effective_ml, reasons_json, ts),
    )

    metadata = {
        "scoring_mode": scoring_mode,
        "ml_status": ml_status,
        "fsm_state": fsm_state,
        "ml_is_anomaly": None if ml_is_anomaly is None else int(bool(ml_is_anomaly)),
    }
    provided = {k: v for k, v in metadata.items() if v is not None}
    if provided:
        assignments = ", ".join(f"{k} = ?" for k in provided)
        conn.execute(
            f"UPDATE session_risks SET {assignments} WHERE session_id = ?",
            (*provided.values(), sess_id),
        )



def get_session_risk(
    conn: sqlite3.Connection,
    session_id: str,
) -> sqlite3.Row | None:
    """Fetch the risk record for a session, or None if it does not exist."""
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    cur = conn.execute(
        "SELECT * FROM session_risks WHERE session_id = ?",
        (session_id.strip(),),
    )
    return cur.fetchone()


def update_session_ml_risk(
    conn: sqlite3.Connection,
    session_id: str,
    ml_score: float,
    updated_at: str | None = None,
) -> None:
    """
    Update only the ml_score field in session_risks for the given session.

    Strictly preserves existing fsm_score, risk_score, and explainability reasons.
    If no record exists yet, creates one with the provided ml_score and defaults
    for fsm_score (0.0) and risk_score (0.0).

    Raises ValueError if session_id is empty or ml_score is outside [0.0, 1.0].
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    sess_id = session_id.strip()

    if not isinstance(ml_score, (int, float)) or isinstance(ml_score, bool):
        raise ValueError("ml_score must be a float between 0.0 and 1.0")
    if not (0.0 <= float(ml_score) <= 1.0):
        raise ValueError(f"ml_score must be between 0.0 and 1.0 (got {ml_score})")

    existing = get_session_risk(conn, sess_id)
    effective_fsm = float(existing["fsm_score"]) if existing is not None else 0.0
    effective_risk = float(existing["risk_score"]) if existing is not None else 0.0
    effective_reasons = existing["reasons"] if existing is not None else "[]"
    ts = updated_at if (isinstance(updated_at, str) and updated_at.strip()) else _now_iso()

    conn.execute(
        """
        INSERT INTO session_risks (session_id, risk_score, fsm_score, ml_score, reasons, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            ml_score   = excluded.ml_score,
            updated_at = excluded.updated_at
        """,
        (sess_id, effective_risk, effective_fsm, float(ml_score), effective_reasons, ts),
    )


# ---------------------------------------------------------------------------
# Risk history & ground-truth label helpers (hybrid dashboard)
# ---------------------------------------------------------------------------

_HISTORY_MAX_ROWS_PER_SESSION = 200


def insert_risk_history(
    conn: sqlite3.Connection,
    session_id: str,
    fsm_score: float,
    ml_score: float | None,
    hybrid_score: float,
    risk_level: str,
    scoring_mode: str,
    ml_status: str,
    fsm_state: str | None = None,
    evaluated_at: str | None = None,
) -> None:
    """Append one hybrid evaluation, keeping the newest rows per session."""
    ts = evaluated_at if (isinstance(evaluated_at, str) and evaluated_at.strip()) else _now_iso()
    conn.execute(
        """
        INSERT INTO session_risk_history
            (session_id, evaluated_at, fsm_score, ml_score, hybrid_score,
             risk_level, scoring_mode, ml_status, fsm_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, ts, float(fsm_score), None if ml_score is None else float(ml_score),
         float(hybrid_score), risk_level, scoring_mode, ml_status, fsm_state),
    )
    conn.execute(
        """
        DELETE FROM session_risk_history
        WHERE session_id = ? AND id NOT IN (
            SELECT id FROM session_risk_history WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        )
        """,
        (session_id, session_id, _HISTORY_MAX_ROWS_PER_SESSION),
    )


def get_risk_history(
    conn: sqlite3.Connection,
    session_id: str,
    limit: int = 50,
) -> list[sqlite3.Row]:
    """Fetch a session's most recent evaluations, oldest first."""
    cur = conn.execute(
        """
        SELECT * FROM (
            SELECT * FROM session_risk_history WHERE session_id = ?
            ORDER BY id DESC LIMIT ?
        ) ORDER BY id ASC
        """,
        (session_id, max(1, min(200, int(limit)))),
    )
    return cur.fetchall()


def upsert_session_label(
    conn: sqlite3.Connection,
    session_id: str,
    is_bot: bool,
    source: str,
) -> None:
    """Record the ground-truth label for a session (e.g. simulator profile)."""
    conn.execute(
        """
        INSERT INTO session_labels (session_id, is_bot, source, labeled_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            is_bot = excluded.is_bot, source = excluded.source, labeled_at = excluded.labeled_at
        """,
        (session_id, int(bool(is_bot)), str(source)[:100], _now_iso()),
    )


# ---------------------------------------------------------------------------
# Aggregation & Dashboard analytics helpers (Phase 7)
# ---------------------------------------------------------------------------
from backend.database.stats import (
    get_admin_analytics,
    get_admin_overview_stats,
    get_recent_high_risk_sessions,
    get_admin_sessions_paginated,
    get_admin_session_detail,
)
