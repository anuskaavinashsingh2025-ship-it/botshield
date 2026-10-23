-- BotShield – SQLite Schema (Phase 2A)
-- Behavioral telemetry, server request logging, and session risk tables.

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    last_seen    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    event_type   TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,
    page         TEXT    NOT NULL DEFAULT '',
    metadata     TEXT    NOT NULL DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Server-observed HTTP request metadata
CREATE TABLE IF NOT EXISTS requests (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    method       TEXT    NOT NULL,
    path         TEXT    NOT NULL,
    status_code  INTEGER NOT NULL,
    timestamp    TEXT    NOT NULL,
    user_agent   TEXT    NOT NULL DEFAULT '',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
);

-- Persistent session risk score for FSM, ML, and combined explainable scoring
CREATE TABLE IF NOT EXISTS session_risks (
    session_id   TEXT PRIMARY KEY,
    risk_score   REAL    NOT NULL DEFAULT 0.0,
    fsm_score    REAL    NOT NULL DEFAULT 0.0,
    ml_score     REAL    NOT NULL DEFAULT 0.0,
    reasons      TEXT    NOT NULL DEFAULT '[]',
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

-- Indexes for efficient querying (telemetry, request correlation, risk lookup)
CREATE INDEX IF NOT EXISTS idx_events_session       ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp     ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type          ON events(event_type);

CREATE INDEX IF NOT EXISTS idx_requests_session     ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp   ON requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_requests_path        ON requests(path);

CREATE INDEX IF NOT EXISTS idx_session_risks_score  ON session_risks(risk_score);
