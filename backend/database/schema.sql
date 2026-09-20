-- BotShield – SQLite Schema (Step 3)
-- Behavioral telemetry tables.

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

-- Indexes for efficient querying (ML feature extraction later)
CREATE INDEX IF NOT EXISTS idx_events_session   ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type      ON events(event_type);
