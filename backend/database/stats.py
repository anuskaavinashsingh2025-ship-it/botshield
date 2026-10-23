"""
BotShield – Database Aggregation & Reporting Helpers (Phase 7)

Provides safe, parameterized, read-only analytics queries supporting the
Security Administrator Dashboard and admin API endpoints:
- Global traffic overview and telemetry metrics
- Risk level distribution (LOW, MEDIUM, HIGH, Unassessed)
- Bounded, paginated session queries with risk filtering and safe sorting
- Drill-down session inspection details with event & request summaries
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

ALLOWED_SORT_FIELDS: dict[str, str] = {
    "last_seen": "s.last_seen",
    "created_at": "s.created_at",
    "risk_score": "COALESCE(sr.risk_score, -1.0)",
    "session_id": "s.session_id",
}


def _get_scoring_components():
    """Lazy load scoring components to prevent circular imports on module initialization."""
    from backend.scoring.service import HybridConfig, RiskLevel, ScoringMode, classify_risk
    return HybridConfig, RiskLevel, ScoringMode, classify_risk


def _safe_json_loads(val: Any) -> list[str]:
    """Safely decode JSON string array or return string items."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(item) for item in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
            return [str(parsed)]
        except (json.JSONDecodeError, ValueError):
            return [val]
    return []


def get_admin_overview_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """
    Compute aggregate platform statistics from SQLite tables.

    Handles empty databases cleanly, returning zeroed counters without errors.
    """
    cur = conn.cursor()

    # 1. Total traffic counts
    cur.execute("SELECT COUNT(*) FROM sessions")
    total_sessions = int(cur.fetchone()[0])

    cur.execute("SELECT COUNT(*) FROM events")
    total_events = int(cur.fetchone()[0])

    cur.execute("SELECT COUNT(*) FROM requests")
    total_requests = int(cur.fetchone()[0])

    cur.execute("SELECT COUNT(*) FROM session_risks")
    total_evaluated = int(cur.fetchone()[0])

    unassessed_count = max(0, total_sessions - total_evaluated)

    # 2. Risk distribution breakdown
    cur.execute(
        """
        SELECT
            COUNT(CASE WHEN risk_score < 0.35 THEN 1 END) AS low_count,
            COUNT(CASE WHEN risk_score >= 0.35 AND risk_score < 0.70 THEN 1 END) AS medium_count,
            COUNT(CASE WHEN risk_score >= 0.70 THEN 1 END) AS high_count
        FROM session_risks
        """
    )
    dist_row = cur.fetchone()
    low_count = int(dist_row[0] or 0)
    med_count = int(dist_row[1] or 0)
    high_count = int(dist_row[2] or 0)

    # 3. Scoring modes breakdown
    cur.execute(
        """
        SELECT
            COUNT(CASE WHEN ml_score > 0.0 THEN 1 END) AS hybrid_count,
            COUNT(CASE WHEN reasons LIKE '%FSM-only%' OR (ml_score = 0.0 AND risk_score > 0.0) THEN 1 END) AS fsm_only_count,
            COUNT(CASE WHEN reasons LIKE '%INSUFFICIENT_SESSION_EVIDENCE%' OR (risk_score = 0.0 AND reasons = '[]') THEN 1 END) AS insufficient_count
        FROM session_risks
        """
    )
    modes_row = cur.fetchone()
    hybrid_count = int(modes_row[0] or 0)
    fsm_only_count = int(modes_row[1] or 0)
    insufficient_count = int(modes_row[2] or 0)

    return {
        "total_sessions": total_sessions,
        "total_events": total_events,
        "total_requests": total_requests,
        "total_evaluated_sessions": total_evaluated,
        "unassessed_sessions": unassessed_count,
        "risk_distribution": {
            "low": low_count,
            "medium": med_count,
            "high": high_count,
            "unassessed": unassessed_count,
        },
        "scoring_modes": {
            "hybrid": hybrid_count,
            "fsm_only": fsm_only_count,
            "insufficient_evidence": insufficient_count,
        },
    }


def get_recent_high_risk_sessions(
    conn: sqlite3.Connection,
    limit: int = 10,
    min_score: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Retrieve recently evaluated suspicious or high-risk sessions.

    Parameters:
        limit: Max sessions to return (bounded 1 to 100).
        min_score: Lower threshold for inclusion (default 0.35, i.e. MEDIUM or HIGH risk).
    """
    bounded_limit = max(1, min(100, int(limit)))
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            s.session_id,
            s.created_at,
            s.last_seen,
            sr.risk_score,
            sr.fsm_score,
            sr.ml_score,
            sr.reasons,
            sr.updated_at
        FROM session_risks sr
        JOIN sessions s ON s.session_id = sr.session_id
        WHERE sr.risk_score >= ?
        ORDER BY sr.updated_at DESC, s.session_id DESC
        LIMIT ?
        """,
        (float(min_score), bounded_limit),
    )

    rows = cur.fetchall()
    results = []
    HybridConfig, RiskLevel, ScoringMode, classify_risk = _get_scoring_components()
    cfg = HybridConfig()

    for r in rows:
        norm_score = float(r["risk_score"])
        comp_score = round(norm_score * 100.0, 1)
        level = classify_risk(comp_score, cfg)
        reasons_list = _safe_json_loads(r["reasons"])

        mode = ScoringMode.HYBRID
        if "FSM-only" in str(r["reasons"]) or float(r["ml_score"]) == 0.0:
            mode = ScoringMode.FSM_ONLY

        results.append({
            "session_id": r["session_id"],
            "created_at": r["created_at"],
            "last_seen": r["last_seen"],
            "risk_score": norm_score,
            "composite_score": comp_score,
            "risk_level": level.value,
            "scoring_mode": mode.value,
            "fsm_score": float(r["fsm_score"]),
            "ml_score": float(r["ml_score"]),
            "reasons": reasons_list,
            "updated_at": r["updated_at"],
        })

    return results


def get_admin_sessions_paginated(
    conn: sqlite3.Connection,
    limit: int = 25,
    offset: int = 0,
    risk_filter: str | None = None,
    sort_by: str = "last_seen",
    sort_order: str = "desc",
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetch paginated session records with optional risk filtering and safe column sorting.

    Returns:
        (session_records_list, total_matching_count)
    """
    try:
        val = int(limit)
        bounded_limit = 25 if val <= 0 else min(100, val)
    except (ValueError, TypeError):
        bounded_limit = 25
    bounded_offset = max(0, int(offset))

    # Allowlisted sorting column
    sort_column = ALLOWED_SORT_FIELDS.get(sort_by, "s.last_seen")
    direction = "ASC" if str(sort_order).lower() == "asc" else "DESC"

    # Construct safe risk filtering clause
    where_conditions: list[str] = []
    params: list[Any] = []

    if risk_filter:
        rf = risk_filter.strip().upper()
        if rf == "LOW":
            where_conditions.append("sr.risk_score IS NOT NULL AND sr.risk_score < 0.35")
        elif rf == "MEDIUM":
            where_conditions.append("sr.risk_score >= 0.35 AND sr.risk_score < 0.70")
        elif rf == "HIGH":
            where_conditions.append("sr.risk_score >= 0.70")
        elif rf == "UNASSESSED":
            where_conditions.append("sr.risk_score IS NULL")
        elif rf == "ALL":
            pass  # No filtering

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    cur = conn.cursor()

    # 1. Total matching count
    count_sql = f"""
        SELECT COUNT(*)
        FROM sessions s
        LEFT JOIN session_risks sr ON s.session_id = sr.session_id
        {where_clause}
    """
    cur.execute(count_sql, params)
    total_count = int(cur.fetchone()[0])

    # 2. Paginated rows
    query_sql = f"""
        SELECT
            s.session_id,
            s.created_at,
            s.last_seen,
            sr.risk_score,
            sr.fsm_score,
            sr.ml_score,
            sr.reasons,
            sr.updated_at AS risk_updated_at,
            (SELECT COUNT(*) FROM events e WHERE e.session_id = s.session_id) AS event_count,
            (SELECT COUNT(*) FROM requests r WHERE r.session_id = s.session_id) AS request_count
        FROM sessions s
        LEFT JOIN session_risks sr ON s.session_id = sr.session_id
        {where_clause}
        ORDER BY {sort_column} {direction}, s.session_id DESC
        LIMIT ? OFFSET ?
    """
    query_params = list(params) + [bounded_limit, bounded_offset]
    cur.execute(query_sql, query_params)
    rows = cur.fetchall()

    HybridConfig, RiskLevel, ScoringMode, classify_risk = _get_scoring_components()
    cfg = HybridConfig()
    records: list[dict[str, Any]] = []

    for r in rows:
        risk_score_raw = r["risk_score"]
        if risk_score_raw is not None:
            norm_score = float(risk_score_raw)
            comp_score = round(norm_score * 100.0, 1)
            risk_level = classify_risk(comp_score, cfg).value
            reasons_list = _safe_json_loads(r["reasons"])
            fsm_score = float(r["fsm_score"])
            ml_score = float(r["ml_score"])
            scoring_mode = (
                ScoringMode.FSM_ONLY.value
                if ("FSM-only" in str(r["reasons"]) or ml_score == 0.0)
                else ScoringMode.HYBRID.value
            )
        else:
            norm_score = None
            comp_score = None
            risk_level = "UNASSESSED"
            reasons_list = []
            fsm_score = None
            ml_score = None
            scoring_mode = None

        records.append({
            "session_id": r["session_id"],
            "created_at": r["created_at"],
            "last_seen": r["last_seen"],
            "risk_score": norm_score,
            "composite_score": comp_score,
            "risk_level": risk_level,
            "scoring_mode": scoring_mode,
            "fsm_score": fsm_score,
            "ml_score": ml_score,
            "reasons": reasons_list,
            "risk_updated_at": r["risk_updated_at"],
            "event_count": int(r["event_count"] or 0),
            "request_count": int(r["request_count"] or 0),
        })

    return records, total_count


def get_admin_session_detail(
    conn: sqlite3.Connection,
    session_id: str,
) -> dict[str, Any] | None:
    """
    Retrieve comprehensive security diagnostics for a single session.

    Never exposes sensitive form payloads, credentials, or raw cookies.
    Returns None if the session does not exist.
    """
    if not session_id or not isinstance(session_id, str):
        return None
    clean_id = session_id.strip()

    cur = conn.cursor()

    # 1. Fetch core session metadata
    cur.execute("SELECT * FROM sessions WHERE session_id = ?", (clean_id,))
    session_row = cur.fetchone()
    if session_row is None:
        return None

    # 2. Fetch risk assessment record
    cur.execute("SELECT * FROM session_risks WHERE session_id = ?", (clean_id,))
    risk_row = cur.fetchone()

    HybridConfig, RiskLevel, ScoringMode, classify_risk = _get_scoring_components()
    cfg = HybridConfig()
    if risk_row is not None:
        norm_score = float(risk_row["risk_score"])
        comp_score = round(norm_score * 100.0, 1)
        risk_level = classify_risk(comp_score, cfg).value
        reasons_list = _safe_json_loads(risk_row["reasons"])
        fsm_score = float(risk_row["fsm_score"])
        ml_score = float(risk_row["ml_score"])
        risk_updated_at = risk_row["updated_at"]
        scoring_mode = (
            ScoringMode.FSM_ONLY.value
            if ("FSM-only" in str(risk_row["reasons"]) or ml_score == 0.0)
            else ScoringMode.HYBRID.value
        )
    else:
        norm_score = None
        comp_score = None
        risk_level = "UNASSESSED"
        reasons_list = []
        fsm_score = None
        ml_score = None
        risk_updated_at = None
        scoring_mode = None

    # 3. Telemetry event summary
    cur.execute(
        """
        SELECT event_type, COUNT(*) as cnt
        FROM events
        WHERE session_id = ?
        GROUP BY event_type
        ORDER BY cnt DESC
        """,
        (clean_id,),
    )
    event_counts = {r["event_type"]: int(r["cnt"]) for r in cur.fetchall()}
    total_events = sum(event_counts.values())

    # 4. Request summary
    cur.execute(
        """
        SELECT method, status_code, COUNT(*) as cnt
        FROM requests
        WHERE session_id = ?
        GROUP BY method, status_code
        ORDER BY cnt DESC
        """,
        (clean_id,),
    )
    request_breakdown = [
        {"method": r["method"], "status_code": int(r["status_code"]), "count": int(r["cnt"])}
        for r in cur.fetchall()
    ]
    total_requests = sum(r["count"] for r in request_breakdown)

    # 5. Recent events (sanitized, max 20)
    cur.execute(
        """
        SELECT id, event_type, timestamp, page
        FROM events
        WHERE session_id = ?
        ORDER BY timestamp DESC, id DESC
        LIMIT 20
        """,
        (clean_id,),
    )
    recent_events = [
        {"id": r["id"], "event_type": r["event_type"], "timestamp": r["timestamp"], "page": r["page"]}
        for r in cur.fetchall()
    ]

    # 6. Recent requests (sanitized, max 20)
    cur.execute(
        """
        SELECT id, method, path, status_code, timestamp
        FROM requests
        WHERE session_id = ?
        ORDER BY timestamp DESC, id DESC
        LIMIT 20
        """,
        (clean_id,),
    )
    recent_requests = [
        {
            "id": r["id"],
            "method": r["method"],
            "path": r["path"],
            "status_code": int(r["status_code"]),
            "timestamp": r["timestamp"],
        }
        for r in cur.fetchall()
    ]

    return {
        "session_id": session_row["session_id"],
        "created_at": session_row["created_at"],
        "last_seen": session_row["last_seen"],
        "risk_assessment": {
            "risk_score": norm_score,
            "composite_score": comp_score,
            "risk_level": risk_level,
            "scoring_mode": scoring_mode,
            "fsm_score": fsm_score,
            "ml_score": ml_score,
            "reasons": reasons_list,
            "updated_at": risk_updated_at,
        },
        "telemetry_summary": {
            "total_events": total_events,
            "event_counts_by_type": event_counts,
            "recent_events": recent_events,
        },
        "request_summary": {
            "total_requests": total_requests,
            "breakdown": request_breakdown,
            "recent_requests": recent_requests,
        },
    }
