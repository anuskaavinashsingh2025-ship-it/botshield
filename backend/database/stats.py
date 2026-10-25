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


def _row_value(row: sqlite3.Row, key: str) -> Any:
    """Read an optional column, tolerating rows selected without it."""
    return row[key] if key in row.keys() else None


def _traffic_source(label_source: str | None) -> dict[str, Any]:
    """Distinguish simulator-generated sessions from real visitor traffic."""
    if label_source and str(label_source).startswith("simulation:"):
        return {"simulated": True, "profile": str(label_source).split(":", 1)[1]}
    return {"simulated": False, "profile": None}


def _interpret_risk_row(row: sqlite3.Row) -> dict[str, Any]:
    """
    Derive scoring mode and the usable ML score for a session_risks row.

    Rows written by the hybrid pipeline store scoring_mode/ml_status explicitly.
    Older rows fall back to the original heuristic (ml_score == 0.0 => FSM-only).
    session_risks.ml_score is NOT NULL, so it holds 0.0 when ML did not run;
    that value is reported as None rather than as a real ML score.
    """
    _, _, ScoringMode, _ = _get_scoring_components()
    stored_mode = _row_value(row, "scoring_mode")
    ml_status = _row_value(row, "ml_status")
    raw_ml = float(row["ml_score"])

    if stored_mode:
        scoring_mode = stored_mode
        ml_score = raw_ml if ml_status == "available" and stored_mode == ScoringMode.HYBRID.value else None
    else:
        legacy_fsm_only = "FSM-only" in str(row["reasons"]) or raw_ml == 0.0
        scoring_mode = ScoringMode.FSM_ONLY.value if legacy_fsm_only else ScoringMode.HYBRID.value
        ml_score = None if legacy_fsm_only else raw_ml

    anomaly = _row_value(row, "ml_is_anomaly")
    return {
        "scoring_mode": scoring_mode,
        "ml_score": ml_score,
        "ml_status": ml_status,
        "fsm_state": _row_value(row, "fsm_state"),
        "ml_is_anomaly": None if anomaly is None else bool(anomaly),
    }


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
            COUNT(CASE WHEN scoring_mode = 'HYBRID'
                       OR (scoring_mode IS NULL AND ml_score > 0.0) THEN 1 END) AS hybrid_count,
            COUNT(CASE WHEN scoring_mode = 'FSM_ONLY'
                       OR (scoring_mode IS NULL AND (reasons LIKE '%FSM-only%' OR (ml_score = 0.0 AND risk_score > 0.0))) THEN 1 END) AS fsm_only_count,
            COUNT(CASE WHEN scoring_mode = 'INSUFFICIENT_EVIDENCE'
                       OR (scoring_mode IS NULL AND (reasons LIKE '%INSUFFICIENT_SESSION_EVIDENCE%' OR (risk_score = 0.0 AND reasons = '[]'))) THEN 1 END) AS insufficient_count
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
            sr.updated_at,
            sr.scoring_mode,
            sr.ml_status,
            sr.fsm_state,
            sr.ml_is_anomaly
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
        interp = _interpret_risk_row(r)

        results.append({
            "session_id": r["session_id"],
            "created_at": r["created_at"],
            "last_seen": r["last_seen"],
            "risk_score": norm_score,
            "composite_score": comp_score,
            "risk_level": level.value,
            "scoring_mode": interp["scoring_mode"],
            "fsm_score": float(r["fsm_score"]),
            "ml_score": interp["ml_score"],
            "ml_status": interp["ml_status"],
            "fsm_state": interp["fsm_state"],
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
    traffic_filter: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fetch paginated session records with optional risk filtering and safe column sorting.

    traffic_filter: 'SIMULATED' (sessions labeled by the traffic simulator) or
    'REAL' (everything else, i.e. visitor traffic); anything else = no filter.

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

    if traffic_filter:
        tf = traffic_filter.strip().upper()
        if tf == "SIMULATED":
            where_conditions.append("sl.source LIKE 'simulation:%'")
        elif tf == "REAL":
            where_conditions.append("(sl.source IS NULL OR sl.source NOT LIKE 'simulation:%')")

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    cur = conn.cursor()

    # 1. Total matching count
    count_sql = f"""
        SELECT COUNT(*)
        FROM sessions s
        LEFT JOIN session_risks sr ON s.session_id = sr.session_id
        LEFT JOIN session_labels sl ON s.session_id = sl.session_id
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
            sr.scoring_mode,
            sr.ml_status,
            sr.fsm_state,
            sr.ml_is_anomaly,
            sl.source AS label_source,
            (SELECT COUNT(*) FROM events e WHERE e.session_id = s.session_id) AS event_count,
            (SELECT COUNT(*) FROM requests r WHERE r.session_id = s.session_id) AS request_count
        FROM sessions s
        LEFT JOIN session_risks sr ON s.session_id = sr.session_id
        LEFT JOIN session_labels sl ON s.session_id = sl.session_id
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
            interp = _interpret_risk_row(r)
            ml_score = interp["ml_score"]
            scoring_mode = interp["scoring_mode"]
        else:
            norm_score = None
            comp_score = None
            risk_level = "UNASSESSED"
            reasons_list = []
            fsm_score = None
            ml_score = None
            scoring_mode = None
            interp = {"ml_status": None, "fsm_state": None, "ml_is_anomaly": None}

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
            "ml_status": interp["ml_status"],
            "fsm_state": interp["fsm_state"],
            "ml_is_anomaly": interp["ml_is_anomaly"],
            "traffic_source": _traffic_source(r["label_source"]),
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
        risk_updated_at = risk_row["updated_at"]
        interp = _interpret_risk_row(risk_row)
        ml_score = interp["ml_score"]
        scoring_mode = interp["scoring_mode"]
    else:
        norm_score = None
        comp_score = None
        risk_level = "UNASSESSED"
        reasons_list = []
        fsm_score = None
        ml_score = None
        risk_updated_at = None
        scoring_mode = None
        interp = {"ml_status": None, "fsm_state": None, "ml_is_anomaly": None}

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

    label_row = cur.execute(
        "SELECT source FROM session_labels WHERE session_id = ?", (clean_id,)
    ).fetchone()

    return {
        "session_id": session_row["session_id"],
        "created_at": session_row["created_at"],
        "last_seen": session_row["last_seen"],
        "traffic_source": _traffic_source(label_row["source"] if label_row else None),
        "risk_assessment": {
            "risk_score": norm_score,
            "composite_score": comp_score,
            "risk_level": risk_level,
            "scoring_mode": scoring_mode,
            "fsm_score": fsm_score,
            "ml_score": ml_score,
            "ml_status": interp["ml_status"],
            "fsm_state": interp["fsm_state"],
            "ml_is_anomaly": interp["ml_is_anomaly"],
            "reasons": reasons_list,
            "updated_at": risk_updated_at,
        },
        "risk_history": [
            {
                "evaluated_at": h["evaluated_at"],
                "fsm_score": h["fsm_score"],
                "ml_score": h["ml_score"],
                "hybrid_score": h["hybrid_score"],
                "risk_level": h["risk_level"],
                "scoring_mode": h["scoring_mode"],
                "ml_status": h["ml_status"],
                "fsm_state": h["fsm_state"],
            }
            for h in cur.execute(
                """
                SELECT * FROM (
                    SELECT * FROM session_risk_history WHERE session_id = ?
                    ORDER BY id DESC LIMIT 50
                ) ORDER BY id ASC
                """,
                (clean_id,),
            ).fetchall()
        ],
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


def _binary_metrics(pairs: list[tuple[int, bool]]) -> dict[str, Any]:
    """Confusion matrix and rates for (true_label, predicted_bot) pairs."""
    tp = sum(1 for y, p in pairs if y == 1 and p)
    fp = sum(1 for y, p in pairs if y == 0 and p)
    tn = sum(1 for y, p in pairs if y == 0 and not p)
    fn = sum(1 for y, p in pairs if y == 1 and not p)

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "samples": len(pairs),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": ratio(fp, fp + tn),
    }


def get_admin_analytics(conn: sqlite3.Connection, bins: int = 10) -> dict[str, Any]:
    """
    Compare FSM, ML and hybrid scores across evaluated sessions.

    Detector decision rules (a session is "flagged as bot" when):
      - FSM:    fsm_score >= high threshold (HIGH band, 0.70 by default)
      - ML:     the model's own anomaly decision (ml_is_anomaly)
      - Hybrid: hybrid classification is HIGH
    Sessions in INSUFFICIENT_EVIDENCE mode carry no evidence and are excluded
    from score averages, histograms and detector metrics.
    Precision/recall/F1/FPR are computed only over sessions with a ground-truth
    label in session_labels; ML metrics only over sessions ML actually scored.
    """
    HybridConfig, _, ScoringMode, _ = _get_scoring_components()
    cfg = HybridConfig()
    high = cfg.high_threshold / 100.0

    rows = conn.execute(
        """
        SELECT sr.*, sl.is_bot, sl.source AS label_source
        FROM session_risks sr
        LEFT JOIN session_labels sl ON sl.session_id = sr.session_id
        """
    ).fetchall()

    scored = []
    for r in rows:
        interp = _interpret_risk_row(r)
        if interp["scoring_mode"] == ScoringMode.INSUFFICIENT_EVIDENCE.value:
            continue
        scored.append({
            "fsm": float(r["fsm_score"]),
            "ml": interp["ml_score"],
            "hybrid": float(r["risk_score"]),
            "ml_flag": interp["ml_is_anomaly"],
            "label": None if r["is_bot"] is None else int(r["is_bot"]),
        })

    def histogram(values: list[float]) -> list[int]:
        counts = [0] * bins
        for v in values:
            counts[min(bins - 1, max(0, int(v * bins)))] += 1
        return counts

    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    fsm_vals = [s["fsm"] for s in scored]
    ml_vals = [s["ml"] for s in scored if s["ml"] is not None]
    hyb_vals = [s["hybrid"] for s in scored]

    fsm_flag = [s["fsm"] >= high for s in scored]
    hyb_flag = [s["hybrid"] >= high for s in scored]
    ml_scored = [s for s in scored if s["ml"] is not None and s["ml_flag"] is not None]

    labeled = [s for s in scored if s["label"] is not None]
    labeled_ml = [s for s in labeled if s["ml"] is not None and s["ml_flag"] is not None]

    label_counts = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(is_bot), 0) FROM session_labels"
    ).fetchone()

    return {
        "evaluated_with_evidence": len(scored),
        "ml_scored_sessions": len(ml_vals),
        "thresholds": {"low": cfg.low_threshold / 100.0, "high": high},
        "weights": {"fsm": cfg.fsm_weight, "ml": cfg.ml_weight},
        "averages": {"fsm": average(fsm_vals), "ml": average(ml_vals), "hybrid": average(hyb_vals)},
        "histograms": {
            "bins": bins,
            "fsm": histogram(fsm_vals),
            "ml": histogram(ml_vals),
            "hybrid": histogram(hyb_vals),
        },
        "detections": {
            "fsm": sum(fsm_flag),
            "ml": sum(1 for s in ml_scored if s["ml_flag"]),
            "hybrid": sum(hyb_flag),
            "fsm_and_ml": sum(1 for s in ml_scored if s["ml_flag"] and s["fsm"] >= high),
        },
        "labels": {
            "total": int(label_counts[0] or 0),
            "bots": int(label_counts[1] or 0),
            "humans": int((label_counts[0] or 0) - (label_counts[1] or 0)),
            "evaluated": len(labeled),
        },
        "labeled_metrics": {
            "fsm": _binary_metrics([(s["label"], s["fsm"] >= high) for s in labeled]),
            "ml": _binary_metrics([(s["label"], bool(s["ml_flag"])) for s in labeled_ml]),
            "hybrid": _binary_metrics([(s["label"], s["hybrid"] >= high) for s in labeled]),
        },
    }
