"""
BotShield – FSM Detection Service (Phase 3)

Coordinates session data extraction from SQLite (events & requests),
chronological normalization with deterministic tie-breaking, FSM evaluation,
and persistence into the session_risks table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from backend.database.db import (
    get_connection,
    get_events,
    get_requests,
    normalize_path,
    upsert_session_risk,
)
from backend.detection.fsm import (
    BehavioralFSM,
    FSMConfig,
    FSMEvent,
    FSMResult,
    FSMSymbol,
    map_path_to_symbol,
    map_telemetry_to_symbol,
    parse_iso_to_epoch,
)

logger = logging.getLogger("botshield.detection.service")


def _get_source_priority(event: FSMEvent) -> int:
    """Deterministic tie-breaking priority when timestamps are equal."""
    if event.source == "HTTP_REQUEST":
        return 1
    if event.raw_type == "PAGE_VIEW":
        return 2
    if event.symbol == FSMSymbol.USER_INTERACTION or event.raw_type == "CLICK":
        return 3
    if event.symbol == FSMSymbol.PAGE_EXIT:
        return 4
    return 5


def normalize_session_records(
    event_rows: list[sqlite3.Row | dict[str, Any]],
    request_rows: list[sqlite3.Row | dict[str, Any]],
) -> tuple[list[FSMEvent], list[str]]:
    """
    Combine and normalize database rows into a strictly sorted, deduplicated
    sequence of FSMEvents with deterministic tie-breaking.
    """
    raw_events: list[FSMEvent] = []
    warnings: list[str] = []

    # 1. Normalize telemetry event rows
    for r in event_rows:
        row_id = r["id"] if "id" in r.keys() else 0
        raw_type = str(r["event_type"]) if r["event_type"] else ""
        raw_ts = str(r["timestamp"]) if r["timestamp"] else ""
        raw_page = str(r["page"]) if r["page"] else ""
        raw_meta = r["metadata"]

        # Parse metadata JSON safely
        meta_dict = {}
        if isinstance(raw_meta, dict):
            meta_dict = raw_meta
        elif isinstance(raw_meta, str) and raw_meta.strip():
            try:
                meta_dict = json.loads(raw_meta)
                if not isinstance(meta_dict, dict):
                    meta_dict = {}
                    warnings.append(f"WARNING_INVALID_METADATA_ROW_{row_id}")
            except (json.JSONDecodeError, ValueError):
                warnings.append(f"WARNING_INVALID_METADATA_ROW_{row_id}")

        epoch = parse_iso_to_epoch(raw_ts)
        if epoch is None:
            warnings.append(f"WARNING_MALFORMED_TIMESTAMP_EVENT_{row_id}")
            epoch = 0.0

        try:
            norm_page = normalize_path(raw_page) if raw_page else ""
        except ValueError:
            norm_page = raw_page

        symbol = map_telemetry_to_symbol(raw_type, norm_page, meta_dict)
        if symbol == FSMSymbol.UNKNOWN and raw_type not in ("PAGE_VIEW", "SCROLL", "CLICK", "MOUSE_ACTIVITY", "KEYBOARD_ACTIVITY", "PAGE_EXIT"):
            warnings.append(f"WARNING_UNKNOWN_EVENT_TYPE_{raw_type}")

        raw_events.append(
            FSMEvent(
                symbol=symbol,
                epoch_timestamp=epoch,
                iso_timestamp=raw_ts,
                source="TELEMETRY",
                path=norm_page,
                source_id=row_id,
                raw_type=raw_type,
                metadata=meta_dict,
            )
        )

    # 2. Normalize HTTP request rows
    for r in request_rows:
        row_id = r["id"] if "id" in r.keys() else 0
        method = str(r["method"]).upper() if r["method"] else "GET"
        raw_path = str(r["path"]) if r["path"] else ""
        raw_ts = str(r["timestamp"]) if r["timestamp"] else ""

        epoch = parse_iso_to_epoch(raw_ts)
        if epoch is None:
            warnings.append(f"WARNING_MALFORMED_TIMESTAMP_REQUEST_{row_id}")
            epoch = 0.0

        try:
            norm_path = normalize_path(raw_path)
        except ValueError:
            norm_path = raw_path

        symbol = map_path_to_symbol(norm_path, method=method)

        raw_events.append(
            FSMEvent(
                symbol=symbol,
                epoch_timestamp=epoch,
                iso_timestamp=raw_ts,
                source="HTTP_REQUEST",
                path=norm_path,
                source_id=row_id,
                raw_type=f"HTTP_{method}",
                metadata={"status_code": r["status_code"] if "status_code" in r.keys() else 200},
            )
        )

    # 3. Deterministic Sorting
    # Key: (epoch_timestamp, source_priority, source_id)
    raw_events.sort(key=lambda e: (e.epoch_timestamp, _get_source_priority(e), e.source_id))

    # 4. Deduplication
    deduped_events: list[FSMEvent] = []
    seen_fingerprints: set[tuple] = set()

    for e in raw_events:
        fp = (e.source, e.symbol, round(e.epoch_timestamp, 4), e.path)
        if fp in seen_fingerprints:
            warnings.append("WARNING_DUPLICATE_EVENT")
            continue
        seen_fingerprints.add(fp)
        deduped_events.append(e)

    return deduped_events, warnings


def evaluate_session_fsm(
    session_id: str,
    db_path: str | None = None,
    config: FSMConfig | None = None,
    persist: bool = True,
) -> FSMResult:
    """
    Extract a session's events and HTTP requests from SQLite, normalize them,
    run the behavioral FSM, and optionally persist the FSM score and reasons.

    Preserves existing ML and composite risk scores on partial updates.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    sess_id = session_id.strip()

    conn = get_connection(db_path)
    try:
        event_rows = get_events(conn, sess_id)
        request_rows = get_requests(conn, session_id=sess_id, limit=1000)

        normalized_events, warnings = normalize_session_records(event_rows, request_rows)

        fsm = BehavioralFSM(config=config)
        result = fsm.process_events(normalized_events)

        # Merge data quality warnings
        if warnings:
            existing_warns = result.evaluation_metadata.get("data_quality_warnings", [])
            result.evaluation_metadata["data_quality_warnings"] = sorted(set(existing_warns + warnings))

        if persist:
            upsert_session_risk(
                conn=conn,
                session_id=sess_id,
                fsm_score=result.risk_score,
                reasons=result.reasons,
            )
            conn.commit()

        return result
    finally:
        conn.close()
