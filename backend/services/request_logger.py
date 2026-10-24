import logging
import re
import time
import sqlite3
from datetime import datetime, timezone
from flask import request, g

from backend.database.db import get_connection, insert_request

logger = logging.getLogger("botshield.telemetry")

IGNORED_PREFIXES = (
    "/static/",
    "/favicon.ico",
    "/api/events",  # Exclude telemetry endpoints to avoid feedback loop
)

def _normalize_endpoint(path: str) -> str:
    # Normalize /product/123 to /product/<id>
    if path.startswith("/product/") and path[9:].isdigit():
        return "/product/<id>"
    return path

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def register_request_logger(app):
    @app.before_request
    def start_timer():
        g.start_time = time.perf_counter()

    @app.after_request
    def log_request(response):
        # Calculate response time
        start = getattr(g, "start_time", None)
        response_time_ms = 0.0
        if start is not None:
            response_time_ms = (time.perf_counter() - start) * 1000.0

        path = request.path

        # Ignore static and telemetry ingestion endpoints
        for prefix in IGNORED_PREFIXES:
            if path.startswith(prefix):
                return response

        normalized_path = _normalize_endpoint(path)
        session_id = request.cookies.get("botshield_session_id")
        method = request.method
        status_code = response.status_code
        timestamp = _now_iso()

        # Database insertion logic gracefully wrapped to prevent app failure
        conn = None
        try:
            conn = get_connection()
            try:
                insert_request(
                    conn=conn,
                    session_id=session_id,
                    timestamp=timestamp,
                    method=method,
                    endpoint=normalized_path,
                    status_code=status_code,
                    response_time_ms=response_time_ms
                )
            except sqlite3.IntegrityError:
                # If session_id foreign key fails (e.g. orphaned cookie), log as None
                insert_request(
                    conn=conn,
                    session_id=None,
                    timestamp=timestamp,
                    method=method,
                    endpoint=normalized_path,
                    status_code=status_code,
                    response_time_ms=response_time_ms
                )
            conn.commit()

            # Wire the HTTP request to the FSM engine if we have a valid session
            if session_id:
                from automata.fsm import process_event
                raw_event = {
                    "type": "REQUEST",
                    "endpoint": normalized_path,
                    "status_code": status_code,
                    "timestamp": timestamp,
                }
                process_event(session_id, raw_event)

        except Exception as e:
            logger.error("Failed to log request telemetry: %s", e)
        finally:
            if conn:
                conn.close()

        return response
