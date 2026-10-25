"""
BotShield – Server-Side HTTP Request Logger & Session Correlation (Phase 2B)

Captures server-observed HTTP request metadata:
- HTTP method (GET, POST, etc.)
- Normalized URL path (stripping query string, fragments, duplicate slashes)
- Response status code
- UTC timestamp
- Sanitized User-Agent
- Correlated Session ID (when verified in database)

Features:
- Configurable route exclusion (static assets, health checks)
- Session correlation using secure cookies and localStorage synchronization
- Untrusted session ID rejection (prevents database pollution)
- Sensitive data exclusion (no query parameters, forms, bodies, or auth tokens)
- Resilience against logger failures (fails open, never crashes user requests)
- Duplicate & recursive logging prevention
"""

import logging
import os
import re
import uuid
from flask import Flask, Request, Response, current_app, g, request
from werkzeug.exceptions import HTTPException

from backend.database.db import (
    get_connection,
    get_session,
    insert_request,
    normalize_path,
    sanitize_user_agent,
)

logger = logging.getLogger("botshield.request_logger")

# UUID v4-compatible pattern (matching app.py and tracker.js)
SESSION_ID_PATTERN = re.compile(r"^[a-f0-9\-]{20,50}$")
DEFAULT_COOKIE_NAME = "botshield_session"

# Internal polling that is not visitor traffic: the Monitor panel calls these
# every 1.5 s. Excluded regardless of REQUEST_LOGGING_EXCLUDE_PATHS, so the
# panel never feeds its own polling into FSM/ML features as "requests".
ALWAYS_EXCLUDED_PATHS = frozenset({"/api/automata/status", "/api/risk/current"})
# The admin dashboard polls its JSON APIs every 15 s; that is operator traffic,
# not the browsing behavior of whichever storefront session shares the cookie.
ALWAYS_EXCLUDED_PREFIXES = ("/admin/api/",)


def is_valid_session_id(session_id: str | None) -> bool:
    """Validate format of session ID string."""
    if not session_id or not isinstance(session_id, str):
        return False
    return bool(SESSION_ID_PATTERN.match(session_id.strip()))


def get_cookie_name() -> str:
    """Return the configured cookie name, prioritizing BOTSHIELD_COOKIE_NAME."""
    return current_app.config.get("BOTSHIELD_COOKIE_NAME", DEFAULT_COOKIE_NAME)


def resolve_session_id(req: Request) -> tuple[str, bool]:
    """
    Extract and validate candidate session ID from client cookie or header.

    Returns:
        (session_id, is_new)
        - If client sent a valid-format session ID: (session_id, False)
        - If client sent no session ID or invalid format: (new_uuid, True)
    """
    cookie_name = get_cookie_name()
    raw_cookie = req.cookies.get(cookie_name)

    if raw_cookie and is_valid_session_id(raw_cookie):
        return raw_cookie.strip(), False

    # Check optional custom header (useful for API clients & tests)
    header_val = req.headers.get("X-Session-ID")
    if header_val and is_valid_session_id(header_val):
        return header_val.strip(), False

    # Generate a fresh cryptographically random UUID v4
    return str(uuid.uuid4()), True


def get_verified_session_id(conn, candidate_id: str | None) -> str | None:
    """
    Check if candidate_id is known in the sessions table.
    Never trust a client-provided ID without verifying its existence.
    Returns candidate_id if present in DB, otherwise None.
    """
    if not candidate_id or not is_valid_session_id(candidate_id):
        return None
    try:
        session_row = get_session(conn, candidate_id)
        if session_row is not None:
            return candidate_id
    except Exception as exc:
        logger.error("[REQUEST_LOGGING] Failed to verify session in database: %s", exc)
    return None


def should_exclude_path(path: str) -> bool:
    """
    Determine if a request path should be excluded from request logging.
    Excludes:
    - Static assets (configurable via REQUEST_LOGGING_EXCLUDE_STATIC)
    - Custom excluded paths (REQUEST_LOGGING_EXCLUDE_PATHS)
    - Internal monitor polling (ALWAYS_EXCLUDED_PATHS)
    """
    if not current_app.config.get("REQUEST_LOGGING_ENABLED", True):
        return True

    if path in ALWAYS_EXCLUDED_PATHS or path.startswith(ALWAYS_EXCLUDED_PREFIXES):
        return True

    static_url = current_app.static_url_path or "/static"
    if current_app.config.get("REQUEST_LOGGING_EXCLUDE_STATIC", True):
        if path.startswith(static_url + "/") or path == static_url:
            return True

    excluded_paths = current_app.config.get("REQUEST_LOGGING_EXCLUDE_PATHS", {"/favicon.ico"})
    if path in excluded_paths:
        return True

    return False


def attach_session_cookie(response: Response, session_id: str) -> None:
    """Attach the session cookie to the response if not already matching."""
    cookie_name = get_cookie_name()
    if request.cookies.get(cookie_name) == session_id:
        return

    max_age = current_app.config.get("BOTSHIELD_COOKIE_MAX_AGE", 86400 * 30)
    path = current_app.config.get("BOTSHIELD_COOKIE_PATH", "/")
    samesite = current_app.config.get("BOTSHIELD_COOKIE_SAMESITE", "Lax")
    secure = current_app.config.get(
        "BOTSHIELD_COOKIE_SECURE",
        os.environ.get("BOTSHIELD_COOKIE_SECURE", "0").lower() in ("1", "true"),
    )
    httponly = current_app.config.get("BOTSHIELD_COOKIE_HTTPONLY", False)

    response.set_cookie(
        cookie_name,
        session_id,
        max_age=max_age,
        path=path,
        samesite=samesite,
        secure=secure,
        httponly=httponly,
    )



def log_request_record(status_code: int) -> None:
    """
    Record HTTP request metadata to the database.
    Guarantees:
    - At most one log entry per request
    - Non-blocking: failures are logged but never crash user requests
    - Excludes query parameters, request bodies, and sensitive data
    """
    if getattr(g, "_botshield_logged", False):
        return
    g._botshield_logged = True

    path = request.path
    if should_exclude_path(path):
        return

    method = request.method
    user_agent = request.headers.get("User-Agent", "")
    candidate_session_id = getattr(g, "botshield_session_id", None)
    verified_session_id = None

    try:
        # Honour app.config["BOTSHIELD_DB"] like the admin, enforcement and
        # scoring components; falls back to the BOTSHIELD_DB env var / default.
        conn = get_connection(current_app.config.get("BOTSHIELD_DB"))
        try:
            if candidate_session_id:
                verified_session_id = get_verified_session_id(conn, candidate_session_id)
                if verified_session_id is None:
                    # Untrusted or unknown session ID from client.
                    # Generate a fresh legitimate session ID for the response cookie.
                    g.botshield_session_id = str(uuid.uuid4())
                    g.botshield_session_is_new = True

            insert_request(
                conn=conn,
                session_id=verified_session_id,
                method=method,
                path=path,
                status_code=status_code,
                user_agent=user_agent,
            )
            conn.commit()
        except Exception as db_exc:
            conn.rollback()
            logger.error("[REQUEST_LOGGING] Database insertion failed: %s", db_exc)
        finally:
            conn.close()
    except Exception as exc:
        logger.error("[REQUEST_LOGGING] Connection failure: %s", exc)


def init_request_logger(app: Flask) -> None:
    """Register request logging lifecycle hooks onto a Flask app."""

    @app.before_request
    def _botshield_before_request():
        if getattr(g, "_botshield_in_before", False):
            return
        g._botshield_in_before = True

        try:
            session_id, is_new = resolve_session_id(request)
            g.botshield_session_id = session_id
            g.botshield_session_is_new = is_new

            # If this is /api/events, client may send session_id in JSON payload
            if request.path == "/api/events" and request.is_json:
                data = request.get_json(silent=True)
                if data and isinstance(data, dict):
                    body_sess = data.get("session_id")
                    if is_valid_session_id(body_sess):
                        g.botshield_session_id = body_sess.strip()
                        g.botshield_session_is_new = False
        except Exception as exc:
            logger.error("[REQUEST_LOGGING] Error in before_request: %s", exc)

    @app.after_request
    def _botshield_after_request(response: Response) -> Response:
        try:
            log_request_record(response.status_code)

            sess_id = getattr(g, "botshield_session_id", None)
            if sess_id and not should_exclude_path(request.path):
                attach_session_cookie(response, sess_id)
        except Exception as exc:
            logger.error("[REQUEST_LOGGING] Error in after_request: %s", exc)

        return response

    @app.teardown_request
    def _botshield_teardown_request(exception: BaseException | None = None) -> None:
        if exception is not None and not getattr(g, "_botshield_logged", False):
            status_code = 500
            if isinstance(exception, HTTPException):
                status_code = exception.code or 500
            try:
                log_request_record(status_code)
            except Exception as exc:
                logger.error("[REQUEST_LOGGING] Error in teardown_request: %s", exc)
