"""
BotShield – Flask Enforcement Middleware (Phase 6)

Intercepts incoming requests on configured sensitive endpoints, queries the
enforcement policy and rate limiter, logs decisions, and returns appropriate
HTTP 429 or 403 responses when active enforcement is enabled.
"""

from __future__ import annotations

import logging
import os
from flask import Flask, Response, current_app, g, jsonify, render_template_string, request

from backend.enforcement.limiter import BoundedRateLimiter
from backend.enforcement.policy import (
    EnforcementAction,
    EnforcementConfig,
    EnforcementMode,
)
from backend.enforcement.service import EnforcementDecision, EnforcementService

logger = logging.getLogger("botshield.enforcement.middleware")

_CHALLENGE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Access Suspended – BotShield</title>
    <style>
        body { background: #0e0e10; color: #f0f0f2; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .card { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 2.5rem; max-width: 480px; text-align: center; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
        h1 { color: #f87171; font-size: 1.5rem; margin-top: 0; }
        p { color: #a1a1aa; line-height: 1.6; font-size: 0.95rem; }
        .meta { margin-top: 1.5rem; font-size: 0.8rem; color: #71717a; border-top: 1px solid #27272a; padding-top: 1rem; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Access Restricted</h1>
        <p>{{ reason }}</p>
        <p>If you believe this is an error, please return to the storefront or contact security support.</p>
        <div class="meta">
            BotShield Protection | Incident Ref: {{ session_id }}
        </div>
    </div>
</body>
</html>
"""


def _get_client_ip() -> str:
    """Extract client IP address safely, respecting reverse-proxy headers if present."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "127.0.0.1"


def init_enforcement_middleware(
    app: Flask,
    config: EnforcementConfig | None = None,
    service: EnforcementService | None = None,
) -> EnforcementService:
    """
    Register risk-based enforcement hooks onto a Flask application.

    Reads defaults from app.config:
    - ENFORCEMENT_MODE: 'monitor' (default), 'enforce', or 'disabled'
    - ENFORCEMENT_FAIL_OPEN: True (default)
    - ENFORCEMENT_SENSITIVE_ROUTES: ('/login',)
    - ENFORCEMENT_EXCLUDED_ROUTES: ('/static', '/favicon.ico', '/api/events')
    - ENFORCEMENT_RATE_LIMIT_REQUESTS: 10
    - ENFORCEMENT_RATE_LIMIT_WINDOW: 10.0
    - ENFORCEMENT_CACHE_TTL: 60.0
    """
    if service is not None:
        active_service = service
    else:
        # Build config from app.config or environment
        mode_val = app.config.get(
            "ENFORCEMENT_MODE",
            os.environ.get("BOTSHIELD_ENFORCEMENT_MODE", "monitor"),
        ).lower()
        mode = EnforcementMode(mode_val) if mode_val in [m.value for m in EnforcementMode] else EnforcementMode.MONITOR

        active_config = config or EnforcementConfig(
            mode=mode,
            fail_open=bool(app.config.get("ENFORCEMENT_FAIL_OPEN", True)),
            sensitive_routes=tuple(app.config.get("ENFORCEMENT_SENSITIVE_ROUTES", ("/login",))),
            excluded_routes=tuple(app.config.get("ENFORCEMENT_EXCLUDED_ROUTES", ("/static", "/favicon.ico", "/api/events"))),
            rate_limit_requests=int(app.config.get("ENFORCEMENT_RATE_LIMIT_REQUESTS", 10)),
            rate_limit_window_seconds=float(app.config.get("ENFORCEMENT_RATE_LIMIT_WINDOW", 10.0)),
            risk_cache_ttl_seconds=float(app.config.get("ENFORCEMENT_CACHE_TTL", 60.0)),
        )
        active_service = EnforcementService(config=active_config)

    app.extensions = getattr(app, "extensions", {})
    app.extensions["botshield_enforcement"] = active_service

    @app.before_request
    def _botshield_enforcement_check():
        if getattr(g, "_botshield_enforcement_checked", False):
            return None
        g._botshield_enforcement_checked = True

        try:
            # Resolve session identifier from Flask g (set by request logger) or cookies
            session_id = getattr(g, "botshield_session_id", None)
            if not session_id:
                cookie_name = app.config.get("BOTSHIELD_COOKIE_NAME", "botshield_session")
                session_id = request.cookies.get(cookie_name)

            client_ip = _get_client_ip()
            path = request.path
            db_path = app.config.get("BOTSHIELD_DB")

            decision = active_service.evaluate_request(
                session_id=session_id,
                client_ip=client_ip,
                path=path,
                db_path=db_path,
            )
            g.botshield_enforcement_decision = decision

            # In MONITOR mode, decision.enforced is guaranteed False
            if not decision.enforced:
                return None

            # Handle active enforcement responses
            wants_json = request.is_json or path.startswith("/api/") or "application/json" in request.headers.get("Accept", "")

            if decision.action == EnforcementAction.RATE_LIMIT:
                retry_seconds = decision.retry_after or 1
                if wants_json:
                    resp = jsonify({
                        "error": "Too Many Requests",
                        "reason": decision.reason,
                        "retry_after": retry_seconds,
                    })
                else:
                    html_content = render_template_string(
                        _CHALLENGE_HTML_TEMPLATE,
                        reason=decision.reason,
                        session_id=session_id or "anonymous",
                    )
                    resp = Response(html_content, mimetype="text/html")
                resp.status_code = 429
                resp.headers["Retry-After"] = str(retry_seconds)
                return resp

            if decision.action in (EnforcementAction.BLOCK, EnforcementAction.CHALLENGE):
                if wants_json:
                    resp = jsonify({
                        "error": "Forbidden",
                        "reason": decision.reason,
                    })
                else:
                    html_content = render_template_string(
                        _CHALLENGE_HTML_TEMPLATE,
                        reason=decision.reason,
                        session_id=session_id or "anonymous",
                    )
                    resp = Response(html_content, mimetype="text/html")
                resp.status_code = 403
                return resp

            return None

        except Exception as exc:
            logger.error("[ENFORCEMENT_MIDDLEWARE] Unexpected failure: %s", exc)
            if active_service.config.fail_open:
                return None  # Fail open: permit legitimate traffic
            return Response("Service Temporarily Unavailable", status=503)

    return active_service
