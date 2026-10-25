"""
BotShield – Security Administrator Authentication & Access Control (Phase 7)

Provides constant-time token verification, header/cookie token extraction,
and decorator-based route protection for admin dashboard and API endpoints.

Security Principles:
- Fails closed: If no administrator token is configured in environment or app config,
  access is strictly DENIED (HTTP 403 Forbidden).
- Constant-time comparison: Uses hmac.compare_digest to prevent timing attacks.
- Multi-channel credential resolution:
  1. Authorization: Bearer <token>
  2. X-Admin-Token: <token>
  3. botshield_admin_token cookie (for browser sessions)
  4. ?token=<token> query parameter
- Zero credential leakage: Tokens and secrets are never logged or returned in responses.
"""

from __future__ import annotations

import functools
import hmac
import logging
import os
from flask import (
    Request,
    Response,
    current_app,
    jsonify,
    redirect,
    render_template_string,
    request,
)

logger = logging.getLogger("botshield.admin.auth")

DEFAULT_ADMIN_COOKIE = "botshield_admin_token"

_LOGIN_PROMPT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BotShield – Admin Authentication</title>
    <style>
        body {
            background: #0e0e10;
            color: #f0f0f2;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 1rem;
        }
        .login-card {
            background: #18181b;
            border: 1px solid #27272a;
            border-radius: 8px;
            padding: 2.5rem;
            max-width: 420px;
            width: 100%;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }
        .brand {
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #c56a32;
            margin-bottom: 0.5rem;
        }
        h1 {
            font-size: 1.5rem;
            margin: 0 0 0.5rem 0;
            color: #ffffff;
        }
        p {
            color: #a1a1aa;
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }
        .error-banner {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            color: #fca5a5;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-bottom: 1.25rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: #d4d4d8;
            margin-bottom: 0.5rem;
        }
        input[type="password"] {
            width: 100%;
            box-sizing: border-box;
            background: #0e0e10;
            border: 1px solid #3f3f46;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="password"]:focus {
            border-color: #c56a32;
        }
        button {
            width: 100%;
            background: #c56a32;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover {
            background: #d97706;
        }
        .footer {
            margin-top: 2rem;
            font-size: 0.75rem;
            color: #71717a;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <div class="brand">BotShield</div>
        <h1>Security Operations</h1>
        <p>Restricted access. Please provide your administrator authentication token to proceed.</p>

        {% if error %}
        <div class="error-banner">{{ error }}</div>
        {% endif %}

        <form method="POST" action="/admin/login">
            <input type="hidden" name="next" value="{{ next_url }}">
            <div class="form-group">
                <label for="admin_token">Administrator Token</label>
                <input type="password" id="admin_token" name="admin_token" placeholder="Enter configured token" required autofocus>
            </div>
            <button type="submit">Authenticate & Enter Dashboard</button>
        </form>

        <div class="footer">
            BotShield Phase 7 &bull; Security Administrator Access
        </div>
    </div>
</body>
</html>
"""


def get_configured_admin_token() -> str | None:
    """
    Retrieve the configured administrator token from Flask config or environment.
    Returns None if no token has been explicitly configured.
    """
    token = current_app.config.get("ADMIN_TOKEN")
    if token and isinstance(token, str) and token.strip():
        return token.strip()

    token = current_app.config.get("BOTSHIELD_ADMIN_TOKEN")
    if token and isinstance(token, str) and token.strip():
        return token.strip()

    env_token = os.environ.get("BOTSHIELD_ADMIN_TOKEN")
    if env_token and env_token.strip():
        return env_token.strip()

    return None


def extract_admin_token(req: Request) -> str | None:
    """
    Extract token candidate from Authorization header, custom header, cookie, or query param.
    """
    # 1. Authorization: Bearer <token>
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        candidate = auth_header[7:].strip()
        if candidate:
            return candidate

    # 2. X-Admin-Token: <token>
    custom_header = req.headers.get("X-Admin-Token", "").strip()
    if custom_header:
        return custom_header

    # 3. Cookie: botshield_admin_token=<token>
    cookie_token = req.cookies.get(DEFAULT_ADMIN_COOKIE, "").strip()
    if cookie_token:
        return cookie_token

    # 4. Query parameter: ?token=<token>
    query_token = req.args.get("token", "").strip()
    if query_token:
        return query_token

    return None


def verify_admin_token(provided_token: str | None) -> tuple[bool, str]:
    """
    Validate provided token against configured secret using constant-time comparison.

    Returns:
        (is_valid, reason_code)
        reason_code is one of: 'OK', 'MISSING_TOKEN', 'INVALID_TOKEN', 'AUTH_NOT_CONFIGURED'
    """
    configured = get_configured_admin_token()

    # Fail closed if no token is configured on the server
    if not configured:
        return False, "AUTH_NOT_CONFIGURED"

    if not provided_token or not isinstance(provided_token, str):
        return False, "MISSING_TOKEN"

    provided_clean = provided_token.strip()
    if not provided_clean:
        return False, "MISSING_TOKEN"

    # Constant-time comparison prevents timing side-channel attacks
    match = hmac.compare_digest(
        provided_clean.encode("utf-8"),
        configured.encode("utf-8"),
    )

    if match:
        return True, "OK"
    return False, "INVALID_TOKEN"


# ---------------------------------------------------------------------------
# Local demo mode (no token) — opt-in, loopback-only, never in production
# ---------------------------------------------------------------------------
_LOOPBACK_ADDRESSES = frozenset({"127.0.0.1", "::1"})
_PROXY_HEADERS = ("X-Forwarded-For", "X-Real-IP", "Forwarded")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_demo_mode_enabled() -> bool:
    """
    True when BOTSHIELD_DEMO_MODE is set and the deployment is not marked production.

    BOTSHIELD_ENV=production always disables demo mode, so a stray variable can
    never remove authentication from a production deployment.
    """
    if os.environ.get("BOTSHIELD_ENV", "").strip().lower() == "production":
        return False
    configured = current_app.config.get("BOTSHIELD_DEMO_MODE")
    if configured is not None:
        return bool(configured)
    return _env_flag("BOTSHIELD_DEMO_MODE")


def is_demo_request(req: Request) -> bool:
    """
    Demo access applies only to requests made on this machine.

    Requests from any other address, or relayed by a proxy (which could make a
    remote client appear local), still require the administrator token.
    """
    if not is_demo_mode_enabled():
        return False
    if req.remote_addr not in _LOOPBACK_ADDRESSES:
        return False
    return not any(h in req.headers for h in _PROXY_HEADERS)


def require_admin_auth(f):
    """
    Decorator enforcing administrator authentication on route endpoints.
    Protects both JSON API routes and HTML view routes.
    In local demo mode, loopback requests are admitted without a token.
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if is_demo_request(request):
            return f(*args, **kwargs)

        token_candidate = extract_admin_token(request)
        is_valid, reason = verify_admin_token(token_candidate)

        if is_valid:
            return f(*args, **kwargs)

        is_api_request = (
            request.path.startswith("/admin/api/")
            or request.is_json
            or "application/json" in request.headers.get("Accept", "")
        )

        if is_api_request:
            if reason == "AUTH_NOT_CONFIGURED":
                return jsonify({
                    "error": "Forbidden",
                    "code": "ADMIN_AUTH_NOT_CONFIGURED",
                    "message": "Administrator authentication is not configured on this server.",
                }), 403

            return jsonify({
                "error": "Unauthorized",
                "code": "ADMIN_AUTH_REQUIRED",
                "message": "Valid administrator token is required to access this endpoint.",
            }), 401

        # For browser UI requests:
        if reason == "AUTH_NOT_CONFIGURED":
            return Response(
                "403 Forbidden: Administrator authentication is not configured. Access is denied.",
                status=403,
                mimetype="text/plain",
            )

        # Render admin login prompt
        error_msg = "Invalid administrator token. Please try again." if reason == "INVALID_TOKEN" else None
        next_url = request.full_path if request.query_string else request.path
        html = render_template_string(_LOGIN_PROMPT_TEMPLATE, error=error_msg, next_url=next_url)
        return Response(html, status=401, mimetype="text/html")

    return decorated_function
