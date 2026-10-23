"""
BotShield – Security Administrator Web Routes and Protected APIs (Phase 7)

Provides protected dashboard views and JSON API endpoints:
- GET  /admin or /admin/dashboard: Main Security Operations Center UI
- POST /admin/login: Form-based token authentication
- GET  /admin/logout: Cookie-clearing session termination
- GET  /admin/api/stats: Platform overview and risk distribution statistics
- GET  /admin/api/sessions: Paginated session list with risk filter and sorting
- GET  /admin/api/sessions/<session_id>: Drill-down session diagnostics
"""

from __future__ import annotations

import logging
from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
)

from backend.admin.auth import (
    DEFAULT_ADMIN_COOKIE,
    require_admin_auth,
    verify_admin_token,
)
from backend.database.db import (
    get_admin_overview_stats,
    get_admin_session_detail,
    get_admin_sessions_paginated,
    get_connection,
    get_recent_high_risk_sessions,
)

logger = logging.getLogger("botshield.admin.routes")

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("")
@admin_bp.route("/")
@admin_bp.route("/dashboard")
@require_admin_auth
def dashboard_view():
    """Render the primary Security Operations Center (SOC) dashboard."""
    return render_template("admin/dashboard.html")


@admin_bp.route("/login", methods=["POST"])
def login_submit():
    """Handle administrator token submission from the login prompt."""
    token = (request.form.get("admin_token") or request.form.get("token") or "").strip()
    next_url = request.form.get("next", "/admin/dashboard")

    is_valid, reason = verify_admin_token(token)
    if not is_valid:
        error_msg = (
            "Administrator authentication is not configured on this server."
            if reason == "AUTH_NOT_CONFIGURED"
            else "Invalid administrator token. Please verify your credentials."
        )
        # Re-render login prompt with error
        from backend.admin.auth import _LOGIN_PROMPT_TEMPLATE
        from flask import render_template_string

        html = render_template_string(
            _LOGIN_PROMPT_TEMPLATE,
            error=error_msg,
            next_url=next_url,
        )
        return Response(html, status=401, mimetype="text/html")

    # Set authentication cookie
    resp = make_response(redirect(next_url))
    secure = current_app.config.get("BOTSHIELD_COOKIE_SECURE", False)
    resp.set_cookie(
        DEFAULT_ADMIN_COOKIE,
        token,
        max_age=86400 * 7,  # 7-day administrative session
        httponly=True,
        samesite="Lax",
        secure=secure,
    )
    return resp


@admin_bp.route("/logout")
def logout():
    """Clear administrator authentication cookie."""
    resp = make_response(redirect("/admin/dashboard"))
    resp.delete_cookie(DEFAULT_ADMIN_COOKIE)
    return resp


# ---------------------------------------------------------------------------
# Admin Protected APIs
# ---------------------------------------------------------------------------

@admin_bp.route("/api/stats", methods=["GET"])
@require_admin_auth
def api_stats():
    """Return platform overview counts and risk distribution breakdown."""
    try:
        conn = get_connection(current_app.config.get("BOTSHIELD_DB"))
        try:
            stats = get_admin_overview_stats(conn)
            recent_high = get_recent_high_risk_sessions(conn, limit=10, min_score=0.70)
        finally:
            conn.close()

        risk_counts = {
            "LOW": stats["risk_distribution"]["low"],
            "MEDIUM": stats["risk_distribution"]["medium"],
            "HIGH": stats["risk_distribution"]["high"],
            "UNASSESSED": stats["unassessed_sessions"],
        }
        scoring_modes = {
            "HYBRID": stats["scoring_modes"]["hybrid"],
            "FSM_ONLY": stats["scoring_modes"]["fsm_only"],
            "INSUFFICIENT_EVIDENCE": stats["scoring_modes"]["insufficient_evidence"],
        }
        stats["risk_counts"] = risk_counts
        stats["scoring_modes_upper"] = scoring_modes
        stats["recent_high_risk"] = recent_high

        return jsonify({
            "ok": True,
            "stats": stats,
            "total_sessions": stats["total_sessions"],
            "total_events": stats["total_events"],
            "total_requests": stats["total_requests"],
            "total_evaluated_sessions": stats["total_evaluated_sessions"],
            "unassessed_sessions": stats["unassessed_sessions"],
            "risk_counts": risk_counts,
            "risk_distribution": stats["risk_distribution"],
            "scoring_modes": scoring_modes,
            "recent_high_risk": recent_high,
        }), 200
    except Exception as exc:
        logger.error("[ADMIN_API] Failed to retrieve overview statistics: %s", exc)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Failed to calculate platform metrics.",
        }), 500


@admin_bp.route("/api/sessions", methods=["GET"])
@require_admin_auth
def api_sessions():
    """
    Return paginated sessions with optional risk filtering and sorting.

    Query parameters:
        limit: Max items to return (1 - 100, default 25)
        offset: Pagination offset (>= 0, default 0)
        risk_level: Optional filter ('LOW', 'MEDIUM', 'HIGH', 'UNASSESSED', 'ALL')
        sort_by: Column to sort by ('last_seen', 'created_at', 'risk_score', 'session_id')
        sort_order: Sort direction ('asc', 'desc')
    """
    # 1. Parse & validate limit
    raw_limit = request.args.get("limit", "25")
    try:
        val = int(raw_limit)
        limit = 25 if val <= 0 else min(100, val)
    except (ValueError, TypeError):
        limit = 25

    # 2. Parse & validate offset
    raw_offset = request.args.get("offset", "0")
    try:
        offset = max(0, int(raw_offset))
    except (ValueError, TypeError):
        offset = 0

    # 3. Parse & validate risk filter
    risk_level = request.args.get("risk_level")
    if risk_level:
        risk_level = risk_level.strip().upper()
        if risk_level not in ("ALL", "LOW", "MEDIUM", "HIGH", "UNASSESSED"):
            risk_level = None

    # 4. Parse & validate sort column and order
    sort_by = request.args.get("sort_by", "last_seen").strip().lower()
    if sort_by not in ("last_seen", "created_at", "risk_score", "session_id"):
        sort_by = "last_seen"

    sort_order = request.args.get("sort_order", "desc").strip().lower()
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"

    try:
        conn = get_connection(current_app.config.get("BOTSHIELD_DB"))
        try:
            sessions, total_count = get_admin_sessions_paginated(
                conn=conn,
                limit=limit,
                offset=offset,
                risk_filter=risk_level,
                sort_by=sort_by,
                sort_order=sort_order,
            )
        finally:
            conn.close()

        return jsonify({
            "ok": True,
            "sessions": sessions,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "filter": risk_level,
            "sort_by": sort_by,
            "sort_order": sort_order,
        }), 200
    except Exception as exc:
        logger.error("[ADMIN_API] Failed to retrieve paginated sessions: %s", exc)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Failed to retrieve session records.",
        }), 500


@admin_bp.route("/api/sessions/<session_id>", methods=["GET"])
@require_admin_auth
def api_session_detail(session_id: str):
    """
    Return detailed diagnostics and evidence for an individual session.

    Never leaks sensitive form payloads, credentials, or cookies.
    """
    if not session_id or not isinstance(session_id, str) or not session_id.strip():
        return jsonify({"error": "Bad Request", "message": "Invalid session identifier."}), 400

    clean_id = session_id.strip()

    try:
        conn = get_connection(current_app.config.get("BOTSHIELD_DB"))
        try:
            detail = get_admin_session_detail(conn, clean_id)
        finally:
            conn.close()

        if detail is None:
            return jsonify({
                "error": "Not Found",
                "message": f"Session '{clean_id}' does not exist.",
            }), 404

        return jsonify({
            "ok": True,
            "session": detail,
            "session_id": detail["session_id"],
            "created_at": detail["created_at"],
            "last_seen": detail["last_seen"],
            "risk_score": detail["risk_assessment"]["risk_score"],
            "risk_level": detail["risk_assessment"]["risk_level"],
            "fsm_score": detail["risk_assessment"]["fsm_score"],
            "ml_score": detail["risk_assessment"]["ml_score"],
            "evaluation_mode": detail["risk_assessment"]["scoring_mode"],
            "reasons": detail["risk_assessment"]["reasons"],
            "event_count": detail["telemetry_summary"]["total_events"],
            "request_count": detail["request_summary"]["total_requests"],
            "recent_events": detail["telemetry_summary"]["recent_events"],
            "recent_requests": detail["request_summary"]["recent_requests"],
        }), 200
    except Exception as exc:
        logger.error("[ADMIN_API] Failed to retrieve session detail for %s: %s", clean_id, exc)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Failed to load session details.",
        }), 500
