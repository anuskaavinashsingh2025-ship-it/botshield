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
    is_demo_request,
    require_admin_auth,
    verify_admin_token,
)
from backend.database.db import (
    get_admin_analytics,
    get_admin_overview_stats,
    get_admin_session_detail,
    get_admin_sessions_paginated,
    get_connection,
    get_recent_high_risk_sessions,
)

logger = logging.getLogger("botshield.admin.routes")

# Offline evaluation is deterministic per model file; cache by (path, mtime, size).
_OFFLINE_EVAL_CACHE: dict[tuple, dict] = {}


def _model_overview() -> dict:
    """Live ML model availability, training metadata and decision threshold."""
    from backend.ml.model import MLModelStatus, get_model_path, load_model

    path = get_model_path(current_app.config.get("BOTSHIELD_MODEL_PATH"))
    bundle, status, message = load_model(path)
    overview = {
        "path": str(path),
        "status": status.value,
        "active": status == MLModelStatus.AVAILABLE,
        "message": message,
        "metadata": None,
        "decision_threshold": None,
        "score_semantics": (
            "Isolation Forest anomaly score s(x) in [0, 1]; higher = more anomalous. "
            "Not a calibrated bot probability."
        ),
    }
    if bundle is not None:
        meta = bundle.get("metadata", {})
        overview["metadata"] = {
            k: meta.get(k)
            for k in ("model_type", "trained_at", "training_sample_count",
                      "dataset_type", "contamination", "random_state", "schema_version")
        }
        forest = bundle["pipeline"].named_steps.get("forest")
        if forest is not None and hasattr(forest, "offset_"):
            overview["decision_threshold"] = round(-float(forest.offset_), 4)
    return overview


def _offline_evaluation(model: dict) -> dict | None:
    """Held-out synthetic benchmark from backend.ml.evaluation (labels are synthetic)."""
    if not model["active"]:
        return None
    from pathlib import Path
    from backend.ml.evaluation import evaluate_model_pipeline

    stat = Path(model["path"]).stat()
    key = (model["path"], stat.st_mtime_ns, stat.st_size)
    if key not in _OFFLINE_EVAL_CACHE:
        dataset = str((model.get("metadata") or {}).get("dataset_type") or "")
        if dataset.startswith("realistic_pipeline"):
            # Evaluate on the same kind of simulated sessions the model was trained on
            from backend.ml.realistic import evaluate_model_realistic
            report = evaluate_model_realistic(model["path"])
        else:
            report = evaluate_model_pipeline(model_path=model["path"])
        _OFFLINE_EVAL_CACHE.clear()
        _OFFLINE_EVAL_CACHE[key] = {
            "evaluation_name": report["evaluation_name"],
            "operating_threshold": report["operating_threshold"],
            "results": report["results"],
            "disclaimer": report["disclaimer"],
            "real_world_accuracy_claim": report["real_world_accuracy_claim"],
        }
    return _OFFLINE_EVAL_CACHE[key]

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("")
@admin_bp.route("/")
@admin_bp.route("/dashboard")
@require_admin_auth
def dashboard_view():
    """Render the primary Security Operations Center (SOC) dashboard."""
    return render_template("admin/dashboard.html", demo_mode=is_demo_request(request))


@admin_bp.route("/login", methods=["GET", "POST"])
def login_submit():
    """Handle administrator token submission from the login prompt."""
    # No token screen in local demo mode; production keeps token login.
    if is_demo_request(request) or request.method == "GET":
        return redirect("/admin/dashboard")

    token =(request.form.get("admin_token") or request.form.get("token") or "").strip()
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
    if is_demo_request(request):
        return redirect("/admin/dashboard")
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


@admin_bp.route("/api/analytics", methods=["GET"])
@require_admin_auth
def api_analytics():
    """Compare FSM, ML and hybrid scores; report model status and performance."""
    try:
        conn = get_connection(current_app.config.get("BOTSHIELD_DB"))
        try:
            analytics = get_admin_analytics(conn)
        finally:
            conn.close()
    except Exception as exc:
        logger.error("[ADMIN_API] Failed to compute analytics: %s", exc)
        return jsonify({
            "error": "Internal Server Error",
            "message": "Failed to compute score analytics.",
        }), 500

    try:
        model = _model_overview()
    except Exception as exc:
        logger.error("[ADMIN_API] Failed to inspect ML model: %s", exc)
        model = {"status": "error", "active": False, "message": "Model inspection failed."}

    offline = None
    offline_error = None
    try:
        offline = _offline_evaluation(model) if model.get("active") else None
    except Exception as exc:
        logger.error("[ADMIN_API] Offline ML evaluation failed: %s", exc)
        offline_error = "Offline evaluation failed."

    from backend.scoring.service import HYBRID_FORMULA
    from backend.scoring import worker as worker_module

    worker = worker_module._GLOBAL_WORKER
    return jsonify({
        "ok": True,
        "analytics": analytics,
        "formula": HYBRID_FORMULA,
        "model": model,
        "offline_evaluation": offline,
        "offline_evaluation_error": offline_error,
        "worker": worker.get_status() if worker is not None else {"running": False},
    }), 200


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

    # 5. Optional traffic filter: simulator-labeled vs real visitor sessions
    traffic = (request.args.get("traffic") or "").strip().upper()
    if traffic not in ("SIMULATED", "REAL"):
        traffic = None

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
                traffic_filter=traffic,
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
            "traffic": traffic,
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
            "ml_status": detail["risk_assessment"]["ml_status"],
            "fsm_state": detail["risk_assessment"]["fsm_state"],
            "risk_history": detail["risk_history"],
            "traffic_source": detail["traffic_source"],
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


# ---------------------------------------------------------------------------
# Live demo scenarios (local demo mode only)
# ---------------------------------------------------------------------------

def require_demo_mode(f):
    """Demo-only routes do not exist outside local demo mode (404, not 401)."""
    import functools

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not is_demo_request(request):
            return jsonify({"error": "Not Found"}), 404
        return f(*args, **kwargs)
    return wrapper


def _same_origin_json() -> bool:
    """
    Block cross-site requests to state-changing demo endpoints.

    A JSON body forces a CORS preflight for cross-origin pages, and an Origin
    header (sent by browsers on POST) must match this server.
    """
    if not request.is_json:
        return False
    origin = request.headers.get("Origin")
    return origin is None or origin.rstrip("/") == request.host_url.rstrip("/")


@admin_bp.route("/demo")
@require_demo_mode
def demo_view():
    """Live demo scenario page."""
    return render_template("admin/demo.html")


@admin_bp.route("/api/demo/scenarios", methods=["GET"])
@require_demo_mode
def demo_scenarios():
    from backend.simulation.live import list_scenarios
    return jsonify({"ok": True, "scenarios": list_scenarios()})


@admin_bp.route("/api/demo/runs", methods=["POST"])
@require_demo_mode
def demo_start_run():
    from backend.simulation.live import start_run

    if not _same_origin_json():
        return jsonify({"error": "Bad Request", "message": "JSON request from this page required."}), 400
    scenario = str((request.get_json(silent=True) or {}).get("scenario", ""))
    try:
        run = start_run(current_app._get_current_object(), scenario)
    except ValueError:
        return jsonify({"error": "Bad Request", "message": "Unknown scenario."}), 400
    except RuntimeError as exc:
        return jsonify({"error": "Too Many Requests", "message": str(exc)}), 429
    return jsonify({"ok": True, "run": run.to_dict()}), 201


@admin_bp.route("/api/demo/runs/<run_id>", methods=["GET"])
@require_demo_mode
def demo_run_status(run_id: str):
    from backend.simulation.live import get_run

    run = get_run(run_id)
    if run is None:
        return jsonify({"error": "Not Found"}), 404
    try:
        since = max(0, int(request.args.get("since", "0")))
    except ValueError:
        since = 0
    return jsonify({"ok": True, "run": run.to_dict(since=since)})


@admin_bp.route("/api/demo/runs/<run_id>/stop", methods=["POST"])
@require_demo_mode
def demo_stop_run(run_id: str):
    from backend.simulation.live import stop_run

    if not _same_origin_json():
        return jsonify({"error": "Bad Request"}), 400
    if not stop_run(run_id):
        return jsonify({"error": "Not Found"}), 404
    return jsonify({"ok": True})


@admin_bp.route("/api/demo/me", methods=["GET"])
@require_demo_mode
def demo_my_session():
    """Scores and explanations for this browser's own (real, unlabeled) session."""
    import re
    from backend.database.db import get_session
    from backend.simulation.live import explain_session

    session_id = request.cookies.get("botshield_session") or request.cookies.get("botshield_session_id")
    if not session_id or not re.match(r"^[a-f0-9\-]{20,50}$", session_id):
        return jsonify({"ok": True, "available": False, "message": "No storefront session in this browser yet."})
    conn = get_connection(current_app.config.get("BOTSHIELD_DB"))
    try:
        known = get_session(conn, session_id) is not None
    finally:
        conn.close()
    if not known:
        return jsonify({"ok": True, "available": False, "message": "Session has no recorded activity yet."})
    try:
        detail = explain_session(
            session_id,
            db_path=current_app.config.get("BOTSHIELD_DB"),
            model_path=current_app.config.get("BOTSHIELD_MODEL_PATH"),
        )
    except Exception as exc:
        logger.error("[DEMO] Failed to explain own session: %s", exc)
        return jsonify({"ok": False, "available": False, "message": "Evaluation failed."}), 503
    return jsonify({"ok": True, "available": True, "explanation": detail})
