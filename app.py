"""
BotShield – Steps 1-3: Foundation + UI + Telemetry
A demo e-commerce site with anonymous behavioral telemetry.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

from flask import Flask, render_template, abort, request, jsonify

from backend.database.db import (
    get_connection,
    get_session,
    init_db,
    insert_events,
    upsert_session,
    VALID_EVENT_TYPES,
)
from backend.middleware.request_logger import init_request_logger
from backend.enforcement import init_enforcement_middleware
from backend.admin import admin_bp
from backend.admin.auth import is_demo_mode_enabled
from automata.fsm import process_event
from automata.history import find_session_context
from automata.scoring import calculate_score
from backend.scoring.service import HYBRID_FORMULA, HybridConfig, evaluate_session_risk
from backend.scoring.worker import init_app_background_worker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("botshield")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="static",
)

# Initialise the database on startup
init_db()

# Initialise HTTP request logging & session correlation middleware (Phase 2B)
init_request_logger(app)

# Initialise risk-based enforcement middleware in safe MONITOR mode (Phase 6)
init_enforcement_middleware(app)

# Register Security Administrator Dashboard Blueprint (Phase 7)
app.register_blueprint(admin_bp)

with app.app_context():
    if is_demo_mode_enabled():
        logger.warning(
            "[DEMO MODE] Admin dashboard is open WITHOUT a token for requests from "
            "this machine (127.0.0.1 / ::1) only. Never use demo mode in production."
        )

# ---------------------------------------------------------------------------
# Fake product catalogue (static data)
# ---------------------------------------------------------------------------
PRODUCTS = [
    {
        "id": 1,
        "name": "Wireless Mouse",
        "price": 1299,
        "category": "Input",
        "category_id": "input",
        "index": "01",
        "short": "Precision wireless control for everyday productivity.",
        "image": "wireless_mouse.jpg",
        "rating": 4.8,
        "reviews": 120,
        "colors": ["#1a1a1a", "#3a3a3a", "#c56a32"],
        "description": (
            "The Wireless Mouse is designed for modern productivity. With a "
            "comfortable ergonomic shape and reliable wireless connectivity, "
            "it's the perfect companion for work, study, and everyday use."
        ),
        "specs": {
            "Connectivity": "2.4 GHz Wireless / Bluetooth 5.1",
            "DPI": "800 / 1200 / 1600 adjustable",
            "Battery": "Rechargeable Li-ion, 60-day life",
            "Weight": "85g",
            "Compatibility": "Windows, macOS, Linux",
        },
        "featured": True,
    },
    {
        "id": 2,
        "name": "Mechanical Keyboard",
        "price": 3499,
        "category": "Desk",
        "category_id": "desk",
        "index": "02",
        "short": "Tactile. Responsive. Built for focus.",
        "image": "mechanical_keyboard.jpg",
        "rating": 4.9,
        "reviews": 205,
        "colors": ["#1a1a1a", "#2c2c2c", "#4a4a4a"],
        "description": (
            "A compact 75-percent mechanical keyboard with hot-swappable "
            "switches, PBT keycaps, and a detachable USB-C cable. "
            "Engineered for precision typing and extended work sessions."
        ),
        "specs": {
            "Layout": "75% Compact (84 keys)",
            "Switches": "Hot-swappable mechanical",
            "Keycaps": "Double-shot PBT",
            "Connection": "USB-C detachable / Bluetooth",
            "Battery": "4000 mAh rechargeable",
        },
        "featured": True,
    },
    {
        "id": 3,
        "name": "USB-C Hub",
        "price": 1899,
        "category": "Connectivity",
        "category_id": "connectivity",
        "index": "03",
        "short": "One hub. More possibilities.",
        "image": "usb_c_hub.jpg",
        "rating": 4.6,
        "reviews": 89,
        "colors": ["#8a8a8a", "#1a1a1a", "#c56a32"],
        "description": (
            "A versatile 7-in-1 USB-C hub that adds HDMI output, two USB-A "
            "3.0 ports, an SD card reader, a microSD slot, USB-C data, and "
            "100W power delivery pass-through to your laptop."
        ),
        "specs": {
            "Ports": "HDMI, 2× USB-A 3.0, SD, microSD, USB-C data, USB-C PD",
            "HDMI Output": "4K @ 60Hz",
            "Power Delivery": "100W pass-through",
            "Material": "Aluminium alloy",
            "Cable Length": "15 cm integrated",
        },
        "featured": True,
    },
    {
        "id": 4,
        "name": "Laptop Stand",
        "price": 2399,
        "category": "Desk",
        "category_id": "desk",
        "index": "04",
        "short": "Elevated. Adjustable. Ergonomic.",
        "image": "laptop_stand.jpg",
        "rating": 4.7,
        "reviews": 64,
        "colors": ["#8a8a8a", "#1a1a1a"],
        "description": (
            "A sturdy adjustable aluminium laptop stand that raises your "
            "screen to eye level. Supports laptops from 10 to 17 inches and "
            "folds flat for easy portability."
        ),
        "specs": {
            "Material": "Aircraft-grade aluminium",
            "Compatibility": "10–17 inch laptops",
            "Adjustable Height": "6 angles up to 45°",
            "Weight Capacity": "15 kg",
            "Foldable": "Yes, flat-pack for travel",
        },
        "featured": False,
    },
    {
        "id": 5,
        "name": "Webcam HD 1080p",
        "price": 2999,
        "category": "Video",
        "category_id": "video",
        "index": "05",
        "short": "Crystal-clear video for every call.",
        "image": "webcam_hd.jpg",
        "rating": 4.5,
        "reviews": 152,
        "colors": ["#1a1a1a", "#3a3a3a"],
        "description": (
            "A plug-and-play full HD 1080p webcam with a built-in noise-"
            "cancelling microphone, auto-focus, and a privacy shutter. "
            "Works with all major video-conferencing apps."
        ),
        "specs": {
            "Resolution": "1920 × 1080 @ 30fps",
            "Microphone": "Dual stereo, noise-cancelling",
            "Focus": "Auto-focus with face tracking",
            "Field of View": "90° wide angle",
            "Privacy": "Physical shutter",
        },
        "featured": False,
    },
    {
        "id": 6,
        "name": "Monitor Light Bar",
        "price": 1799,
        "category": "Accessories",
        "category_id": "accessories",
        "index": "06",
        "short": "Better light. Less glare.",
        "image": "monitor_light_bar.jpg",
        "rating": 4.7,
        "reviews": 98,
        "colors": ["#1a1a1a", "#3a3a3a"],
        "description": (
            "An asymmetric LED light bar that clips onto your monitor to "
            "illuminate your desk without screen glare. Features adjustable "
            "brightness and colour temperature from warm to cool white."
        ),
        "specs": {
            "LED Type": "80 CRI asymmetric",
            "Colour Temperature": "2700 K – 6500 K",
            "Brightness": "Stepless dimming",
            "Power": "USB powered (5V/1A)",
            "Mount": "Gravity clamp, fits 1–3 cm bezels",
        },
        "featured": False,
    },
]

_PRODUCTS_BY_ID = {p["id"]: p for p in PRODUCTS}
CATEGORIES = ["All", "Input", "Desk", "Connectivity", "Video", "Accessories"]

# Simple UUID-ish pattern for session ID validation
_SESSION_ID_RE = re.compile(r"^[a-f0-9\-]{20,50}$")
_MAX_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    featured = [p for p in PRODUCTS if p["featured"]]
    return render_template("index.html", featured=featured, products=PRODUCTS)


@app.route("/products")
def products():
    return render_template("products.html", products=PRODUCTS, categories=CATEGORIES)


@app.route("/product/<int:product_id>")
def product_detail(product_id):
    product = _PRODUCTS_BY_ID.get(product_id)
    if product is None:
        abort(404)
    return render_template("product.html", product=product)


@app.route("/login", methods=["GET", "POST"])
def login():
    return render_template("login.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/_test/crash")
def _test_crash():
    if not app.config.get("TESTING"):
        abort(404)
    raise RuntimeError("Simulated crash for testing")


@app.route("/_test/redirect")
def _test_redirect():
    if not app.config.get("TESTING"):
        abort(404)
    from flask import redirect
    return redirect("/products", code=302)


# ---------------------------------------------------------------------------
# Telemetry API
# ---------------------------------------------------------------------------
@app.route("/api/events", methods=["POST"])
def api_events():
    """Accept a batch of behavioral events from tracker.js."""

    # Parse JSON body
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "invalid JSON"}), 400

    session_id = data.get("session_id", "")
    events = data.get("events")

    # --- Validate session_id ---
    if not session_id or not _SESSION_ID_RE.match(session_id):
        return jsonify({"error": "missing or invalid session_id"}), 400

    # --- Validate events list ---
    if not isinstance(events, list) or len(events) == 0:
        return jsonify({"error": "events must be a non-empty list"}), 400

    if len(events) > _MAX_BATCH_SIZE:
        return jsonify({"error": f"batch too large (max {_MAX_BATCH_SIZE})"}), 400

    # --- Store ---
    conn = get_connection()
    try:
        upsert_session(conn, session_id)
        count = insert_events(conn, session_id, events)
        conn.commit()

        # Wire the telemetry events to the FSM engine. Events are already
        # stored, so an automata failure must not turn this into an error.
        try:
            for evt in events:
                process_event(session_id, evt)
        except Exception as exc:
            logger.error("[AUTOMATA] Failed to process events: %s", exc)

        logger.info(
            "[TELEMETRY] session=%s events=%d types=%s",
            session_id[:12] + "…",
            count,
            ", ".join(sorted({e.get("event_type", "?") for e in events})),
        )
        return jsonify({"ok": True, "stored": count}), 200

    except ValueError as exc:
        conn.rollback()
        return jsonify({"error": str(exc)}), 400
    finally:
        conn.close()


def _automata_session_id() -> str | None:
    """
    Session ID used by the live automata monitor.

    tracker.js and the request logger share the `botshield_session` cookie;
    `botshield_session_id` is the original Step 5.1 cookie name.
    """
    for name in ("botshield_session_id", "botshield_session"):
        value = request.cookies.get(name)
        if value and _SESSION_ID_RE.match(value):
            return value
    return None


def _automata_endpoint(path: str) -> str:
    """Group product pages so crawling /product/1, /product/2, ... counts as repeated."""
    if path.startswith("/product/") and path[9:].isdigit():
        return "/product/<id>"
    return path


@app.after_request
def _feed_request_to_automata(response):
    """Feed page requests into the automata FSM (rapid/repeated/failed requests)."""
    path = request.path
    # Skip assets and APIs: the monitor polls /api/automata/status every 1.5 s,
    # which would otherwise register as repeated requests from the visitor.
    if path.startswith(("/static/", "/api/")) or path == "/favicon.ico":
        return response

    session_id = _automata_session_id()
    if session_id:
        try:
            process_event(session_id, {
                "type": "REQUEST",
                "endpoint": _automata_endpoint(path),
                "status_code": response.status_code,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            })
        except Exception as exc:
            logger.error("[AUTOMATA] Failed to process request: %s", exc)
    return response


@app.route("/api/automata/status", methods=["GET"])
def automata_status():
    """Return the current behavioral state and risk score for the session."""
    session_id = _automata_session_id()
    ctx = find_session_context(session_id) if session_id else None
    if ctx is None:
        return jsonify({
            "session_id": session_id,
            "state": "START",
            "score": 0.0,
            "risk_level": "LOW",
            "reasons": ["No behavioral activity recorded yet"]
        })

    score, reasons = calculate_score(ctx)

    risk_level = "LOW"
    if score >= 0.70:
        risk_level = "HIGH"
    elif score >= 0.40:
        risk_level = "MEDIUM"

    if not reasons or len(ctx.history) == 0:
        reasons = ["No behavioral activity recorded yet"]

    return jsonify({
        "session_id": session_id,
        "state": str(ctx.current_state),
        "score": score,
        "risk_level": risk_level,
        "reasons": reasons
    })


# ---------------------------------------------------------------------------
# Live hybrid risk for the Monitor panel
# ---------------------------------------------------------------------------
_RISK_CACHE_TTL_SECONDS = 1.0
_RISK_CACHE_MAX_ENTRIES = 1000
_risk_cache: dict[str, tuple[float, dict]] = {}


def _current_risk_payload(session_id: str) -> dict:
    """Run the same hybrid evaluation the scoring worker persists (without persisting)."""
    result = evaluate_session_risk(
        session_id=session_id,
        db_path=app.config.get("BOTSHIELD_DB"),
        model_path=app.config.get("BOTSHIELD_MODEL_PATH"),
        persist=False,
    )
    cfg = HybridConfig()
    ml_active = result.ml_score is not None
    ctx = find_session_context(session_id)
    automata_score = calculate_score(ctx)[0] if ctx is not None else None
    return {
        "session_id": session_id,
        "available": True,
        "evaluated_at": result.evaluated_at,
        "fsm": {
            "score": result.fsm_score,
            "state": result.fsm_result.final_state.value,
        },
        "ml": {
            "score": result.ml_score,
            "status": result.model_status.value,
            "active": ml_active,
            "is_anomaly": result.ml_result.is_anomaly if ml_active else None,
            "decision_threshold": result.ml_result.model_metadata.get("decision_threshold"),
        },
        "hybrid": {
            "score": result.normalized_score,
            "composite": result.composite_score,
            "risk_level": result.risk_level.value,
            "scoring_mode": result.scoring_mode.value,
            "weights": {"fsm": cfg.fsm_weight, "ml": cfg.ml_weight},
            "formula": HYBRID_FORMULA,
        },
        # FSM/ML explanation flags only; warnings can contain server paths.
        "reasons": result.reasons,
        "automata": {
            "state": str(ctx.current_state) if ctx is not None else "START",
            "score": automata_score,
        },
    }


@app.route("/api/risk/current", methods=["GET"])
def current_risk():
    """FSM, ML and hybrid scores for the caller's own session (Monitor panel)."""
    session_id = _automata_session_id()
    if not session_id:
        return jsonify({"session_id": None, "available": False,
                        "message": "No session yet."})

    now = time.monotonic()
    cached = _risk_cache.get(session_id)
    if cached and now - cached[0] < _RISK_CACHE_TTL_SECONDS:
        return jsonify(cached[1])

    try:
        conn = get_connection(app.config.get("BOTSHIELD_DB"))
        try:
            known = get_session(conn, session_id) is not None
        finally:
            conn.close()
        if not known:
            return jsonify({"session_id": session_id, "available": False,
                            "message": "Session has no recorded activity yet."})
        payload = _current_risk_payload(session_id)
    except Exception as exc:
        logger.error("[RISK] Failed to evaluate session %s: %s", session_id[:12], exc)
        return jsonify({"session_id": session_id, "available": False,
                        "message": "Risk evaluation failed."}), 503

    if len(_risk_cache) >= _RISK_CACHE_MAX_ENTRIES:
        _risk_cache.pop(next(iter(_risk_cache)))
    _risk_cache[session_id] = (now, payload)
    return jsonify(payload)


def _start_scoring_worker() -> None:
    """Persist FSM/ML/hybrid scores for all sessions in the background."""
    if os.environ.get("BOTSHIELD_SCORING_WORKER", "1").lower() in ("0", "false", "no"):
        logger.info("[ASYNC_WORKER] Disabled via BOTSHIELD_SCORING_WORKER")
        return
    init_app_background_worker(app, interval_seconds=5.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # With the debug reloader, only the serving child process runs the worker.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _start_scoring_worker()
    app.run(debug=True)
