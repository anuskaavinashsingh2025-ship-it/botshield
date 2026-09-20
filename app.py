"""
BotShield – Steps 1-3: Foundation + UI + Telemetry
A demo e-commerce site with anonymous behavioral telemetry.
"""

import json
import logging
import re

from flask import Flask, render_template, abort, request, jsonify

from backend.database.db import (
    get_connection,
    init_db,
    insert_events,
    upsert_session,
    VALID_EVENT_TYPES,
)

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


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/about")
def about():
    return render_template("about.html")


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
