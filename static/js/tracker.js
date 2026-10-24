/*
 * BotShield – tracker.js (Step 3)
 *
 * Collects anonymous behavioral telemetry and sends it to /api/events.
 *
 * Events tracked:
 *   PAGE_VIEW, CLICK, SCROLL, MOUSE_ACTIVITY, KEYBOARD_ACTIVITY, PAGE_EXIT
 *
 * Privacy safeguards:
 *   - Session ID is a random UUID stored in localStorage (non-identifying)
 *   - No form values, passwords, keystrokes, or raw coordinates recorded
 *   - Mouse and keyboard activity recorded only as aggregate counts
 *   - Events are batched to minimise HTTP requests
 */

(function () {
  "use strict";

  // ----------------------------------------------------------------
  // Configuration
  // ----------------------------------------------------------------
  var API_URL = "/api/events";
  var BATCH_INTERVAL_MS = 3000;     // flush queue every 3 s
  var BATCH_MAX_SIZE = 20;          // flush if queue reaches this size
  var SCROLL_THROTTLE_MS = 500;
  var ACTIVITY_INTERVAL_MS = 4000;  // aggregate mouse/keyboard every 4 s
  var SESSION_KEY = "botshield_session_id";

  // ----------------------------------------------------------------
  // Session ID
  // ----------------------------------------------------------------
  function generateId() {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
    // Fallback: random hex string
    var hex = "";
    for (var i = 0; i < 32; i++) {
      hex += Math.floor(Math.random() * 16).toString(16);
    }
    return (
      hex.slice(0, 8) + "-" + hex.slice(8, 12) + "-" +
      hex.slice(12, 16) + "-" + hex.slice(16, 20) + "-" + hex.slice(20)
    );
  }

  function getSessionId() {
    var id;
    try {
      id = localStorage.getItem(SESSION_KEY);
    } catch (e) {
      // localStorage blocked
    }
    if (!id) {
      id = generateId();
      try { localStorage.setItem(SESSION_KEY, id); } catch (e) {}
    }

    // Set first-party cookie for server-side request telemetry
    document.cookie = SESSION_KEY + "=" + id + "; path=/; max-age=31536000; SameSite=Lax";

    return id;
  }

  var sessionId = getSessionId();
  var currentPage = window.location.pathname;

  // ----------------------------------------------------------------
  // Event queue & flushing
  // ----------------------------------------------------------------
  var queue = [];

  function enqueue(eventType, extra) {
    var evt = {
      event_type: eventType,
      timestamp: new Date().toISOString(),
      page: currentPage
    };
    if (extra) {
      for (var k in extra) {
        if (extra.hasOwnProperty(k)) {
          evt[k] = extra[k];
        }
      }
    }
    queue.push(evt);
    if (queue.length >= BATCH_MAX_SIZE) {
      flush();
    }
  }

  function flush() {
    if (queue.length === 0) return;
    var payload = JSON.stringify({ session_id: sessionId, events: queue });
    queue = [];
    // Prefer sendBeacon for reliability during page unload
    if (navigator.sendBeacon) {
      navigator.sendBeacon(API_URL, new Blob([payload], { type: "application/json" }));
    } else {
      var xhr = new XMLHttpRequest();
      xhr.open("POST", API_URL, true);
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.send(payload);
    }
  }

  // Periodic flush
  var flushTimer = setInterval(flush, BATCH_INTERVAL_MS);

  // ----------------------------------------------------------------
  // PAGE_VIEW
  // ----------------------------------------------------------------
  enqueue("PAGE_VIEW");

  // ----------------------------------------------------------------
  // CLICK tracking
  // ----------------------------------------------------------------
  document.addEventListener("click", function (e) {
    var target = e.target;
    // Walk up to find the nearest meaningful element
    var el = target;
    for (var i = 0; i < 5 && el && el !== document.body; i++) {
      if (el.id || el.tagName === "A" || el.tagName === "BUTTON" ||
          el.type === "submit") {
        break;
      }
      el = el.parentElement;
    }
    if (!el) el = target;

    var elementType = (el.tagName || "").toLowerCase();
    var elementId = el.id || el.getAttribute("data-track") ||
                    el.className.split(" ")[0] || "";

    // Safety: never capture input values
    enqueue("CLICK", {
      element_type: elementType,
      element_id: elementId.substring(0, 80)
    });
  }, true);

  // ----------------------------------------------------------------
  // SCROLL tracking (throttled)
  // ----------------------------------------------------------------
  var lastScrollSend = 0;

  function getScrollDepth() {
    var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    var docHeight = Math.max(
      document.body.scrollHeight, document.documentElement.scrollHeight,
      document.body.offsetHeight, document.documentElement.offsetHeight
    );
    var winHeight = window.innerHeight;
    if (docHeight <= winHeight) return 100;
    var pct = Math.round((scrollTop / (docHeight - winHeight)) * 100);
    // Round to nearest 5
    return Math.min(100, Math.round(pct / 5) * 5);
  }

  window.addEventListener("scroll", function () {
    var now = Date.now();
    if (now - lastScrollSend < SCROLL_THROTTLE_MS) return;
    lastScrollSend = now;
    enqueue("SCROLL", { scroll_depth: getScrollDepth() });
  }, { passive: true });

  // ----------------------------------------------------------------
  // MOUSE_ACTIVITY (aggregated)
  // ----------------------------------------------------------------
  var mouseCount = 0;

  document.addEventListener("mousemove", function () {
    mouseCount++;
  }, { passive: true });

  // ----------------------------------------------------------------
  // KEYBOARD_ACTIVITY (aggregated, NO key names/values)
  // ----------------------------------------------------------------
  var keyCount = 0;

  document.addEventListener("keydown", function (e) {
    // Safety: ensure we never record which key was pressed
    keyCount++;
  }, true);

  // Periodic emission of activity aggregates
  setInterval(function () {
    if (mouseCount > 0) {
      enqueue("MOUSE_ACTIVITY", { activity_count: mouseCount });
      mouseCount = 0;
    }
    if (keyCount > 0) {
      enqueue("KEYBOARD_ACTIVITY", { activity_count: keyCount });
      keyCount = 0;
    }
  }, ACTIVITY_INTERVAL_MS);

  // ----------------------------------------------------------------
  // PAGE_EXIT
  // ----------------------------------------------------------------
  function handleExit() {
    enqueue("PAGE_EXIT");
    flush();
  }

  document.addEventListener("visibilitychange", function () {
    if (document.visibilityState === "hidden") {
      handleExit();
    }
  });

  window.addEventListener("pagehide", handleExit);

  // Also flush on normal unload as a fallback
  window.addEventListener("beforeunload", function () {
    flush();
  });

})();
