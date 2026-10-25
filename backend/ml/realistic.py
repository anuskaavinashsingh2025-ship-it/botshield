"""
BotShield – Pipeline-faithful session generator for ML training and evaluation.

The original synthetic baseline (backend.ml.train.generate_synthetic_baseline)
draws feature vectors directly, and real sessions look nothing like them: every
real visitor scored as anomalous. This module instead generates the raw rows the
live pipeline stores — `requests` written by the request logger and `events`
written by /api/events — reproducing how tracker.js and the logger behave:

- a page load is a server request, then a PAGE_VIEW 80–600 ms later
- interactions while reading (SCROLL throttled to 500 ms, MOUSE_ACTIVITY
  aggregated every 4 s, CLICK, KEYBOARD_ACTIVITY on the login form)
- the telemetry queue is flushed every 3 s, each flush being a logged
  POST /api/events request; leaving a page adds PAGE_EXIT and a final flush
- the session ID lives in localStorage, so one "session" can span many visits

Rows then go through the real extract_session_features(). Humans are used for
training (including sessions cut off part-way, since the live pipeline scores
sessions in progress). Bots are generated only for evaluation.

Everything here is SIMULATED. Metrics computed on it measure separability under
these generators, not real-world accuracy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.ml.features import extract_session_features

PAGES = ["/", "/products", "/product/1", "/product/2", "/product/3",
         "/product/4", "/product/5", "/product/6", "/about", "/login"]
FLUSH_INTERVAL = 3.0
ACTIVITY_INTERVAL = 4.0
SCROLL_THROTTLE = 0.5

HUMAN = "human"
BOT_KINDS = ("scraper", "credential", "stealth", "headless")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass
class GeneratedSession:
    kind: str
    requests: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def request(self, t: float, path: str, method: str = "GET", status: int = 200) -> None:
        self.requests.append({"id": len(self.requests) + 1, "method": method, "path": path,
                              "status_code": status, "timestamp": _iso(t)})

    def event(self, t: float, event_type: str, page: str) -> None:
        self.events.append({"id": len(self.events) + 1, "event_type": event_type,
                            "page": page, "timestamp": _iso(t), "metadata": "{}"})

    def truncated(self, cutoff: float) -> "GeneratedSession":
        """The session as the scoring worker would see it at time `cutoff`."""
        def before(rows):
            return [r for r in rows if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp() <= cutoff]
        return GeneratedSession(self.kind, before(self.requests), before(self.events))

    def span(self) -> tuple[float, float]:
        stamps = [datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp()
                  for r in self.requests + self.events]
        return (min(stamps), max(stamps)) if stamps else (0.0, 0.0)

    def features(self) -> list[float]:
        return extract_session_features(events=self.events, requests=self.requests).feature_vector


class _Tracker:
    """Mimics tracker.js queueing and flushing for one open page."""

    def __init__(self, session: GeneratedSession):
        self.s = session
        self.pending = 0
        self.next_flush = None

    def enqueue(self, t: float, event_type: str, page: str) -> None:
        self.flush_due(t)
        self.s.event(t, event_type, page)
        self.pending += 1
        if self.next_flush is None:
            self.next_flush = t + FLUSH_INTERVAL

    def flush_due(self, t: float) -> None:
        while self.next_flush is not None and self.next_flush <= t:
            if self.pending:
                self.s.request(self.next_flush + 0.004, "/api/events", "POST")
                self.pending = 0
            self.next_flush = self.next_flush + FLUSH_INTERVAL if self.pending else None

    def flush_now(self, t: float) -> None:
        self.flush_due(t)
        if self.pending:
            self.s.request(t + 0.004, "/api/events", "POST")
        self.pending = 0
        self.next_flush = None


def _human_page(s: GeneratedSession, rng: random.Random, t: float, page: str, clicky: float) -> float:
    """
    One human page visit whose request arrives at t; returns the next request time.

    Timings follow a measured local browser session: PAGE_VIEW 10–60 ms after the
    request (occasionally slower), and on navigation the next page's request
    arrives before the old page's PAGE_EXIT and final flush.
    """
    status = 404 if rng.random() < 0.02 else 200
    s.request(t, page, status=status)
    tracker = _Tracker(s)
    loaded = t + (rng.uniform(0.01, 0.06) if rng.random() < 0.85 else rng.uniform(0.06, 0.4))
    tracker.enqueue(loaded, "PAGE_VIEW", page)

    dwell = min(120.0, max(2.5, rng.lognormvariate(2.4, 0.7)))  # median ~11 s
    leave = loaded + dwell
    now = loaded
    last_mouse = loaded
    while True:
        now += rng.expovariate(1.0 / rng.uniform(0.8, 3.5))
        if now >= leave:
            break
        roll = rng.random()
        if roll < 0.35:
            # a scroll gesture: several throttled SCROLL events
            for _ in range(rng.randint(1, 4)):
                tracker.enqueue(now, "SCROLL", page)
                now += SCROLL_THROTTLE + rng.uniform(0.0, 0.3)
        elif roll < 0.35 + 0.25 * clicky:
            tracker.enqueue(now, "CLICK", page)
        elif page == "/login" and roll < 0.75:
            tracker.enqueue(now, "KEYBOARD_ACTIVITY", page)
        if now - last_mouse >= ACTIVITY_INTERVAL and rng.random() < 0.8:
            tracker.enqueue(now, "MOUSE_ACTIVITY", page)
            last_mouse = now
        if rng.random() < 0.03:
            # tab hidden (switching tabs/windows): PAGE_EXIT + flush, then back
            tracker.enqueue(now, "PAGE_EXIT", page)
            tracker.flush_now(now)

    tracker.enqueue(leave, "CLICK", page)  # the link that navigates away
    next_request = leave + rng.uniform(0.005, 0.03)
    exit_at = next_request + rng.uniform(0.002, 0.01)
    tracker.enqueue(exit_at, "PAGE_EXIT", page)
    tracker.flush_now(exit_at)
    return next_request


def generate_human(rng: random.Random, start: float = 1_800_000_000.0) -> GeneratedSession:
    s = GeneratedSession(HUMAN)
    t = start
    clicky = rng.choice([0.3, 0.6, 1.0, 2.0])  # some people click a lot
    visits = 1 if rng.random() < 0.6 else rng.randint(2, 6)
    for v in range(visits):
        page = "/"
        for _ in range(rng.randint(1, 8)):
            t = _human_page(s, rng, t, page, clicky)
            if rng.random() < 0.08:
                continue  # reload the same page
            page = rng.choice(PAGES)
        t += rng.uniform(120.0, 7200.0)  # return later with the same localStorage ID
    return s


def generate_bot(kind: str, rng: random.Random, start: float = 1_800_000_000.0) -> GeneratedSession:
    s = GeneratedSession(kind)
    t = start
    if kind == "scraper":
        for i in range(rng.randint(6, 40)):
            s.request(t, PAGES[i % 8])
            t += rng.uniform(0.005, 0.08)
    elif kind == "credential":
        for _ in range(rng.randint(5, 30)):
            s.request(t, "/login", method=rng.choice(["GET", "POST"]))
            t += rng.uniform(0.02, 0.4)
    elif kind == "stealth":
        for i in range(rng.randint(5, 25)):
            s.request(t, rng.choice(PAGES[:9]))
            t += rng.uniform(0.6, 4.0)
    elif kind == "headless":
        # Browser automation that runs tracker.js but never scrolls or moves the mouse
        for i in range(rng.randint(5, 25)):
            page = PAGES[i % 9]
            s.request(t, page)
            tracker = _Tracker(s)
            tracker.enqueue(t + rng.uniform(0.05, 0.2), "PAGE_VIEW", page)
            t += rng.uniform(0.3, 1.5)
            tracker.enqueue(t, "PAGE_EXIT", page)
            tracker.flush_now(t)
            t += rng.uniform(0.02, 0.1)
    else:
        raise ValueError(f"unknown bot kind: {kind}")
    return s


def build_training_vectors(n_sessions: int = 400, seed: int = 42) -> list[list[float]]:
    """Feature vectors for simulated humans, including partially observed sessions."""
    rng = random.Random(seed)
    vectors = []
    for _ in range(n_sessions):
        session = generate_human(rng)
        first, last = session.span()
        vectors.append(session.features())
        # The worker scores sessions while they are still running, including
        # the first few seconds of a visit (one page request + a few events)
        for cutoff in (first + rng.uniform(0.05, 6.0),
                       first + rng.uniform(1.0, max(2.0, last - first)),
                       first + rng.uniform(1.0, max(2.0, last - first))):
            partial = session.truncated(cutoff)
            if partial.requests or partial.events:
                vectors.append(partial.features())
    return vectors


def build_evaluation_set(n_humans: int = 200, n_bots_each: int = 50, seed: int = 1042
                         ) -> tuple[list[list[float]], list[int], list[str]]:
    """Held-out labeled vectors (different seed from training): (X, y, kinds)."""
    rng = random.Random(seed)
    X, y, kinds = [], [], []
    for _ in range(n_humans):
        X.append(generate_human(rng).features()); y.append(0); kinds.append(HUMAN)
    for kind in BOT_KINDS:
        for _ in range(n_bots_each):
            X.append(generate_bot(kind, rng).features()); y.append(1); kinds.append(kind)
    return X, y, kinds


def evaluate_model_realistic(model_path: str | None = None, seed: int = 1042) -> dict[str, Any]:
    """
    Score held-out simulated sessions with the model's own anomaly decision.

    Uses the trained decision boundary (no threshold tuned on the test set) and
    reports overall and per-scenario detection rates.
    """
    from backend.ml.features import ExtractedFeatures, FEATURE_NAMES
    from backend.ml.model import IsolationForestModel, MLModelStatus

    model = IsolationForestModel(model_path=model_path)
    if model.status != MLModelStatus.AVAILABLE:
        raise RuntimeError(f"model unavailable: {model.status.value}")

    X, y, kinds = build_evaluation_set(seed=seed)
    flags, per_kind = [], {}
    for vec, label, kind in zip(X, y, kinds):
        res = model.predict(ExtractedFeatures(
            feature_vector=vec, feature_names=FEATURE_NAMES,
            feature_dict=dict(zip(FEATURE_NAMES, vec)),
        ))
        flagged = bool(res.is_anomaly)
        flags.append(flagged)
        bucket = per_kind.setdefault(kind, {"sessions": 0, "flagged": 0})
        bucket["sessions"] += 1
        bucket["flagged"] += int(flagged)

    tp = sum(1 for f, l in zip(flags, y) if f and l == 1)
    fp = sum(1 for f, l in zip(flags, y) if f and l == 0)
    tn = sum(1 for f, l in zip(flags, y) if not f and l == 0)
    fn = sum(1 for f, l in zip(flags, y) if not f and l == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "evaluation_name": "Simulated pipeline sessions (held-out)",
        "operating_threshold": model.bundle["metadata"].get("decision_threshold")
        or round(-float(model.bundle["pipeline"].named_steps["forest"].offset_), 4),
        "results": {
            "total_samples": len(y),
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            },
            "per_scenario": per_kind,
        },
        "real_world_accuracy_claim": False,
        "disclaimer": (
            "NOTICE: Evaluated on held-out SIMULATED sessions generated to mimic tracker.js "
            "and the request logger. This measures separability under these simulations, "
            "not real-world bot detection accuracy."
        ),
    }
