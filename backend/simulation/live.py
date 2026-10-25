"""
BotShield – Live demo scenarios (SIMULATED traffic, demo mode only).

Each run creates a fresh session labeled ``simulation:live-<scenario>`` and drives
it in real time through the Flask app (same middleware, request logger and
database as real visitors), re-scoring it with the hybrid pipeline as it goes.
Every score change is explained from the evaluation itself:

- FSM: which rules fired, their penalties and evidence
- ML: the model's anomaly score and the inputs furthest from its training data
- Hybrid: the formula with the actual numbers

Fresh sessions matter: ML features summarize a whole session, so a long-lived
browser session (localStorage keeps its ID for good) barely moves when one more
click happens. A new session shows each signal as it appears.
"""

from __future__ import annotations

import math
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from backend.database.db import get_connection, upsert_session, upsert_session_label
from backend.detection.fsm import FSMConfig, FSMState, anomaly_penalties
from backend.ml.features import FEATURE_NAMES
from backend.ml.model import load_model
from backend.scoring.service import HybridConfig, evaluate_session_risk

EVAL_MIN_INTERVAL = 0.5     # seconds between evaluations during a run
MAX_CONCURRENT_RUNS = 2
MAX_STORED_RUNS = 10

_RULE_TITLES = {
    "ANOMALY_RAPID_TRAVERSAL": "Rapid page traversal",
    "ANOMALY_HIGH_REQUEST_LOW_TELEMETRY": "Requests without browser activity",
    "ANOMALY_PATH_CYCLING": "Repeated same-page requests",
    "ANOMALY_RAPID_REQUEST_BURST": "Request burst",
    "ANOMALY_INVALID_ORDERING": "Impossible action order",
    "SIGNAL_LOW_INTERACTION": "No mouse/keyboard interaction (context only)",
}

_FEATURE_UNITS = {
    "session_duration_seconds": "s", "request_rate": "/s", "telemetry_rate": "/s",
    "mean_inter_event_time": "s", "std_inter_event_time": "s", "min_inter_event_time": "s",
}


def _now_iso_ms() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def fsm_breakdown(fsm_result) -> dict[str, Any]:
    """Per-rule contributions that add up to the FSM score."""
    cfg = FSMConfig()
    penalties = anomaly_penalties(cfg)
    codes = list(fsm_result.anomaly_codes)
    has_primary = any(c != "SIGNAL_LOW_INTERACTION" for c in codes)
    rules = []
    for code, reason in zip(codes, fsm_result.reasons):
        counted = has_primary or code != "SIGNAL_LOW_INTERACTION"
        rules.append({
            "code": code,
            "title": _RULE_TITLES.get(code, code),
            "penalty": penalties.get(code, 0.10) if counted else 0.0,
            "explanation": reason,
        })
    recovery = fsm_result.final_state == FSMState.RECOVERY
    return {
        "rules": rules,
        "raw_total": round(sum(r["penalty"] for r in rules), 4),
        "recovery_factor": cfg.recovery_mitigation_factor if recovery else None,
        "score": fsm_result.risk_score,
        "state": fsm_result.final_state.value,
    }


def ml_breakdown(ml_result, model_path: str | None = None, top: int = 3) -> dict[str, Any]:
    """
    ML score plus the inputs furthest from the training data.

    Distance is measured in the model's own preprocessing space (log1p if used,
    then standardized). This shows what is unusual about the session; it is not
    a causal attribution of the Isolation Forest's score.
    """
    out = {
        "status": ml_result.status.value,
        "score": ml_result.anomaly_score,
        "is_anomaly": ml_result.is_anomaly,
        "threshold": ml_result.model_metadata.get("decision_threshold"),
        "unusual_inputs": [],
    }
    if ml_result.anomaly_score is None:
        return out
    bundle, _, _ = load_model(model_path)
    if bundle is None:
        return out
    steps = bundle["pipeline"].named_steps
    scaler = steps.get("scaler")
    if scaler is None:
        return out
    use_log = "log1p" in steps
    deviations = []
    for i, name in enumerate(FEATURE_NAMES):
        value = float(ml_result.feature_vector[i])
        x = math.log1p(max(0.0, value)) if use_log else value
        scale = float(scaler.scale_[i]) or 1.0
        z = (x - float(scaler.mean_[i])) / scale
        typical = math.expm1(float(scaler.mean_[i])) if use_log else float(scaler.mean_[i])
        deviations.append((abs(z), name, value, typical, z))
    for _, name, value, typical, z in sorted(deviations, reverse=True)[:top]:
        unit = _FEATURE_UNITS.get(name, "")
        out["unusual_inputs"].append({
            "feature": name,
            "value": round(value, 3),
            "typical": round(typical, 3),
            "direction": "higher" if z > 0 else "lower",
            "z": round(z, 2),
            "text": f"{name.replace('_', ' ')} = {value:.3g}{unit} "
                    f"({'higher' if z > 0 else 'lower'} than typical ≈ {typical:.3g}{unit})",
        })
    return out


def explain_session(session_id: str, db_path: str | None = None, model_path: str | None = None,
                    persist: bool = False) -> dict[str, Any]:
    """Evaluate a session and return scores with their explanations."""
    result = evaluate_session_risk(session_id, db_path=db_path, model_path=model_path, persist=persist)
    cfg = HybridConfig()
    fsm = fsm_breakdown(result.fsm_result)
    ml = ml_breakdown(result.ml_result, model_path)
    if result.ml_score is not None:
        arithmetic = (f"{cfg.fsm_weight:.2f} × {result.fsm_score:.3f} + {cfg.ml_weight:.2f} × "
                      f"{result.ml_score:.3f} = {result.normalized_score:.3f}")
    else:
        arithmetic = f"ML unavailable ({result.model_status.value}) → hybrid = FSM = {result.normalized_score:.3f}"
    return {
        "session_id": session_id,
        "evaluated_at": result.evaluated_at,
        "fsm": fsm,
        "ml": ml,
        "hybrid": {
            "score": result.normalized_score,
            "risk_level": result.risk_level.value,
            "mode": result.scoring_mode.value,
            "arithmetic": arithmetic,
            "thresholds": {"medium": cfg.low_threshold / 100.0, "high": cfg.high_threshold / 100.0},
        },
        "activity": {
            "requests": int(result.fsm_result.evaluation_metadata.get("total_requests", 0)),
            "telemetry_events": int(result.fsm_result.evaluation_metadata.get("total_events", 0)),
        },
    }


def describe_changes(prev: dict | None, cur: dict) -> list[str]:
    """Plain-language notes for what changed between two explanations."""
    notes: list[str] = []
    if prev is None:
        notes.append(f"First evaluation: FSM {cur['fsm']['score']:.2f}, "
                     f"ML {_fmt(cur['ml']['score'])}, hybrid {cur['hybrid']['score']:.2f} "
                     f"({cur['hybrid']['risk_level']}).")
        for rule in cur["fsm"]["rules"]:
            notes.append(f"FSM rule active — {rule['title']} (+{rule['penalty']:.2f}): {rule['explanation']}")
        return notes

    before = {r["code"] for r in prev["fsm"]["rules"]}
    for rule in cur["fsm"]["rules"]:
        if rule["code"] not in before:
            notes.append(f"FSM rule fired — {rule['title']} (+{rule['penalty']:.2f}): {rule['explanation']}")
    for code in before - {r["code"] for r in cur["fsm"]["rules"]}:
        notes.append(f"FSM rule cleared — {_RULE_TITLES.get(code, code)}.")
    if prev["fsm"]["state"] != cur["fsm"]["state"]:
        notes.append(f"Behavioral state {prev['fsm']['state']} → {cur['fsm']['state']}.")

    p_ml, c_ml = prev["ml"]["score"], cur["ml"]["score"]
    if c_ml is not None and (p_ml is None or abs(c_ml - p_ml) >= 0.02):
        unusual = "; ".join(u["text"] for u in cur["ml"]["unusual_inputs"][:2])
        notes.append(f"ML anomaly score {_fmt(p_ml)} → {c_ml:.2f}. Most unusual inputs: {unusual}.")
    if prev["ml"]["is_anomaly"] != cur["ml"]["is_anomaly"] and cur["ml"]["is_anomaly"] is not None:
        side = "above" if cur["ml"]["is_anomaly"] else "below"
        notes.append(f"ML decision: now {side} the model's threshold ({_fmt(cur['ml']['threshold'])}).")

    if prev["hybrid"]["risk_level"] != cur["hybrid"]["risk_level"]:
        notes.append(f"Risk level {prev['hybrid']['risk_level']} → {cur['hybrid']['risk_level']}: "
                     f"{cur['hybrid']['arithmetic']}.")
    return notes


def _fmt(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.2f}"


# ---------------------------------------------------------------------------
# Scenario scripts
# ---------------------------------------------------------------------------

class _Driver:
    """Sends storefront traffic for one session through the Flask app."""

    def __init__(self, app, session_id: str, stop: threading.Event, on_activity: Callable[[], None]):
        self.client = app.test_client()
        self.client.set_cookie("botshield_session", session_id)
        self.client.set_cookie("botshield_session_id", session_id)
        self.session_id = session_id
        self.stop = stop
        self.on_activity = on_activity
        self.queue: list[dict] = []

    def wait(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self.stop.is_set() and time.monotonic() < end:
            time.sleep(min(0.1, max(0.0, end - time.monotonic())))
            self.on_activity()

    def get(self, path: str, method: str = "GET") -> None:
        if self.stop.is_set():
            return
        if method == "POST":
            self.client.post(path, data={"email": "demo@example.test", "password": "not-a-real-password"})
        else:
            self.client.get(path)
        self.on_activity()

    def event(self, event_type: str, page: str, **extra) -> None:
        self.queue.append({"event_type": event_type, "timestamp": _now_iso_ms(), "page": page, **extra})

    def flush(self) -> None:
        if self.queue and not self.stop.is_set():
            self.client.post("/api/events", json={"session_id": self.session_id, "events": self.queue})
            self.queue = []
            self.on_activity()

    def human_page(self, page: str, rng: random.Random, dwell: float) -> None:
        """Load a page like a browser running tracker.js, then read it."""
        self.get(page)
        self.wait(0.03)
        self.event("PAGE_VIEW", page)
        end = time.monotonic() + dwell
        while not self.stop.is_set() and time.monotonic() < end:
            self.wait(rng.uniform(0.7, 1.6))
            roll = rng.random()
            if roll < 0.45:
                self.event("SCROLL", page, scroll_depth=rng.choice([10, 25, 40, 60, 85]))
            elif roll < 0.75:
                self.event("MOUSE_ACTIVITY", page, activity_count=rng.randint(5, 40))
            else:
                self.event("CLICK", page, element_type="a", element_id="product-link")
            self.flush()
        self.event("CLICK", page, element_type="a", element_id="nav-link")
        self.event("PAGE_EXIT", page)
        self.flush()


def _human(d: _Driver, stage: Callable[[str], None], rng: random.Random) -> None:
    stage("A shopper reads the home page, scrolling and moving the mouse.")
    d.human_page("/", rng, 6)
    stage("They open the catalogue and look around.")
    d.human_page("/products", rng, 6)
    stage("They read a product page.")
    d.human_page("/product/2", rng, 7)
    stage("They check the About page, then stop.")
    d.human_page("/about", rng, 5)


def _scraper(d: _Driver, stage: Callable[[str], None], rng: random.Random) -> None:
    stage("A script requests catalogue pages back-to-back, with no browser and no JavaScript.")
    pages = ["/", "/products"] + [f"/product/{i}" for i in range(1, 7)]
    for _ in range(3):
        for p in pages:
            d.get(p)
            d.wait(rng.uniform(0.02, 0.06))
    stage("It pauses, then sweeps the catalogue again.")
    d.wait(3)
    for p in pages * 2:
        d.get(p)
        d.wait(rng.uniform(0.02, 0.06))
    d.wait(2)


def _credential(d: _Driver, stage: Callable[[str], None], rng: random.Random) -> None:
    stage("A script loads the login page directly, without browsing first.")
    d.get("/login")
    d.wait(1.0)
    stage("It submits login attempts in quick succession (credential stuffing).")
    for _ in range(18):
        d.get("/login", method="POST")
        d.wait(rng.uniform(0.05, 0.15))
    d.wait(2)


def _stealth(d: _Driver, stage: Callable[[str], None], rng: random.Random) -> None:
    stage("A slow crawler fetches one page every 1–2 s to stay under rate limits.")
    pages = ["/", "/products", "/product/1", "/product/2", "/product/3", "/about"]
    for i in range(14):
        if i == 7:
            stage("It keeps going: steady pacing, but still no browser activity at all.")
        d.get(pages[i % len(pages)])
        d.wait(rng.uniform(1.0, 2.0))


def _headless(d: _Driver, stage: Callable[[str], None], rng: random.Random) -> None:
    stage("A headless browser runs the page's JavaScript but never scrolls or moves the mouse.")
    pages = ["/", "/products", "/product/1", "/product/2", "/product/3", "/product/4", "/about"]
    for i in range(16):
        if i == 8:
            stage("It keeps clicking through pages every ~0.5 s.")
        p = pages[i % len(pages)]
        d.get(p)
        d.wait(0.03)
        d.event("PAGE_VIEW", p)
        d.flush()
        d.wait(rng.uniform(0.35, 0.7))
        d.event("PAGE_EXIT", p)
        d.flush()


def _takeover(d: _Driver, stage: Callable[[str], None], rng: random.Random) -> None:
    stage("Phase 1 — normal browsing: a visitor reads two pages.")
    d.human_page("/", rng, 6)
    d.human_page("/products", rng, 6)
    stage("Phase 2 — automation takes over: 25 product pages requested in about 1.5 s.")
    for i in range(25):
        d.get(f"/product/{i % 6 + 1}")
        d.wait(0.05)
    stage("Phase 3 — the burst stops; the evidence stays in the session.")
    d.wait(4)


SCENARIOS: dict[str, dict[str, Any]] = {
    "human": {"title": "Human shopper", "is_bot": False, "run": _human,
              "summary": "Reads four pages with scrolling, mouse movement and clicks.",
              "expect": "FSM stays 0; ML stays below its threshold; hybrid LOW."},
    "takeover": {"title": "Human, then automated burst", "is_bot": True, "run": _takeover,
                 "summary": "Normal browsing, then 25 page requests in ~1.5 s from the same session.",
                 "expect": "Low at first; FSM burst and traversal rules fire when automation starts."},
    "scraper": {"title": "Catalogue scraper", "is_bot": True, "run": _scraper,
                "summary": "Back-to-back page requests with no browser telemetry.",
                "expect": "FSM rapid-traversal, burst and no-telemetry rules; ML anomalous."},
    "credential": {"title": "Credential stuffing", "is_bot": True, "run": _credential,
                   "summary": "Repeated login submissions without browsing.",
                   "expect": "FSM repeated-path, burst and no-telemetry rules."},
    "stealth": {"title": "Low-and-slow crawler", "is_bot": True, "run": _stealth,
                "summary": "One page every 1–2 s, never any browser activity.",
                "expect": "Evades speed rules; caught by no-telemetry rule and ML."},
    "headless": {"title": "Headless browser", "is_bot": True, "run": _headless,
                 "summary": "Runs JavaScript (page views) but no scrolling or mouse movement.",
                 "expect": "Little or no FSM evidence; ML flags the missing interaction."},
}


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

@dataclass
class LiveRun:
    run_id: str
    scenario: str
    session_id: str
    started_at: float = field(default_factory=time.time)
    status: str = "running"         # running | finished | stopped | error
    stage: str = ""
    timeline: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    latest: dict | None = None
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    _last_eval: float = 0.0

    def to_dict(self, since: int = 0) -> dict[str, Any]:
        meta = SCENARIOS[self.scenario]
        return {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "title": meta["title"],
            "expect": meta["expect"],
            "session_id": self.session_id,
            "label": f"simulation:live-{self.scenario}",
            "status": self.status,
            "stage": self.stage,
            "elapsed": round(time.time() - self.started_at, 1),
            "timeline": self.timeline,
            "log": self.log[since:],
            "log_total": len(self.log),
            "latest": self.latest,
            "error": self.error,
        }


_RUNS: dict[str, LiveRun] = {}
_LOCK = threading.Lock()


def list_scenarios() -> list[dict[str, Any]]:
    return [{"id": k, "title": v["title"], "summary": v["summary"], "expect": v["expect"],
             "is_bot": v["is_bot"]} for k, v in SCENARIOS.items()]


def get_run(run_id: str) -> LiveRun | None:
    return _RUNS.get(run_id)


def stop_run(run_id: str) -> bool:
    run = _RUNS.get(run_id)
    if run is None:
        return False
    run.stop_event.set()
    return True


def start_run(app, scenario: str, seed: int | None = None) -> LiveRun:
    """Start a scenario on a background thread. Raises ValueError/RuntimeError."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    with _LOCK:
        active = [r for r in _RUNS.values() if r.status == "running"]
        if len(active) >= MAX_CONCURRENT_RUNS:
            raise RuntimeError("Too many scenarios running; stop one first.")
        run = LiveRun(run_id=uuid.uuid4().hex[:12], scenario=scenario, session_id=str(uuid.uuid4()))
        _RUNS[run.run_id] = run
        for old in sorted(_RUNS.values(), key=lambda r: r.started_at)[:-MAX_STORED_RUNS]:
            if old.status != "running":
                _RUNS.pop(old.run_id, None)

    db_path = app.config.get("BOTSHIELD_DB")
    model_path = app.config.get("BOTSHIELD_MODEL_PATH")
    conn = get_connection(db_path)
    try:
        upsert_session(conn, run.session_id)
        upsert_session_label(conn, run.session_id, is_bot=SCENARIOS[scenario]["is_bot"],
                             source=f"simulation:live-{scenario}")
        conn.commit()
    finally:
        conn.close()

    def evaluate(force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - run._last_eval < EVAL_MIN_INTERVAL:
            return
        run._last_eval = now
        cur = explain_session(run.session_id, db_path=db_path, model_path=model_path, persist=True)
        t = round(time.time() - run.started_at, 2)
        run.timeline.append({"t": t, "fsm": cur["fsm"]["score"], "ml": cur["ml"]["score"],
                             "hybrid": cur["hybrid"]["score"], "risk": cur["hybrid"]["risk_level"]})
        for note in describe_changes(run.latest, cur):
            run.log.append({"t": t, "kind": "score", "text": note})
        run.latest = cur

    def set_stage(text: str) -> None:
        run.stage = text
        run.log.append({"t": round(time.time() - run.started_at, 2), "kind": "stage", "text": text})

    def worker() -> None:
        rng = random.Random(seed if seed is not None else 2026)
        try:
            # No app_context() here: Flask would reuse it for every test-client
            # request, so `g` (and the request logger's once-per-request guard)
            # would persist and only the first request would be logged.
            driver = _Driver(app, run.session_id, run.stop_event, evaluate)
            SCENARIOS[scenario]["run"](driver, set_stage, rng)
            evaluate(force=True)
            run.status = "stopped" if run.stop_event.is_set() else "finished"
            run.stage = "Stopped." if run.status == "stopped" else "Scenario finished."
        except Exception as exc:  # report, never crash the server
            run.status = "error"
            run.error = "Scenario failed; see server log."
            app.logger.error("[LIVE_DEMO] run %s failed: %s", run.run_id, exc)

    threading.Thread(target=worker, name=f"BotShield-LiveDemo-{run.run_id}", daemon=True).start()
    return run
