"""BotShield – Repeatable demo scenario (SIMULATED traffic).

Sends a fixed, labeled mix of simulated human and bot sessions to a running
BotShield server, scores each one with the same hybrid FSM + ML pipeline the
server uses, stores the result so the admin dashboard shows it immediately, and
prints FSM, ML and hybrid scores with their reasons.

Every session is recorded in session_labels as ``simulation:<profile>`` and is
shown as SIMULATED on the dashboard, separate from real visitor traffic.

Usage (with the demo server running on 127.0.0.1:5000):
    python -m backend.simulation.demo
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request

from backend.scoring.service import HybridConfig, evaluate_session_risk
from backend.simulation.profiles import ProfileType
from backend.simulation.runner import SimulationConfig, SimulationRunner

# Fixed scenario: deterministic profiles, two sessions each
DEMO_PROFILES = [
    ProfileType.HUMAN,
    ProfileType.RAPID_SCRAPER,
    ProfileType.CREDENTIAL_STUFFER,
    ProfileType.STEALTH_CRAWLER,
]

DISCLAIMER = (
    "All sessions above are SIMULATED by the BotShield traffic generator.\n"
    "Scores describe automation-like behavior. They are not proof that anyone is a bot,\n"
    "a fraudster or a scammer, and this run does not measure real-world accuracy."
)


def _server_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/about", timeout=5.0):
            return True
    except (urllib.error.URLError, OSError):
        return False


def _fmt(value: float | None) -> str:
    return "N/A " if value is None else f"{value:.2f}"


def run_demo(
    url: str = "http://127.0.0.1:5000",
    sessions_per_profile: int = 2,
    dwell_scale: float = 1.0,
    db_path: str | None = None,
    model_path: str | None = None,
    mode: str = "http",
) -> list[dict]:
    """Run the scenario and return one row per simulated session."""
    summary = SimulationRunner(SimulationConfig(
        mode=mode,
        target_url=url,
        profiles=DEMO_PROFILES,
        sessions_per_profile=sessions_per_profile,
        evaluate_detection=False,
        db_path=db_path,
        model_path=model_path,
        dwell_scale=dwell_scale,
    )).run()

    rows = []
    for sess in summary.sessions:
        # persist=True: the dashboard shows the same result without waiting for the worker
        result = evaluate_session_risk(
            sess.session_id, db_path=db_path, model_path=model_path, persist=True
        )
        rows.append({
            "profile": sess.profile_type,
            "label": "human" if sess.profile_type == ProfileType.HUMAN.value else "bot",
            "session_id": sess.session_id,
            "fsm": result.fsm_score,
            "ml": result.ml_score,
            "ml_status": result.model_status.value,
            "hybrid": result.normalized_score,
            "risk_level": result.risk_level.value,
            "mode": result.scoring_mode.value,
            "reasons": result.reasons,
            "error": sess.error,
        })
    return rows


def _print_report(rows: list[dict]) -> None:
    cfg = HybridConfig()
    print("=" * 96)
    print("BOTSHIELD DEMO SCENARIO — SIMULATED TRAFFIC")
    print(f"Hybrid = {cfg.fsm_weight:.2f} x FSM + {cfg.ml_weight:.2f} x ML when ML is available; FSM only otherwise.")
    print("=" * 96)
    print(f"{'Profile':<11} {'Truth':<6} {'Session':<14} {'FSM':>5} {'ML':>5} {'Hybrid':>7}  {'Risk':<7} {'Mode':<9}")
    print("-" * 96)
    for r in rows:
        print(f"{r['profile']:<11} {r['label']:<6} {r['session_id'][:12] + '…':<14} "
              f"{_fmt(r['fsm']):>5} {_fmt(r['ml']):>5} {_fmt(r['hybrid']):>7}  {r['risk_level']:<7} {r['mode']:<9}")
        for reason in r["reasons"][:3]:
            print(f"{'':<14}  - {reason[:78]}")
        if r["error"]:
            print(f"{'':<14}  ! simulation error: {r['error']}")
    print("-" * 96)
    if rows and rows[0]["ml_status"] != "available":
        print(f"ML model status: {rows[0]['ml_status']} — scores are FSM-only.")
    print(DISCLAIMER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BotShield SIMULATED demo scenario.")
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="Base URL of the running demo server")
    parser.add_argument("--sessions", type=int, default=2, help="Simulated sessions per profile")
    parser.add_argument("--dwell-scale", type=float, default=1.0, help="Speed of the simulated human (lower = faster)")
    parser.add_argument("--db", default=None, help="Database the server uses (default: BOTSHIELD_DB or botshield.db)")
    parser.add_argument("--model", default=None, help="Model path (default: BOTSHIELD_MODEL_PATH or models/)")
    args = parser.parse_args()

    if not _server_reachable(args.url):
        print(f"Cannot reach the BotShield server at {args.url}. Start it first.", file=sys.stderr)
        sys.exit(1)

    rows = run_demo(args.url, args.sessions, args.dwell_scale, args.db, args.model)
    _print_report(rows)


if __name__ == "__main__":
    main()
