"""Red-Team Bot Traffic Simulator and Benchmark Runner.

Executes behavioral traffic profiles against BotShield (in-process Flask
test client or live HTTP server), evaluates detection outcomes, and measures
latency benchmarks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
import uuid

from backend.database.db import get_connection, upsert_session
from backend.scoring.service import evaluate_session_risk
from backend.simulation.profiles import (
    BaseBehavioralProfile,
    ProfileType,
    SimulationStep,
    get_profile,
)


@dataclass
class SessionSimulationResult:
    """Outcome of a single simulated session."""
    session_id: str
    profile_type: str
    request_count: int = 0
    telemetry_count: int = 0
    http_statuses: List[int] = field(default_factory=list)
    request_latencies_ms: List[float] = field(default_factory=list)
    telemetry_latencies_ms: List[float] = field(default_factory=list)
    total_execution_ms: float = 0.0
    
    # Detection & Evaluation (optional if evaluate=True)
    evaluated: bool = False
    risk_level: Optional[str] = None
    risk_score: Optional[float] = None
    fsm_score: Optional[float] = None
    ml_score: Optional[float] = None
    scoring_mode: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    scoring_duration_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ProfileBenchmarkResult:
    """Aggregated benchmark statistics for a single behavioral profile."""
    profile_type: str
    total_sessions: int = 0
    successful_sessions: int = 0
    failed_sessions: int = 0
    risk_distribution: Dict[str, int] = field(default_factory=lambda: {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0})
    scoring_modes: Dict[str, int] = field(default_factory=dict)
    mean_request_latency_ms: float = 0.0
    p95_request_latency_ms: float = 0.0
    mean_scoring_latency_ms: float = 0.0
    expected_detection_rate: float = 0.0  # % matching expected risk (e.g. LOW for human, HIGH for scraper)


@dataclass
class SimulationBenchmarkSummary:
    """Complete summary of a red-team simulation benchmark run."""
    mode: str
    target_url: Optional[str]
    total_sessions: int
    profiles: Dict[str, ProfileBenchmarkResult] = field(default_factory=dict)
    total_duration_seconds: float = 0.0
    overall_detection_rate: float = 0.0
    sessions: List[SessionSimulationResult] = field(default_factory=list)


@dataclass
class SimulationConfig:
    """Configuration options for a simulation run."""
    mode: str = "in-process"  # "in-process" | "http"
    target_url: str = "http://127.0.0.1:5000"
    profiles: List[ProfileType] = field(default_factory=lambda: list(ProfileType))
    sessions_per_profile: int = 3
    evaluate_detection: bool = True
    db_path: Optional[Path | str] = None
    model_path: Optional[Path | str] = None
    dwell_scale: float = 1.0


class SimulationRunner:
    """Engine executing red-team profiles and gathering metrics."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._app = None
        self._client = None

        if self.config.mode == "in-process":
            from app import app
            self._app = app
            self._client = app.test_client()

    def run(self) -> SimulationBenchmarkSummary:
        start_time = time.time()
        all_results: List[SessionSimulationResult] = []
        profile_benchmarks: Dict[str, ProfileBenchmarkResult] = {}

        for profile_type in self.config.profiles:
            profile = get_profile(profile_type, dwell_scale=self.config.dwell_scale)
            session_results: List[SessionSimulationResult] = []

            for _ in range(self.config.sessions_per_profile):
                res = self._execute_session(profile)
                session_results.append(res)
                all_results.append(res)

            # Aggregate stats for this profile
            profile_benchmarks[profile_type.value] = self._aggregate_profile(profile_type, session_results)

        total_duration = time.time() - start_time

        # Calculate overall detection accuracy across all evaluated profiles
        evaluated_sessions = [s for s in all_results if s.evaluated and s.risk_level]
        matching_count = 0
        for s in evaluated_sessions:
            if s.profile_type == ProfileType.HUMAN.value and s.risk_level == "LOW":
                matching_count += 1
            elif s.profile_type in (ProfileType.RAPID_SCRAPER.value, ProfileType.CREDENTIAL_STUFFER.value) and s.risk_level == "HIGH":
                matching_count += 1
            elif s.profile_type == ProfileType.STEALTH_CRAWLER.value and s.risk_level in ("MEDIUM", "HIGH"):
                matching_count += 1

        overall_acc = (matching_count / len(evaluated_sessions) * 100.0) if evaluated_sessions else 0.0

        return SimulationBenchmarkSummary(
            mode=self.config.mode,
            target_url=self.config.target_url if self.config.mode == "http" else None,
            total_sessions=len(all_results),
            profiles=profile_benchmarks,
            total_duration_seconds=round(total_duration, 3),
            overall_detection_rate=round(overall_acc, 2),
            sessions=all_results,
        )

    def _execute_session(self, profile: BaseBehavioralProfile) -> SessionSimulationResult:
        session_id = uuid.uuid4().hex  # 32-char hex UUID conforms to session ID regex
        plan = profile.generate_plan(session_id)
        
        result = SessionSimulationResult(
            session_id=session_id,
            profile_type=profile.profile_type.value,
        )

        # Register session in database so request_logger recognizes it as verified
        conn = get_connection(self.config.db_path)
        try:
            upsert_session(conn, session_id)
            conn.commit()
        finally:
            conn.close()

        sess_start = time.time()

        try:
            for step in plan:
                if step.dwell_seconds > 0:
                    time.sleep(step.dwell_seconds)

                if step.step_type == "request":
                    t0 = time.time()
                    status = self._send_request(session_id, step.method, step.path, step.data)
                    lat = (time.time() - t0) * 1000.0
                    result.http_statuses.append(status)
                    result.request_latencies_ms.append(round(lat, 2))
                    result.request_count += 1

                elif step.step_type == "telemetry":
                    t0 = time.time()
                    status = self._send_telemetry(session_id, step.events)
                    lat = (time.time() - t0) * 1000.0
                    result.telemetry_latencies_ms.append(round(lat, 2))
                    result.telemetry_count += len(step.events)

        except Exception as exc:
            result.error = str(exc)

        result.total_execution_ms = round((time.time() - sess_start) * 1000.0, 2)

        # Post-session risk evaluation
        if self.config.evaluate_detection and not result.error:
            eval_t0 = time.time()
            try:
                risk_res = evaluate_session_risk(
                    session_id=session_id,
                    db_path=self.config.db_path,
                    model_path=self.config.model_path,
                )
                eval_duration = (time.time() - eval_t0) * 1000.0
                result.evaluated = True
                result.risk_level = risk_res.risk_level.value
                result.risk_score = risk_res.composite_score
                result.fsm_score = risk_res.fsm_score
                result.ml_score = risk_res.ml_score
                result.scoring_mode = risk_res.scoring_mode.value
                result.reasons = risk_res.reasons
                result.scoring_duration_ms = round(eval_duration, 2)
            except Exception as exc:
                result.error = f"Evaluation error: {exc}"

        return result

    def _send_request(self, session_id: str, method: str, path: str, data: Optional[Dict[str, Any]]) -> int:
        if self.config.mode == "in-process":
            assert self._client is not None
            self._client.set_cookie("botshield_session", session_id)
            if method.upper() == "POST":
                resp = self._client.post(path, data=data or {})
            else:
                resp = self._client.get(path)
            return resp.status_code
        else:
            url = f"{self.config.target_url.rstrip('/')}{path}"
            req = urllib.request.Request(url, method=method.upper())
            req.add_header("Cookie", f"botshield_session={session_id}")
            req.add_header("User-Agent", "BotShield-RedTeam-Simulator/1.0")

            if data is not None and method.upper() == "POST":
                encoded = json.dumps(data).encode("utf-8")
                req.data = encoded
                req.add_header("Content-Type", "application/json")

            try:
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    return resp.status
            except urllib.error.HTTPError as exc:
                return exc.code

    def _send_telemetry(self, session_id: str, events: List[Dict[str, Any]]) -> int:
        payload = {"session_id": session_id, "events": events}
        if self.config.mode == "in-process":
            assert self._client is not None
            self._client.set_cookie("botshield_session", session_id)
            resp = self._client.post("/api/events", json=payload)
            return resp.status_code
        else:
            url = f"{self.config.target_url.rstrip('/')}/api/events"
            req = urllib.request.Request(url, method="POST")
            req.add_header("Cookie", f"botshield_session={session_id}")
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "BotShield-RedTeam-Simulator/1.0")
            req.data = json.dumps(payload).encode("utf-8")

            try:
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    return resp.status
            except urllib.error.HTTPError as exc:
                return exc.code

    def _aggregate_profile(self, profile_type: ProfileType, sessions: List[SessionSimulationResult]) -> ProfileBenchmarkResult:
        total = len(sessions)
        successes = sum(1 for s in sessions if not s.error)
        fails = total - successes

        dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "UNKNOWN": 0}
        scoring_modes: Dict[str, int] = {}
        all_req_latencies: List[float] = []
        scoring_latencies: List[float] = []

        matching_evals = 0
        evaluated_count = 0

        for s in sessions:
            all_req_latencies.extend(s.request_latencies_ms)
            if s.evaluated and s.risk_level:
                evaluated_count += 1
                dist[s.risk_level] = dist.get(s.risk_level, 0) + 1
                if s.scoring_mode:
                    scoring_modes[s.scoring_mode] = scoring_modes.get(s.scoring_mode, 0) + 1
                scoring_latencies.append(s.scoring_duration_ms)

                # Match expectations
                if profile_type == ProfileType.HUMAN and s.risk_level == "LOW":
                    matching_evals += 1
                elif profile_type in (ProfileType.RAPID_SCRAPER, ProfileType.CREDENTIAL_STUFFER) and s.risk_level == "HIGH":
                    matching_evals += 1
                elif profile_type == ProfileType.STEALTH_CRAWLER and s.risk_level in ("MEDIUM", "HIGH"):
                    matching_evals += 1
            else:
                dist["UNKNOWN"] += 1

        mean_lat = statistics.mean(all_req_latencies) if all_req_latencies else 0.0
        p95_lat = statistics.quantiles(all_req_latencies, n=20)[-1] if len(all_req_latencies) >= 20 else (max(all_req_latencies) if all_req_latencies else 0.0)
        mean_score_lat = statistics.mean(scoring_latencies) if scoring_latencies else 0.0
        det_rate = (matching_evals / evaluated_count * 100.0) if evaluated_count else 0.0

        return ProfileBenchmarkResult(
            profile_type=profile_type.value,
            total_sessions=total,
            successful_sessions=successes,
            failed_sessions=fails,
            risk_distribution=dist,
            scoring_modes=scoring_modes,
            mean_request_latency_ms=round(mean_lat, 2),
            p95_request_latency_ms=round(p95_lat, 2),
            mean_scoring_latency_ms=round(mean_score_lat, 2),
            expected_detection_rate=round(det_rate, 2),
        )


def run_simulation(
    mode: str = "in-process",
    target_url: str = "http://127.0.0.1:5000",
    profiles: Optional[List[ProfileType]] = None,
    sessions_per_profile: int = 3,
    evaluate_detection: bool = True,
    db_path: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
    dwell_scale: float = 1.0,
) -> SimulationBenchmarkSummary:
    """Convenience entry point for running a traffic simulation benchmark."""
    config = SimulationConfig(
        mode=mode,
        target_url=target_url,
        profiles=profiles or list(ProfileType),
        sessions_per_profile=sessions_per_profile,
        evaluate_detection=evaluate_detection,
        db_path=db_path,
        model_path=model_path,
        dwell_scale=dwell_scale,
    )
    runner = SimulationRunner(config)
    return runner.run()


def _format_table(summary: SimulationBenchmarkSummary) -> str:
    """Render a human-readable ANSI/text summary table."""
    lines: List[str] = [
        "=" * 78,
        "              BOTSHIELD RED-TEAM SIMULATION & DETECTION BENCHMARK",
        "=" * 78,
        f"Mode: {summary.mode.upper()}  |  Target: {summary.target_url or 'Flask Test Client'}  |  Duration: {summary.total_duration_seconds}s",
        f"Total Sessions: {summary.total_sessions}  |  Overall Detection Accuracy: {summary.overall_detection_rate}%",
        "-" * 78,
        f"{'Profile':<20} | {'Sessions':<8} | {'LOW':<5} | {'MED':<5} | {'HIGH':<5} | {'Req Lat':<10} | {'Match Rate'}",
        "-" * 78,
    ]

    for p_name, bm in summary.profiles.items():
        dist = bm.risk_distribution
        lines.append(
            f"{p_name:<20} | {bm.total_sessions:<8} | {dist.get('LOW', 0):<5} | "
            f"{dist.get('MEDIUM', 0):<5} | {dist.get('HIGH', 0):<5} | "
            f"{bm.mean_request_latency_ms:>6.2f} ms | {bm.expected_detection_rate:>6.1f}%"
        )

    lines.append("=" * 78)
    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for running traffic simulations."""
    parser = argparse.ArgumentParser(
        description="BotShield Red-Team Traffic Simulator & Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["in-process", "http"],
        default="in-process",
        help="Execution mode (in-process Flask test client vs live HTTP server)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:5000",
        help="Base URL of live BotShield server (used when --mode=http)",
    )
    parser.add_argument(
        "--profiles",
        default="all",
        help="Comma-separated profiles: all, human, scraper, credential, stealth",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=3,
        help="Number of simulated sessions per profile",
    )
    parser.add_argument(
        "--dwell-scale",
        type=float,
        default=1.0,
        help="Multiplier for human dwell pauses (lower for faster simulation in tests)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip post-session hybrid risk scoring evaluation",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw benchmark summary in JSON format",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional custom SQLite database path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional custom Isolation Forest model artifact path",
    )

    args = parser.parse_args()

    # Parse profiles
    if args.profiles.lower() == "all":
        profiles = list(ProfileType)
    else:
        profile_keys = [p.strip().lower() for p in args.profiles.split(",") if p.strip()]
        profiles = []
        for k in profile_keys:
            try:
                profiles.append(ProfileType(k))
            except ValueError:
                print(f"Error: Unknown profile '{k}'. Valid: {[p.value for p in ProfileType]}", file=sys.stderr)
                sys.exit(2)

    summary = run_simulation(
        mode=args.mode,
        target_url=args.url,
        profiles=profiles,
        sessions_per_profile=args.sessions,
        evaluate_detection=not args.no_evaluate,
        db_path=args.db,
        model_path=args.model,
        dwell_scale=args.dwell_scale,
    )

    if args.json:
        # Convert dataclass to JSON
        print(json.dumps(asdict(summary), indent=2))
    else:
        print(_format_table(summary))


if __name__ == "__main__":
    main()
