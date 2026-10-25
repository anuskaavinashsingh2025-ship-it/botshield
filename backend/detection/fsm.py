"""
BotShield – Finite-State Machine (FSM) Behavioral Detection Engine (Phase 3)

Implements a deterministic finite-state transducer (DFST) that evaluates a session's
normalized browser telemetry events and server-observed HTTP requests to detect
suspicious automation patterns and compute an explainable risk score.

Automaton Formalism:
- Q (States): {INITIAL, BROWSING, PRODUCT_VIEW, CART_ACTION, LOGIN_ATTEMPT, SUSPICIOUS, RECOVERY}
- Sigma (Alphabet): {NAV_HOME, NAV_CATALOG, NAV_PRODUCT, NAV_ABOUT, NAV_LOGIN,
                     ACTION_CART, ACTION_LOGIN, USER_INTERACTION, PAGE_EXIT,
                     HTTP_API, UNKNOWN}
- delta (Transition function): Q x Sigma x Context -> Q x P(Anomalies)
- q0 (Initial state): INITIAL
- Output: Ordered transition history, anomaly codes, human-readable explanations,
          and a deterministic normalized risk score in [0.0, 1.0].
"""

from __future__ import annotations

import enum
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.database.db import normalize_path

logger = logging.getLogger("botshield.detection.fsm")


class FSMState(str, enum.Enum):
    """Finite set of operational and detection states (Q)."""
    INITIAL = "INITIAL"
    BROWSING = "BROWSING"
    PRODUCT_VIEW = "PRODUCT_VIEW"
    CART_ACTION = "CART_ACTION"
    LOGIN_ATTEMPT = "LOGIN_ATTEMPT"
    SUSPICIOUS = "SUSPICIOUS"
    RECOVERY = "RECOVERY"


class FSMSymbol(str, enum.Enum):
    """Normalized input event alphabet (Sigma)."""
    NAV_HOME = "NAV_HOME"
    NAV_CATALOG = "NAV_CATALOG"
    NAV_PRODUCT = "NAV_PRODUCT"
    NAV_ABOUT = "NAV_ABOUT"
    NAV_LOGIN = "NAV_LOGIN"
    ACTION_CART = "ACTION_CART"
    ACTION_LOGIN = "ACTION_LOGIN"
    USER_INTERACTION = "USER_INTERACTION"
    PAGE_EXIT = "PAGE_EXIT"
    HTTP_API = "HTTP_API"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FSMConfig:
    """Configurable thresholds and anomaly penalties with descriptive names."""
    # Rapid navigation threshold between distinct page loads (milliseconds)
    rapid_traversal_threshold_ms: float = 150.0

    # Path cycling thresholds
    max_repeated_path_count: int = 4
    repeat_window_seconds: float = 10.0

    # Disproportionate request-to-telemetry threshold
    min_requests_for_telemetry_check: int = 5
    min_telemetry_ratio: float = 0.15

    # Request burst rate threshold
    burst_rps_threshold: float = 6.0
    burst_window_seconds: float = 2.0

    # Human recovery criteria
    recovery_interaction_min_count: int = 3
    recovery_dwell_seconds: float = 2.0
    recovery_mitigation_factor: float = 0.40  # 60% penalty reduction upon sustained recovery

    # Contextual low-interaction signal threshold (context only, never solitary bot proof)
    low_interaction_context_threshold: int = 8

    # Anomaly penalties
    penalty_rapid_traversal: float = 0.35
    penalty_high_req_low_telemetry: float = 0.30
    penalty_path_cycling: float = 0.25
    penalty_burst_rate: float = 0.25
    penalty_invalid_ordering: float = 0.20
    penalty_low_interaction_context: float = 0.05


@dataclass
class FSMEvent:
    """A normalized event consuming telemetry or server-observed requests."""
    symbol: FSMSymbol
    epoch_timestamp: float
    iso_timestamp: str
    source: str  # 'HTTP_REQUEST' or 'TELEMETRY'
    path: str = ""
    source_id: int = 0
    raw_type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AnomalyRecord:
    """Detailed record of an emitted anomaly with non-sensitive evidence."""
    code: str
    explanation: str
    evidence: dict = field(default_factory=dict)


@dataclass
class TransitionStep:
    """A single deterministic state transition in the session execution trace."""
    from_state: FSMState
    to_state: FSMState
    symbol: FSMSymbol
    timestamp: str
    path: str
    anomalies: list[str] = field(default_factory=list)


@dataclass
class FSMResult:
    """Documented output contract returned by the behavioral FSM."""
    final_state: FSMState
    transition_history: list[TransitionStep]
    anomaly_codes: list[str]
    reasons: list[str]
    risk_score: float  # Normalized to [0.0, 1.0]
    evaluation_metadata: dict = field(default_factory=dict)

    @property
    def fsm_score(self) -> float:
        """Alias for risk_score to align with session_risks.fsm_score column."""
        return self.risk_score


# ---------------------------------------------------------------------------
# Path & Timestamp parsing helpers
# ---------------------------------------------------------------------------

_PRODUCT_PATH_RE = re.compile(r"^/product/\d+$")

# Max gap between a server request and the browser PAGE_VIEW for the same page
# load (HTML response + tracker.js start-up). Used to avoid counting one load twice.
_PAGE_VIEW_MIRROR_SECONDS = 5.0


def parse_iso_to_epoch(ts: str | None) -> float | None:
    """
    Safely convert ISO 8601 string to UTC epoch float.
    Returns None if missing, malformed, or invalid.
    """
    if not ts or not isinstance(ts, str):
        return None
    cleaned = ts.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def map_path_to_symbol(path: str, method: str = "GET") -> FSMSymbol:
    """Map a normalized URL path to an alphabet symbol."""
    if path == "/" or path == "":
        return FSMSymbol.NAV_HOME
    elif path == "/products":
        return FSMSymbol.NAV_CATALOG
    elif _PRODUCT_PATH_RE.match(path):
        return FSMSymbol.NAV_PRODUCT
    elif path == "/about":
        return FSMSymbol.NAV_ABOUT
    elif path == "/login":
        if method.upper() == "POST":
            return FSMSymbol.ACTION_LOGIN
        return FSMSymbol.NAV_LOGIN
    elif path == "/api/events":
        return FSMSymbol.HTTP_API
    return FSMSymbol.UNKNOWN


def map_telemetry_to_symbol(event_type: str, page: str, metadata: dict) -> FSMSymbol:
    """Map a browser telemetry event into an alphabet symbol."""
    event_upper = event_type.strip().upper() if isinstance(event_type, str) else ""

    if event_upper == "PAGE_VIEW":
        return map_path_to_symbol(page)
    elif event_upper == "PAGE_EXIT":
        return FSMSymbol.PAGE_EXIT
    elif event_upper in ("SCROLL", "MOUSE_ACTIVITY", "KEYBOARD_ACTIVITY"):
        return FSMSymbol.USER_INTERACTION
    elif event_upper == "CLICK":
        el_id = str(metadata.get("element_id", "")).lower()
        if "add-to-cart" in el_id or el_id == "cart-btn":
            return FSMSymbol.ACTION_CART
        elif "login" in el_id:
            return FSMSymbol.ACTION_LOGIN
        return FSMSymbol.USER_INTERACTION

    return FSMSymbol.UNKNOWN


def anomaly_penalties(cfg: FSMConfig) -> dict[str, float]:
    """Score contribution of each anomaly code (unknown codes contribute 0.10)."""
    return {
        "ANOMALY_RAPID_TRAVERSAL": cfg.penalty_rapid_traversal,
        "ANOMALY_HIGH_REQUEST_LOW_TELEMETRY": cfg.penalty_high_req_low_telemetry,
        "ANOMALY_PATH_CYCLING": cfg.penalty_path_cycling,
        "ANOMALY_RAPID_REQUEST_BURST": cfg.penalty_burst_rate,
        "ANOMALY_INVALID_ORDERING": cfg.penalty_invalid_ordering,
        "SIGNAL_LOW_INTERACTION": cfg.penalty_low_interaction_context,
    }


# ---------------------------------------------------------------------------
# Core Behavioral FSM Transducer
# ---------------------------------------------------------------------------

class BehavioralFSM:
    """
    Deterministic Finite-State Transducer (DFST) for user session analysis.
    """

    def __init__(self, config: FSMConfig | None = None) -> None:
        self.config = config or FSMConfig()

    def _is_navigation_symbol(self, symbol: FSMSymbol) -> bool:
        return symbol in (
            FSMSymbol.NAV_HOME,
            FSMSymbol.NAV_CATALOG,
            FSMSymbol.NAV_PRODUCT,
            FSMSymbol.NAV_ABOUT,
            FSMSymbol.NAV_LOGIN,
        )

    def _next_base_state(self, current: FSMState, symbol: FSMSymbol) -> tuple[FSMState, list[AnomalyRecord]]:
        """
        Deterministic transition table without anomaly overrides.
        Returns: (next_state, ordering_anomalies)
        """
        anomalies: list[AnomalyRecord] = []

        if current == FSMState.INITIAL:
            if symbol in (FSMSymbol.NAV_HOME, FSMSymbol.NAV_CATALOG, FSMSymbol.NAV_ABOUT):
                return FSMState.BROWSING, anomalies
            elif symbol == FSMSymbol.NAV_PRODUCT:
                return FSMState.PRODUCT_VIEW, anomalies
            elif symbol == FSMSymbol.NAV_LOGIN:
                return FSMState.LOGIN_ATTEMPT, anomalies
            elif symbol == FSMSymbol.ACTION_CART:
                anomalies.append(
                    AnomalyRecord(
                        code="ANOMALY_INVALID_ORDERING",
                        explanation="Inconsistent event sequence: cart action triggered in INITIAL state without prior browsing.",
                        evidence={"from_state": current.value, "symbol": symbol.value},
                    )
                )
                return FSMState.SUSPICIOUS, anomalies
            elif symbol == FSMSymbol.USER_INTERACTION:
                return FSMState.BROWSING, anomalies
            return FSMState.INITIAL, anomalies

        elif current == FSMState.BROWSING:
            if symbol == FSMSymbol.NAV_PRODUCT:
                return FSMState.PRODUCT_VIEW, anomalies
            elif symbol in (FSMSymbol.NAV_HOME, FSMSymbol.NAV_CATALOG, FSMSymbol.NAV_ABOUT):
                return FSMState.BROWSING, anomalies
            elif symbol == FSMSymbol.NAV_LOGIN:
                return FSMState.LOGIN_ATTEMPT, anomalies
            elif symbol == FSMSymbol.ACTION_CART:
                return FSMState.CART_ACTION, anomalies
            return FSMState.BROWSING, anomalies

        elif current == FSMState.PRODUCT_VIEW:
            if symbol == FSMSymbol.ACTION_CART:
                return FSMState.CART_ACTION, anomalies
            elif symbol == FSMSymbol.NAV_PRODUCT:
                return FSMState.PRODUCT_VIEW, anomalies
            elif symbol in (FSMSymbol.NAV_HOME, FSMSymbol.NAV_CATALOG, FSMSymbol.NAV_ABOUT):
                return FSMState.BROWSING, anomalies
            elif symbol == FSMSymbol.NAV_LOGIN:
                return FSMState.LOGIN_ATTEMPT, anomalies
            return FSMState.PRODUCT_VIEW, anomalies

        elif current == FSMState.CART_ACTION:
            if symbol == FSMSymbol.NAV_PRODUCT:
                return FSMState.PRODUCT_VIEW, anomalies
            elif symbol in (FSMSymbol.NAV_HOME, FSMSymbol.NAV_CATALOG, FSMSymbol.NAV_ABOUT):
                return FSMState.BROWSING, anomalies
            elif symbol == FSMSymbol.NAV_LOGIN:
                return FSMState.LOGIN_ATTEMPT, anomalies
            return FSMState.CART_ACTION, anomalies

        elif current == FSMState.LOGIN_ATTEMPT:
            if symbol in (FSMSymbol.NAV_LOGIN, FSMSymbol.ACTION_LOGIN):
                return FSMState.LOGIN_ATTEMPT, anomalies
            elif symbol in (FSMSymbol.NAV_HOME, FSMSymbol.NAV_CATALOG, FSMSymbol.NAV_ABOUT):
                return FSMState.BROWSING, anomalies
            elif symbol == FSMSymbol.NAV_PRODUCT:
                return FSMState.PRODUCT_VIEW, anomalies
            return FSMState.LOGIN_ATTEMPT, anomalies

        elif current == FSMState.SUSPICIOUS:
            # Remains in SUSPICIOUS unless recovery rules trigger
            return FSMState.SUSPICIOUS, anomalies

        elif current == FSMState.RECOVERY:
            if symbol in (FSMSymbol.NAV_HOME, FSMSymbol.NAV_CATALOG, FSMSymbol.NAV_ABOUT):
                return FSMState.BROWSING, anomalies
            elif symbol == FSMSymbol.NAV_PRODUCT:
                return FSMState.PRODUCT_VIEW, anomalies
            elif symbol == FSMSymbol.NAV_LOGIN:
                return FSMState.LOGIN_ATTEMPT, anomalies
            elif symbol == FSMSymbol.ACTION_CART:
                return FSMState.CART_ACTION, anomalies
            return FSMState.RECOVERY, anomalies

        return current, anomalies

    def process_events(self, events: list[FSMEvent]) -> FSMResult:
        """
        Execute deterministic state machine processing on a sorted event list.
        """
        cfg = self.config
        state = FSMState.INITIAL
        history: list[TransitionStep] = []
        emitted_anomalies: list[AnomalyRecord] = []
        warnings: list[str] = []

        if not events:
            return FSMResult(
                final_state=FSMState.INITIAL,
                transition_history=[],
                anomaly_codes=[],
                reasons=[],
                risk_score=0.0,
                evaluation_metadata={
                    "total_events": 0,
                    "total_requests": 0,
                    "total_inputs": 0,
                    "telemetry_ratio": 0.0,
                    "duration_seconds": 0.0,
                    "data_quality_warnings": ["WARNING_EMPTY_SESSION"],
                },
            )

        # Counters & tracking state
        http_requests_count = sum(1 for e in events if e.source == "HTTP_REQUEST" and e.symbol != FSMSymbol.HTTP_API)
        telemetry_count = sum(1 for e in events if e.source == "TELEMETRY")
        interaction_count = sum(1 for e in events if e.symbol == FSMSymbol.USER_INTERACTION)

        last_nav_event: FSMEvent | None = None
        consecutive_same_path_count = 0
        consecutive_same_path_str = ""
        first_same_path_time = 0.0

        consecutive_human_interactions = 0
        last_interaction_time = 0.0

        # Process events chronologically
        for i, event in enumerate(events):
            step_anomalies: list[AnomalyRecord] = []
            prev_nav_event = last_nav_event

            # 1. Check Rapid Traversal between distinct navigation events
            if self._is_navigation_symbol(event.symbol):
                if last_nav_event is not None and event.path != last_nav_event.path:
                    delta_ms = (event.epoch_timestamp - last_nav_event.epoch_timestamp) * 1000.0
                    # Check physically impossible traversal threshold
                    if 0.0 <= delta_ms < cfg.rapid_traversal_threshold_ms:
                        step_anomalies.append(
                            AnomalyRecord(
                                code="ANOMALY_RAPID_TRAVERSAL",
                                explanation=(
                                    f"Excessively rapid page traversal observed ({delta_ms:.1f} ms "
                                    f"between '{last_nav_event.path}' and '{event.path}', "
                                    f"threshold: {cfg.rapid_traversal_threshold_ms:.0f} ms)."
                                ),
                                evidence={
                                    "time_delta_ms": round(delta_ms, 2),
                                    "from_path": last_nav_event.path,
                                    "to_path": event.path,
                                    "threshold_ms": cfg.rapid_traversal_threshold_ms,
                                },
                            )
                        )

                last_nav_event = event

            # 2. Check Path Cycling / Repetitive loops
            # A page load is observed twice: the server request and the browser's
            # PAGE_VIEW for the same path moments later. Count it once.
            is_page_view_mirror = (
                event.source == "TELEMETRY"
                and prev_nav_event is not None
                and prev_nav_event.source == "HTTP_REQUEST"
                and prev_nav_event.path == event.path
                and 0.0 <= event.epoch_timestamp - prev_nav_event.epoch_timestamp <= _PAGE_VIEW_MIRROR_SECONDS
            )
            if self._is_navigation_symbol(event.symbol) and not is_page_view_mirror:
                if event.path == consecutive_same_path_str:
                    consecutive_same_path_count += 1
                else:
                    consecutive_same_path_str = event.path
                    consecutive_same_path_count = 1
                    first_same_path_time = event.epoch_timestamp

                duration_cycle = event.epoch_timestamp - first_same_path_time
                if (
                    consecutive_same_path_count >= cfg.max_repeated_path_count
                    and duration_cycle <= cfg.repeat_window_seconds
                ):
                    step_anomalies.append(
                        AnomalyRecord(
                            code="ANOMALY_PATH_CYCLING",
                            explanation=(
                                f"Repetitive cyclic requests to path '{event.path}' "
                                f"({consecutive_same_path_count} consecutive requests in "
                                f"{duration_cycle:.1f}s, threshold: {cfg.max_repeated_path_count})."
                            ),
                            evidence={
                                "path": event.path,
                                "repetitions": consecutive_same_path_count,
                                "window_seconds": round(duration_cycle, 2),
                            },
                        )
                    )
            elif event.symbol == FSMSymbol.USER_INTERACTION:
                # User interaction breaks path cycling loop
                consecutive_same_path_count = 0

            # 3. Check Recovery qualification
            if event.symbol == FSMSymbol.USER_INTERACTION:
                if last_interaction_time > 0.0:
                    dwell = event.epoch_timestamp - last_interaction_time
                    if dwell >= cfg.recovery_dwell_seconds:
                        consecutive_human_interactions += 1
                    else:
                        consecutive_human_interactions = max(1, consecutive_human_interactions)
                else:
                    consecutive_human_interactions = 1
                last_interaction_time = event.epoch_timestamp
            elif self._is_navigation_symbol(event.symbol):
                # Reset consecutive interactions on page hop unless sustained
                pass

            # 4. Compute next FSM state
            base_next, ordering_anomalies = self._next_base_state(state, event.symbol)
            step_anomalies.extend(ordering_anomalies)

            # State transition resolution
            next_state = base_next
            if step_anomalies:
                next_state = FSMState.SUSPICIOUS
            elif state == FSMState.SUSPICIOUS:
                # Evaluate recovery condition
                if consecutive_human_interactions >= cfg.recovery_interaction_min_count:
                    next_state = FSMState.RECOVERY

            # Record step and update state
            codes = [a.code for a in step_anomalies]
            emitted_anomalies.extend(step_anomalies)

            history.append(
                TransitionStep(
                    from_state=state,
                    to_state=next_state,
                    symbol=event.symbol,
                    timestamp=event.iso_timestamp,
                    path=event.path,
                    anomalies=codes,
                )
            )
            state = next_state

        # -------------------------------------------------------------------
        # Session-Level Global Invariant Checks
        # -------------------------------------------------------------------

        duration = events[-1].epoch_timestamp - events[0].epoch_timestamp
        duration = max(0.0, duration)

        # A. High Request Volume with Absent/Low Browser Telemetry
        telemetry_ratio = telemetry_count / max(1, http_requests_count)
        if (
            http_requests_count >= cfg.min_requests_for_telemetry_check
            and telemetry_ratio < cfg.min_telemetry_ratio
        ):
            emitted_anomalies.append(
                AnomalyRecord(
                    code="ANOMALY_HIGH_REQUEST_LOW_TELEMETRY",
                    explanation=(
                        f"High volume of server requests ({http_requests_count}) with "
                        f"disproportionately low browser telemetry ({telemetry_count} events, "
                        f"ratio: {telemetry_ratio:.2f}, expected >= {cfg.min_telemetry_ratio})."
                    ),
                    evidence={
                        "request_count": http_requests_count,
                        "telemetry_count": telemetry_count,
                        "telemetry_ratio": round(telemetry_ratio, 3),
                        "threshold_ratio": cfg.min_telemetry_ratio,
                    },
                )
            )
            if state != FSMState.RECOVERY:
                state = FSMState.SUSPICIOUS

        # B. Burst Request Rate (superhuman request frequency)
        # Measured over the densest stretch of page requests that fits within
        # burst_window_seconds, so a burst is not diluted by earlier normal browsing.
        request_times = sorted(
            e.epoch_timestamp for e in events
            if e.source == "HTTP_REQUEST" and e.symbol != FSMSymbol.HTTP_API
        )
        rps = 0.0
        burst_count = 0
        burst_span = 0.0
        start = 0
        for end in range(len(request_times)):
            while request_times[end] - request_times[start] > cfg.burst_window_seconds:
                start += 1
            count = end - start + 1
            span = request_times[end] - request_times[start]
            if count >= 5 and span > 0.0 and count / span > rps:
                rps, burst_count, burst_span = count / span, count, span

        if burst_count:
            if rps >= cfg.burst_rps_threshold:
                emitted_anomalies.append(
                    AnomalyRecord(
                        code="ANOMALY_RAPID_REQUEST_BURST",
                        explanation=(
                            f"Burst request rate of {rps:.1f} req/s exceeds human limit "
                            f"({burst_count} requests in {burst_span:.2f}s, "
                            f"threshold: {cfg.burst_rps_threshold:.1f} req/s)."
                        ),
                        evidence={
                            "burst_rate_rps": round(rps, 2),
                            "burst_request_count": burst_count,
                            "burst_span_seconds": round(burst_span, 3),
                            "duration_seconds": round(duration, 2),
                            "threshold_rps": cfg.burst_rps_threshold,
                        },
                    )
                )
                if state != FSMState.RECOVERY:
                    state = FSMState.SUSPICIOUS

        # C. Contextual Signal: Low Interaction
        # (Recorded as evidence, NEVER as sole bot proof, preserving a11y & mobile users)
        if http_requests_count >= cfg.low_interaction_context_threshold and interaction_count == 0:
            emitted_anomalies.append(
                AnomalyRecord(
                    code="SIGNAL_LOW_INTERACTION",
                    explanation=(
                        f"Contextual observation: {http_requests_count} pages browsed with "
                        "no recorded mouse or keyboard interaction gestures."
                    ),
                    evidence={
                        "request_count": http_requests_count,
                        "interaction_count": 0,
                    },
                )
            )

        # -------------------------------------------------------------------
        # Explainability & Risk Scoring Computation
        # -------------------------------------------------------------------
        unique_anomalies: dict[str, AnomalyRecord] = {}
        for a in emitted_anomalies:
            if a.code not in unique_anomalies:
                unique_anomalies[a.code] = a

        # Compute deterministic penalty score
        penalty_map = anomaly_penalties(cfg)

        raw_score = 0.0
        # If ONLY low interaction context exists, score remains 0.0 (safe for mobile/a11y)
        has_primary_anomaly = any(code != "SIGNAL_LOW_INTERACTION" for code in unique_anomalies)

        for code in unique_anomalies:
            if code == "SIGNAL_LOW_INTERACTION" and not has_primary_anomaly:
                continue
            raw_score += penalty_map.get(code, 0.10)

        # Mitigate penalty if sustained human recovery occurred
        if state == FSMState.RECOVERY:
            raw_score *= cfg.recovery_mitigation_factor

        # Bounded between 0.0 and 1.0
        final_score = round(max(0.0, min(1.0, raw_score)), 4)

        anomaly_codes = list(unique_anomalies.keys())
        reasons = [a.explanation for a in unique_anomalies.values()]

        return FSMResult(
            final_state=state,
            transition_history=history,
            anomaly_codes=anomaly_codes,
            reasons=reasons,
            risk_score=final_score,
            evaluation_metadata={
                "total_events": telemetry_count,
                "total_requests": http_requests_count,
                "total_inputs": len(events),
                "duration_seconds": round(duration, 3),
                "telemetry_ratio": round(telemetry_ratio, 3),
                "data_quality_warnings": warnings,
            },
        )
