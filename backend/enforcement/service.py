"""
BotShield – Enforcement Decision Service (Phase 6)

Evaluates incoming HTTP requests against configured enforcement policies,
persisted session risk scores, route sensitivity, and sliding-window rate limits.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.database.db import get_connection, get_session_risk
from backend.enforcement.limiter import BoundedRateLimiter
from backend.enforcement.policy import (
    EnforcementAction,
    EnforcementConfig,
    EnforcementMode,
    EnforcementPolicy,
)
from backend.scoring import HybridConfig, RiskLevel, ScoringMode, classify_risk

logger = logging.getLogger("botshield.enforcement.service")


def _now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_to_epoch(ts_str: str | None) -> float | None:
    """Parse an ISO 8601 timestamp string into epoch seconds."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        # Python 3.11+ handles Z and offsets cleanly
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class EnforcementDecision:
    """
    Structured outcome of an enforcement evaluation.

    Attributes:
        action: Determined mitigation action (ALLOW, RATE_LIMIT, BLOCK, CHALLENGE).
        enforced: True if action was actively enforced; False if observed in MONITOR mode.
        session_id: Correlated session ID, or None if anonymous/untrusted.
        client_ip: Client remote IP address.
        path: Normalized request path.
        risk_level: Evaluated RiskLevel (LOW, MEDIUM, HIGH) or None.
        risk_score: Normalized risk score in [0.0, 1.0] from risk record, or None.
        composite_score: Scaled composite score in [0.0, 100.0], or None.
        scoring_mode: Mode used to calculate risk, or None.
        reason: Human-readable rationale for the decision.
        retry_after: Integer seconds before retry is allowed (for RATE_LIMIT), or None.
        is_stale: True if persisted risk score was older than TTL.
        mode: Active EnforcementMode (MONITOR, ENFORCE, DISABLED).
        timestamp: UTC ISO timestamp of decision.
    """
    action: EnforcementAction
    enforced: bool
    session_id: str | None
    client_ip: str
    path: str
    risk_level: RiskLevel | None
    risk_score: float | None
    composite_score: float | None
    scoring_mode: ScoringMode | None
    reason: str
    retry_after: int | None = None
    is_stale: bool = False
    mode: EnforcementMode = EnforcementMode.MONITOR
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize decision to a JSON-compatible dictionary without secrets."""
        return {
            "action": self.action.value,
            "enforced": self.enforced,
            "session_id": self.session_id,
            "client_ip": self.client_ip,
            "path": self.path,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "risk_score": self.risk_score,
            "composite_score": self.composite_score,
            "scoring_mode": self.scoring_mode.value if self.scoring_mode else None,
            "reason": self.reason,
            "retry_after": self.retry_after,
            "is_stale": self.is_stale,
            "mode": self.mode.value,
            "timestamp": self.timestamp,
        }


class EnforcementService:
    """
    Orchestrates session risk lookups, route sensitivity checks,
    policy evaluation, rate limiting, and decision reporting.
    """

    def __init__(
        self,
        config: EnforcementConfig | None = None,
        limiter: BoundedRateLimiter | None = None,
    ) -> None:
        self.config = config or EnforcementConfig()
        self.limiter = limiter or BoundedRateLimiter(max_tracked_clients=self.config.max_tracked_clients)

    def is_route_excluded(self, path: str) -> bool:
        """Check if request path matches configured exclusion rules."""
        for exc in self.config.excluded_routes:
            if path == exc or path.startswith(exc + "/") or (exc.endswith("*") and path.startswith(exc[:-1])):
                return True
        return False

    def is_route_sensitive(self, path: str) -> bool:
        """Check if request path is configured as sensitive."""
        for sens in self.config.sensitive_routes:
            if path == sens or path.startswith(sens + "/"):
                return True
        return False

    def evaluate_request(
        self,
        session_id: str | None,
        client_ip: str,
        path: str,
        db_path: str | None = None,
        now: float | None = None,
    ) -> EnforcementDecision:
        """
        Evaluate policy and rate limits for an incoming request.

        Guarantees:
        - Never blocks requests when mode == MONITOR
        - Never blocks requests to excluded routes (static assets, health checks, telemetry)
        - Respects route sensitivity filters
        - Fails open gracefully upon unexpected database or internal errors
        """
        current_time = time.time() if now is None else float(now)
        mode = self.config.mode

        # 1. Enforcement disabled
        if mode == EnforcementMode.DISABLED:
            return EnforcementDecision(
                action=EnforcementAction.ALLOW,
                enforced=False,
                session_id=session_id,
                client_ip=client_ip,
                path=path,
                risk_level=None,
                risk_score=None,
                composite_score=None,
                scoring_mode=None,
                reason="Enforcement subsystem is disabled.",
                mode=mode,
            )

        # 2. Excluded route bypass
        if self.is_route_excluded(path):
            return EnforcementDecision(
                action=EnforcementAction.ALLOW,
                enforced=False,
                session_id=session_id,
                client_ip=client_ip,
                path=path,
                risk_level=None,
                risk_score=None,
                composite_score=None,
                scoring_mode=None,
                reason="Path is in excluded routes; permitted without enforcement.",
                mode=mode,
            )

        # 3. Non-sensitive route bypass
        if self.config.sensitive_routes and not self.is_route_sensitive(path):
            return EnforcementDecision(
                action=EnforcementAction.ALLOW,
                enforced=False,
                session_id=session_id,
                client_ip=client_ip,
                path=path,
                risk_level=None,
                risk_score=None,
                composite_score=None,
                scoring_mode=None,
                reason="Path is non-sensitive; permitted normally.",
                mode=mode,
            )

        # 4. Retrieve risk record from database with freshness check
        risk_level: RiskLevel | None = None
        risk_score: float | None = None
        composite_score: float | None = None
        scoring_mode: ScoringMode | None = None
        is_stale = False
        has_error = False

        if session_id:
            try:
                conn = get_connection(db_path)
                try:
                    risk_row = get_session_risk(conn, session_id)
                finally:
                    conn.close()

                if risk_row is not None:
                    risk_score = float(risk_row["risk_score"])
                    composite_score = round(risk_score * 100.0, 1)
                    risk_level = classify_risk(composite_score, HybridConfig())

                    updated_epoch = _parse_iso_to_epoch(risk_row["updated_at"])
                    if updated_epoch is not None:
                        age = current_time - updated_epoch
                        if age > self.config.risk_cache_ttl_seconds:
                            is_stale = True

                    reasons_str = str(risk_row["reasons"])
                    if "FSM-only" in reasons_str or risk_row["ml_score"] == 0.0:
                        scoring_mode = ScoringMode.FSM_ONLY
                    else:
                        scoring_mode = ScoringMode.HYBRID
            except Exception as exc:
                logger.error("[ENFORCEMENT] Error loading risk record for session %s: %s", session_id, exc)
                has_error = True

        # 5. Evaluate policy
        policy_action, reason = EnforcementPolicy.evaluate(
            risk_level=risk_level,
            scoring_mode=scoring_mode,
            is_stale=is_stale,
            has_error=has_error,
            config=self.config,
        )

        retry_after: int | None = None

        # 6. Apply rate limiting if policy dictates RATE_LIMIT
        if policy_action == EnforcementAction.RATE_LIMIT:
            limiter_key = session_id or client_ip
            is_limited, wait_seconds = self.limiter.check(
                key=limiter_key,
                max_requests=self.config.rate_limit_requests,
                window_seconds=self.config.rate_limit_window_seconds,
                now=current_time,
            )
            if is_limited:
                retry_after = wait_seconds
                reason += f" Rate limit exceeded ({self.config.rate_limit_requests} req / {self.config.rate_limit_window_seconds}s). Retry after {retry_after}s."
            else:
                # Quota has not been exhausted yet
                policy_action = EnforcementAction.ALLOW
                reason += " Rate limiting monitored (within quota)."

        # 7. Determine active enforcement
        if mode == EnforcementMode.MONITOR:
            enforced = False
            logger.info(
                "[ENFORCEMENT_MONITOR] session=%s ip=%s path=%s action=%s enforced=False reason='%s'",
                session_id or "anon",
                client_ip,
                path,
                policy_action.value,
                reason,
            )
        else:
            enforced = (policy_action in (EnforcementAction.BLOCK, EnforcementAction.RATE_LIMIT, EnforcementAction.CHALLENGE))
            if enforced:
                logger.warning(
                    "[ENFORCEMENT_ACTIVE] session=%s ip=%s path=%s action=%s enforced=True reason='%s'",
                    session_id or "anon",
                    client_ip,
                    path,
                    policy_action.value,
                    reason,
                )

        return EnforcementDecision(
            action=policy_action,
            enforced=enforced,
            session_id=session_id,
            client_ip=client_ip,
            path=path,
            risk_level=risk_level,
            risk_score=risk_score,
            composite_score=composite_score,
            scoring_mode=scoring_mode,
            reason=reason,
            retry_after=retry_after,
            is_stale=is_stale,
            mode=mode,
        )
