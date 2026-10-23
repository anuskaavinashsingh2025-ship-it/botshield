"""
BotShield – Risk-Based Enforcement Policy (Phase 6)

Maps hybrid risk evaluation outcomes (RiskLevel, ScoringMode, freshness, and error states)
to explicit, configurable enforcement actions:
- ALLOW: permit request to proceed without interruption.
- RATE_LIMIT: apply bounded rate limiting to throttle excessive requests.
- BLOCK: reject access with HTTP 403 Forbidden.
- CHALLENGE: candidate for progressive verification.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from backend.scoring import RiskLevel, ScoringMode


class EnforcementMode(str, enum.Enum):
    """Operational mode for the enforcement subsystem."""
    MONITOR = "monitor"    # Observe and log decisions without blocking or throttling (default)
    ENFORCE = "enforce"    # Actively apply rate limiting or blocking per policy
    DISABLED = "disabled"  # Completely bypass policy evaluation


class EnforcementAction(str, enum.Enum):
    """Categorical mitigation action determined by policy."""
    ALLOW = "ALLOW"
    RATE_LIMIT = "RATE_LIMIT"
    BLOCK = "BLOCK"
    CHALLENGE = "CHALLENGE"


@dataclass(frozen=True)
class EnforcementConfig:
    """
    Configuration parameters for enforcement policy and middleware behavior.
    """
    mode: EnforcementMode = EnforcementMode.MONITOR
    fail_open: bool = True
    low_risk_action: EnforcementAction = EnforcementAction.ALLOW
    medium_risk_action: EnforcementAction = EnforcementAction.RATE_LIMIT
    high_risk_action: EnforcementAction = EnforcementAction.BLOCK
    fallback_action: EnforcementAction = EnforcementAction.ALLOW
    risk_cache_ttl_seconds: float = 60.0
    rate_limit_requests: int = 10
    rate_limit_window_seconds: float = 10.0
    sensitive_routes: tuple[str, ...] = ("/login",)
    excluded_routes: tuple[str, ...] = ("/static", "/favicon.ico", "/api/events")
    max_tracked_clients: int = 5000

    def __post_init__(self) -> None:
        if self.rate_limit_requests <= 0:
            raise ValueError("rate_limit_requests must be positive.")
        if self.rate_limit_window_seconds <= 0.0:
            raise ValueError("rate_limit_window_seconds must be positive.")
        if self.risk_cache_ttl_seconds <= 0.0:
            raise ValueError("risk_cache_ttl_seconds must be positive.")
        if self.max_tracked_clients <= 0:
            raise ValueError("max_tracked_clients must be positive.")


class EnforcementPolicy:
    """
    Pure, independently testable decision policy evaluating session risk status.
    """

    @staticmethod
    def evaluate(
        risk_level: RiskLevel | None,
        scoring_mode: ScoringMode | None = None,
        is_stale: bool = False,
        has_error: bool = False,
        config: EnforcementConfig | None = None,
    ) -> tuple[EnforcementAction, str]:
        """
        Determine the appropriate enforcement action and explainability reason.

        Parameters:
            risk_level: Evaluated RiskLevel (LOW, MEDIUM, HIGH) or None if absent.
            scoring_mode: ScoringMode used to compute risk (HYBRID, FSM_ONLY, INSUFFICIENT_EVIDENCE).
            is_stale: True if the persisted risk record exceeded risk_cache_ttl_seconds.
            has_error: True if database or evaluation failures occurred.
            config: Active EnforcementConfig (defaults to EnforcementConfig()).

        Returns:
            (action, reason)
        """
        cfg = config or EnforcementConfig()

        if has_error:
            return cfg.fallback_action, "Evaluation or database error; applied configured fallback action."

        if is_stale:
            return cfg.fallback_action, "Risk record is stale; applied configured fallback action."

        if risk_level is None:
            return cfg.fallback_action, "No existing risk record found; applied configured fallback action."

        if scoring_mode == ScoringMode.INSUFFICIENT_EVIDENCE:
            return EnforcementAction.ALLOW, "Insufficient session evidence; normal browsing permitted."

        if risk_level == RiskLevel.LOW:
            return cfg.low_risk_action, "Session classified as LOW risk."

        if risk_level == RiskLevel.MEDIUM:
            reason = "Session classified as MEDIUM risk."
            if scoring_mode == ScoringMode.FSM_ONLY:
                reason += " (Evaluated under FSM-only fallback)."
            return cfg.medium_risk_action, reason

        if risk_level == RiskLevel.HIGH:
            reason = "Session classified as HIGH risk."
            if scoring_mode == ScoringMode.FSM_ONLY:
                reason += " (Evaluated under FSM-only fallback)."
            return cfg.high_risk_action, reason

        return cfg.fallback_action, f"Unrecognized risk level '{risk_level}'; applied fallback."
