"""
BotShield – Hybrid Risk Scoring Package (Phase 5)

Exports the public API for composite session risk evaluation.
"""

from backend.scoring.service import (
    HybridConfig,
    HybridRiskResult,
    RiskLevel,
    ScoringMode,
    classify_risk,
    compute_hybrid_score,
    evaluate_session_risk,
)

from backend.scoring.worker import (
    AsyncEvaluationWorker,
    get_pending_sessions,
    init_app_background_worker,
)

__all__ = [
    "AsyncEvaluationWorker",
    "HybridConfig",
    "HybridRiskResult",
    "RiskLevel",
    "ScoringMode",
    "classify_risk",
    "compute_hybrid_score",
    "evaluate_session_risk",
    "get_pending_sessions",
    "init_app_background_worker",
]
