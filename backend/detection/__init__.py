"""
BotShield – Behavioral Detection Package (Phase 3)
"""

from backend.detection.fsm import (
    AnomalyRecord,
    BehavioralFSM,
    FSMConfig,
    FSMEvent,
    FSMResult,
    FSMState,
    FSMSymbol,
    TransitionStep,
)
from backend.detection.service import evaluate_session_fsm

__all__ = [
    "FSMState",
    "FSMSymbol",
    "FSMEvent",
    "FSMResult",
    "FSMConfig",
    "AnomalyRecord",
    "TransitionStep",
    "BehavioralFSM",
    "evaluate_session_fsm",
]

