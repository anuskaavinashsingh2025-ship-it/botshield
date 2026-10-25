"""
BotShield – Hybrid Risk Scoring Service (Phase 5)

Combines deterministic Finite State Machine (FSM) behavioral detection
with machine-learned behavioral anomaly detection (Isolation Forest)
into a unified, explainable session risk score and classification.

Design Principles:
- FSM score (0.0 - 1.0): Rule-based violations of web navigation invariants.
- ML score (0.0 - 1.0): Unsupervised behavioral anomaly detection.
- Deterministic combination with explicit weights and thresholds.
- Safe cold-start handling: When ML is untrained or unavailable, scores
  rely purely on FSM evidence without fabricating an ML score or assuming zero risk.
- Explicit scoring modes: HYBRID, FSM_ONLY, INSUFFICIENT_EVIDENCE.
- Clean persistence semantics: Non-persisting evaluation by default (persist=False),
  transactional upsert to session_risks when persist=True.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database.db import (
    get_connection,
    get_events,
    get_requests,
    insert_risk_history,
    upsert_session_risk,
)
from backend.detection.fsm import FSMConfig, FSMResult
from backend.detection.service import evaluate_session_fsm
from backend.ml.model import (
    MLInferenceResult,
    MLModelStatus,
)

logger = logging.getLogger("botshield.scoring.service")

# Score contract (see docs/hybrid_scoring.md). Every score is in [0, 1] and
# higher always means more bot-like.
#   HYBRID:                H = fsm_weight * FSM + ml_weight * ML   (ML available)
#   FSM_ONLY:              H = FSM                                  (ML unavailable)
#   INSUFFICIENT_EVIDENCE: H = 0                                    (no activity)
# Classification uses H * 100 against HybridConfig.low/high_threshold.
HYBRID_FORMULA = "H = 0.60 * FSM + 0.40 * ML (HYBRID); H = FSM (FSM_ONLY); H = 0 (INSUFFICIENT_EVIDENCE)"


def _now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class RiskLevel(str, enum.Enum):
    """Categorical risk classification for a session."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ScoringMode(str, enum.Enum):
    """Operational mode used to compute the composite session risk."""
    HYBRID = "HYBRID"
    FSM_ONLY = "FSM_ONLY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class HybridConfig:
    """
    Configuration parameters for hybrid score weighting and risk classification.

    Weights must be non-negative and sum to 1.0.
    Thresholds define risk bands:
      - [0.0, low_threshold) -> LOW
      - [low_threshold, high_threshold) -> MEDIUM
      - [high_threshold, 100.0] -> HIGH
    """
    fsm_weight: float = 0.60
    ml_weight: float = 0.40
    low_threshold: float = 35.0
    high_threshold: float = 70.0

    def __post_init__(self) -> None:
        if self.fsm_weight < 0.0 or self.ml_weight < 0.0:
            raise ValueError("Scoring weights must be non-negative.")
        if not math.isclose(self.fsm_weight + self.ml_weight, 1.0, rel_tol=1e-5):
            raise ValueError(
                f"Weights must sum to 1.0 (got fsm_weight={self.fsm_weight}, ml_weight={self.ml_weight})."
            )
        if not (0.0 <= self.low_threshold < self.high_threshold <= 100.0):
            raise ValueError(
                f"Thresholds must satisfy 0.0 <= low_threshold < high_threshold <= 100.0 "
                f"(got low={self.low_threshold}, high={self.high_threshold})."
            )


@dataclass(frozen=True)
class HybridRiskResult:
    """
    Structured outcome of hybrid session risk evaluation.

    Attributes:
        session_id: Target session identifier.
        composite_score: Combined risk score in range [0.0, 100.0], rounded to 1 decimal place.
        normalized_score: Combined risk score in range [0.0, 1.0], rounded to 4 decimal places.
        risk_level: Categorical classification (LOW, MEDIUM, HIGH).
        scoring_mode: Mode used for calculation (HYBRID, FSM_ONLY, INSUFFICIENT_EVIDENCE).
        fsm_score: Deterministic behavioral score [0.0, 1.0].
        ml_score: ML anomaly score [0.0, 1.0] if model was available and valid, else None.
        fsm_result: Full FSM evaluation object.
        ml_result: Full ML inference result object.
        reasons: Combined list of explainable behavioral flags.
        warnings: Data quality warnings and model status notices.
        model_status: Status of the ML anomaly detector.
        evaluated_at: UTC ISO timestamp of evaluation.
    """
    session_id: str
    composite_score: float
    normalized_score: float
    risk_level: RiskLevel
    scoring_mode: ScoringMode
    fsm_score: float
    ml_score: float | None
    fsm_result: FSMResult
    ml_result: MLInferenceResult
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_status: MLModelStatus = MLModelStatus.UNTRAINED
    evaluated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation result to a JSON-compatible dictionary."""
        return {
            "session_id": self.session_id,
            "composite_score": self.composite_score,
            "normalized_score": self.normalized_score,
            "risk_level": self.risk_level.value,
            "scoring_mode": self.scoring_mode.value,
            "fsm_score": self.fsm_score,
            "ml_score": self.ml_score,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "model_status": self.model_status.value,
            "evaluated_at": self.evaluated_at,
            "fsm_summary": {
                "risk_score": self.fsm_result.risk_score,
                "final_state": self.fsm_result.final_state.value,
                "reasons": self.fsm_result.reasons,
            },
            "ml_summary": {
                "status": self.ml_result.status.value,
                "anomaly_score": self.ml_result.anomaly_score,
                "is_anomaly": self.ml_result.is_anomaly,
                "decision_threshold": self.ml_result.model_metadata.get("decision_threshold"),
                "feature_summary": self.ml_result.feature_summary,
            },
        }


def compute_hybrid_score(
    fsm_score: float,
    ml_score: float | None,
    config: HybridConfig,
    ml_status: MLModelStatus = MLModelStatus.AVAILABLE,
) -> tuple[float, float, ScoringMode, list[str]]:
    """
    Compute composite and normalized scores using defined weighting and fallback rules.

    Returns:
        (composite_score, normalized_score, scoring_mode, warnings)
        where composite_score in [0.0, 100.0] and normalized_score in [0.0, 1.0].
    """
    warnings: list[str] = []

    # Validate and clamp FSM score
    if not isinstance(fsm_score, (int, float)) or isinstance(fsm_score, bool) or math.isnan(fsm_score) or math.isinf(fsm_score):
        fsm_clean = 0.0
        warnings.append("WARNING_INVALID_FSM_SCORE_RESET_TO_ZERO")
    else:
        fsm_clean = max(0.0, min(1.0, float(fsm_score)))

    # Determine whether ML score is usable
    ml_usable = False
    ml_clean: float | None = None

    if ml_status == MLModelStatus.AVAILABLE and ml_score is not None:
        if isinstance(ml_score, (int, float)) and not isinstance(ml_score, bool):
            if not (math.isnan(ml_score) or math.isinf(ml_score)):
                ml_usable = True
                ml_clean = max(0.0, min(1.0, float(ml_score)))
            else:
                warnings.append("WARNING_NON_FINITE_ML_SCORE: ML anomaly score was non-finite (NaN/Inf).")
        else:
            warnings.append("WARNING_INVALID_ML_SCORE_TYPE")

    if ml_usable and ml_clean is not None:
        # Hybrid combination mode
        normalized = (config.fsm_weight * fsm_clean) + (config.ml_weight * ml_clean)
        normalized = max(0.0, min(1.0, normalized))
        composite = round(normalized * 100.0, 1)
        mode = ScoringMode.HYBRID
    else:
        # Safe FSM-only fallback mode
        normalized = fsm_clean
        composite = round(normalized * 100.0, 1)
        mode = ScoringMode.FSM_ONLY
        if ml_status == MLModelStatus.UNTRAINED:
            warnings.append("ML model is untrained; evaluated in FSM-only fallback mode.")
        elif ml_status in (MLModelStatus.INCOMPATIBLE, MLModelStatus.CORRUPT, MLModelStatus.ERROR):
            warnings.append(f"ML model error ({ml_status.value}); evaluated in FSM-only fallback mode.")
        elif ml_score is None:
            warnings.append("ML score is missing; evaluated in FSM-only fallback mode.")

    return composite, round(normalized, 4), mode, warnings


def classify_risk(composite_score: float, config: HybridConfig) -> RiskLevel:
    """Map a composite score (0.0 - 100.0) to categorical RiskLevel."""
    if composite_score < config.low_threshold:
        return RiskLevel.LOW
    elif composite_score < config.high_threshold:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.HIGH


def evaluate_session_risk(
    session_id: str,
    db_path: str | None = None,
    model_path: str | Path | None = None,
    fsm_config: FSMConfig | None = None,
    config: HybridConfig | None = None,
    persist: bool = False,
) -> HybridRiskResult:
    """
    Evaluate hybrid behavioral risk for a session.

    Coordinates:
    1. Retrieval and validation of session records from SQLite.
    2. Execution of behavioral FSM detector (persist=False).
    3. Execution of ML anomaly detector (persist=False).
    4. Deterministic hybrid score calculation with fallback when ML is unavailable.
    5. Categorical risk level classification (LOW, MEDIUM, HIGH).
    6. Aggregation of explainability reasons and quality warnings.
    7. Optional transactional persistence to session_risks table when persist=True.

    Parameters:
        session_id: Session identifier.
        db_path: Optional SQLite database file path.
        model_path: Optional path to serialized Isolation Forest model.
        fsm_config: Optional FSM configuration overrides.
        config: Optional HybridConfig specifying weights and thresholds.
        persist: If True, upsert the composite result into session_risks table.

    Returns:
        HybridRiskResult containing all scores, modes, reasons, and metadata.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string.")
    clean_session_id = session_id.strip()

    from backend.ml.service import evaluate_session_ml

    active_config = config if config is not None else HybridConfig()

    conn = get_connection(db_path)
    try:
        event_rows = get_events(conn, session_id=clean_session_id)
        request_rows = get_requests(conn, session_id=clean_session_id, limit=5000)
    finally:
        conn.close()

    # Step 1: Handle sessions with zero recorded activity (insufficient evidence)
    if not event_rows and not request_rows:
        empty_fsm = evaluate_session_fsm(
            session_id=clean_session_id,
            db_path=db_path,
            config=fsm_config,
            persist=False,
        )
        empty_ml = evaluate_session_ml(
            session_id=clean_session_id,
            db_path=db_path,
            model_path=model_path,
            persist=False,
        )

        result = HybridRiskResult(
            session_id=clean_session_id,
            composite_score=0.0,
            normalized_score=0.0,
            risk_level=RiskLevel.LOW,
            scoring_mode=ScoringMode.INSUFFICIENT_EVIDENCE,
            fsm_score=0.0,
            ml_score=None,
            fsm_result=empty_fsm,
            ml_result=empty_ml,
            reasons=[],
            warnings=["INSUFFICIENT_SESSION_EVIDENCE: No telemetry events or HTTP requests recorded."],
            model_status=empty_ml.status,
        )

        if persist:
            _persist_result(result, db_path)

        return result

    # Step 2: Evaluate FSM behavioral detection in non-persisting mode
    fsm_res = evaluate_session_fsm(
        session_id=clean_session_id,
        db_path=db_path,
        config=fsm_config,
        persist=False,
    )

    # Step 3: Evaluate ML behavioral anomaly detector in non-persisting mode
    ml_res = evaluate_session_ml(
        session_id=clean_session_id,
        db_path=db_path,
        model_path=model_path,
        persist=False,
    )

    # Step 4: Calculate composite and normalized scores
    comp_score, norm_score, mode, score_warnings = compute_hybrid_score(
        fsm_score=fsm_res.risk_score,
        ml_score=ml_res.anomaly_score,
        config=active_config,
        ml_status=ml_res.status,
    )

    # Step 5: Risk Level classification
    risk_level = classify_risk(comp_score, active_config)

    # Step 6: Assemble explainability reasons and surface warnings
    combined_reasons = list(fsm_res.reasons)
    if ml_res.is_anomaly and "ML_BEHAVIORAL_ANOMALY" not in combined_reasons:
        combined_reasons.append("ML_BEHAVIORAL_ANOMALY")
    if mode == ScoringMode.FSM_ONLY and "FSM-only fallback mode (ML model untrained or unavailable)" not in combined_reasons:
        combined_reasons.append("FSM-only fallback mode (ML model untrained or unavailable)")

    combined_warnings = list(score_warnings)
    if fsm_res.evaluation_metadata and "data_quality_warnings" in fsm_res.evaluation_metadata:
        combined_warnings.extend(fsm_res.evaluation_metadata["data_quality_warnings"])
    if ml_res.warnings:
        combined_warnings.extend(ml_res.warnings)

    # Deduplicate warnings while preserving order
    deduped_warnings = list(dict.fromkeys(combined_warnings))

    # Effective ml_score for result dataclass (None if unusable)
    effective_ml_score = ml_res.anomaly_score if (ml_res.status == MLModelStatus.AVAILABLE and ml_res.anomaly_score is not None) else None

    result = HybridRiskResult(
        session_id=clean_session_id,
        composite_score=comp_score,
        normalized_score=norm_score,
        risk_level=risk_level,
        scoring_mode=mode,
        fsm_score=fsm_res.risk_score,
        ml_score=effective_ml_score,
        fsm_result=fsm_res,
        ml_result=ml_res,
        reasons=combined_reasons,
        warnings=deduped_warnings,
        model_status=ml_res.status,
    )

    # Step 7: Explicit persistence into session_risks table if requested
    if persist:
        _persist_result(result, db_path)
        logger.info(
            "Persisted hybrid risk score %.4f (composite %.1f, %s) for session %s",
            norm_score,
            comp_score,
            risk_level.value,
            clean_session_id,
        )

    return result


def _persist_result(result: HybridRiskResult, db_path: str | None) -> None:
    """Upsert the latest scores into session_risks and append a history row."""
    ml_available = result.ml_score is not None
    write_conn = get_connection(db_path)
    try:
        upsert_session_risk(
            conn=write_conn,
            session_id=result.session_id,
            risk_score=result.normalized_score,
            fsm_score=result.fsm_score,
            ml_score=result.ml_score,
            reasons=result.reasons,
            scoring_mode=result.scoring_mode.value,
            ml_status=result.model_status.value,
            fsm_state=result.fsm_result.final_state.value,
            # Only meaningful when ML actually scored real activity
            ml_is_anomaly=result.ml_result.is_anomaly if ml_available else None,
        )
        insert_risk_history(
            write_conn,
            session_id=result.session_id,
            fsm_score=result.fsm_score,
            ml_score=result.ml_score,
            hybrid_score=result.normalized_score,
            risk_level=result.risk_level.value,
            scoring_mode=result.scoring_mode.value,
            ml_status=result.model_status.value,
            fsm_state=result.fsm_result.final_state.value,
        )
        write_conn.commit()
    finally:
        write_conn.close()
