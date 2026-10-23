"""
BotShield – ML Behavioral Evaluation Service (Phase 4)

Extracts a session's browser telemetry and server-observed HTTP requests from SQLite,
computes normalized behavioral features, evaluates the ML anomaly detector, and
persists the ML score while strictly preserving FSM score, combined risk score, and reasons.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.database.db import (
    get_connection,
    get_events,
    get_requests,
    update_session_ml_risk,
)
from backend.ml.features import extract_session_features
from backend.ml.model import (
    IsolationForestModel,
    MLInferenceResult,
    MLModelStatus,
)

logger = logging.getLogger("botshield.ml.service")


def evaluate_session_ml(
    session_id: str,
    db_path: str | None = None,
    model_path: str | Path | None = None,
    persist: bool = False,
) -> MLInferenceResult:
    """
    Extract session data from the database, compute features, and evaluate the ML model.

    session_id: The UUID session identifier to evaluate.
    db_path: Optional SQLite database file path.
    model_path: Optional path to serialized model artifact.
    persist: If True and the model is AVAILABLE with a valid score, persist ml_score
             to session_risks while preserving existing fsm_score, risk_score, and reasons.

    Returns: MLInferenceResult containing status, score, features, and warnings.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    clean_session_id = session_id.strip()

    conn = get_connection(db_path)
    try:
        # Load session telemetry and requests using existing database helpers
        event_rows = get_events(conn, session_id=clean_session_id)
        request_rows = get_requests(conn, session_id=clean_session_id, limit=5000)

        # Extract normalized behavioral feature vector
        features = extract_session_features(events=event_rows, requests=request_rows)

        # Execute ML inference
        model = IsolationForestModel(model_path=model_path)
        result = model.predict(features)

        # Persist ML score if requested and inference produced a valid score
        if persist and result.status == MLModelStatus.AVAILABLE and result.anomaly_score is not None:
            update_session_ml_risk(
                conn,
                session_id=clean_session_id,
                ml_score=result.anomaly_score,
            )
            conn.commit()
            logger.info(
                "Persisted ML anomaly score %.4f for session %s",
                result.anomaly_score,
                clean_session_id,
            )

        return result
    finally:
        conn.close()
