"""
BotShield – Behavioral ML Model & Anomaly Detection (Phase 4)

Provides unsupervised behavioral anomaly detection using an Isolation Forest pipeline.
Supports model persistence, validation, safe deserialization, and deterministic inference.

When no trained model artifact is available, the component reports an explicit UNTRAINED
status and returns an anomaly score of None, rather than a misleading zero.
"""

from __future__ import annotations

import enum
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.ml.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    ExtractedFeatures,
)

logger = logging.getLogger("botshield.ml.model")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "isolation_forest.joblib"


class MLModelStatus(str, enum.Enum):
    """Execution status for the ML anomaly detection model."""
    AVAILABLE = "available"
    UNTRAINED = "untrained"
    INCOMPATIBLE = "incompatible"
    CORRUPT = "corrupt"
    ERROR = "error"


@dataclass(frozen=True)
class MLInferenceResult:
    """Documented structured output from ML behavioral anomaly scoring."""
    status: MLModelStatus
    schema_version: str
    anomaly_score: float | None
    is_anomaly: bool | None
    feature_vector: list[float]
    feature_summary: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "schema_version": self.schema_version,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "feature_vector": self.feature_vector,
            "feature_summary": self.feature_summary,
            "warnings": self.warnings,
            "model_metadata": self.model_metadata,
        }


def get_model_path(custom_path: str | Path | None = None) -> Path:
    """Resolve configured model path, honoring BOTSHIELD_MODEL_PATH env var."""
    if custom_path:
        return Path(custom_path)
    env_path = os.environ.get("BOTSHIELD_MODEL_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_MODEL_PATH


def train_unsupervised_model(
    training_data: list[list[float]] | np.ndarray,
    model_path: str | Path | None = None,
    contamination: float = 0.1,
    random_state: int = 42,
    min_samples: int = 10,
    dataset_type: str = "synthetic_baseline",
) -> dict[str, Any]:
    """
    Train an unsupervised Isolation Forest anomaly detection pipeline on behavioral feature vectors.

    training_data: list of 21-element feature vectors or (N, 21) numpy array
    model_path: optional destination path for saving the model artifact
    contamination: expected proportion of outliers (0.001 to 0.5)
    random_state: seed for reproducibility
    min_samples: minimum training samples required (minimum 5)
    dataset_type: description of source dataset (e.g. 'synthetic_baseline', 'database_sessions')

    Uses atomic artifact replacement via temporary file and os.replace to prevent
    corrupting an existing active model if training or serialization fails.

    Raises ValueError if training data fails validation.
    """
    if training_data is None:
        raise ValueError("training_data cannot be None")

    X = np.asarray(training_data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"training_data must be a 2D array or list of lists (got ndim={X.ndim})")

    n_samples, n_features = X.shape
    if n_features != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature count mismatch: expected {len(FEATURE_NAMES)} features, got {n_features}"
        )

    if n_samples < min_samples:
        raise ValueError(
            f"Insufficient training samples: got {n_samples}, minimum required is {min_samples}"
        )

    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("training_data contains NaN or Infinite values")

    if not (0.001 <= contamination <= 0.5):
        raise ValueError(f"contamination must be between 0.001 and 0.5 (got {contamination})")

    # Construct pipeline with StandardScaler and IsolationForest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "forest",
            IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=random_state,
                n_jobs=1,
            ),
        ),
    ])

    pipeline.fit(X)

    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_sample_count": int(n_samples),
        "model_type": "IsolationForest",
        "contamination": float(contamination),
        "random_state": int(random_state),
        "dataset_type": str(dataset_type),
        "scaler": "StandardScaler",
    }

    payload = {
        "pipeline": pipeline,
        "metadata": metadata,
    }

    target_path = get_model_path(model_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write to temporary file then replace to prevent corrupting active artifact
    temp_file = tempfile.NamedTemporaryFile(
        dir=target_path.parent,
        prefix="model_tmp_",
        suffix=".joblib",
        delete=False,
    )
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        joblib.dump(payload, temp_path)
        os.replace(temp_path, target_path)
        logger.info("Atomically saved trained IsolationForest model to %s (samples: %d)", target_path, n_samples)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return metadata


def load_model(
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, MLModelStatus, str | None]:
    """
    Safely load a model bundle from disk.

    Trust Boundary:
    Deserialization relies on joblib/pickle. Model artifacts MUST only be loaded from
    trusted local paths configured by system administrators (BOTSHIELD_MODEL_PATH).
    Never deserialize arbitrary untrusted user uploads.

    Returns: (bundle, status, warning_or_error_message)
    Never crashes on missing, corrupt, or incompatible model artifacts.
    """
    path = get_model_path(model_path)
    if not path.exists():
        return (
            None,
            MLModelStatus.UNTRAINED,
            f"No trained model artifact found at '{path}'. Model status is UNTRAINED.",
        )

    try:
        payload = joblib.load(path)
    except Exception as exc:
        logger.warning("Failed to deserialize model at %s: %s", path, exc)
        return (
            None,
            MLModelStatus.CORRUPT,
            f"Corrupt or unreadable model file at '{path}': {exc}",
        )

    if not isinstance(payload, dict) or "pipeline" not in payload or "metadata" not in payload:
        return (
            None,
            MLModelStatus.CORRUPT,
            f"Model artifact at '{path}' is missing required bundle keys ('pipeline', 'metadata').",
        )

    meta = payload.get("metadata", {})
    schema_ver = meta.get("schema_version")
    if schema_ver != FEATURE_SCHEMA_VERSION:
        return (
            None,
            MLModelStatus.INCOMPATIBLE,
            f"Model schema version '{schema_ver}' is incompatible with current schema '{FEATURE_SCHEMA_VERSION}'.",
        )

    saved_features = meta.get("feature_names", [])
    if list(saved_features) != list(FEATURE_NAMES):
        return (
            None,
            MLModelStatus.INCOMPATIBLE,
            f"Model feature contract mismatch: saved features ({len(saved_features)}) do not match expected schema order.",
        )

    return payload, MLModelStatus.AVAILABLE, None


class IsolationForestModel:
    """Wrapper class for inference against a saved Isolation Forest model."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = get_model_path(model_path)
        self.bundle, self.status, self.init_warning = load_model(self.model_path)

    def predict(self, features: ExtractedFeatures) -> MLInferenceResult:
        """
        Evaluate an ExtractedFeatures object and return a structured MLInferenceResult.

        Guarantees deterministic output and safe cold-start handling.
        """
        warnings: list[str] = list(features.warnings)
        if self.init_warning:
            warnings.append(self.init_warning)

        # Handle untrained or unavailable model states gracefully
        if self.status != MLModelStatus.AVAILABLE or self.bundle is None:
            return MLInferenceResult(
                status=self.status,
                schema_version=features.schema_version,
                anomaly_score=None,
                is_anomaly=None,
                feature_vector=features.feature_vector,
                feature_summary=features.feature_dict,
                warnings=warnings,
                model_metadata={},
            )

        pipeline: Pipeline = self.bundle["pipeline"]
        metadata: dict[str, Any] = self.bundle.get("metadata", {})

        X = np.asarray([features.feature_vector], dtype=np.float64)

        try:
            # score_samples outputs negative anomaly scores in [-1.0, 0.0]
            # Higher opposite score = more anomalous
            raw_scores = pipeline.score_samples(X)
            raw_score = -float(raw_scores[0])

            # Clamp normalized anomaly score strictly to [0.0, 1.0]
            normalized_score = max(0.0, min(1.0, raw_score))
            if math.isnan(normalized_score) or math.isinf(normalized_score):
                normalized_score = 0.5
                warnings.append("WARNING_NON_FINITE_MODEL_SCORE")

            # Determine anomaly classification: forest predict returns -1 for outlier, 1 for inlier
            raw_preds = pipeline.predict(X)
            is_anomaly = bool(raw_preds[0] == -1)

            return MLInferenceResult(
                status=MLModelStatus.AVAILABLE,
                schema_version=features.schema_version,
                anomaly_score=round(normalized_score, 4),
                is_anomaly=is_anomaly,
                feature_vector=features.feature_vector,
                feature_summary=features.feature_dict,
                warnings=warnings,
                model_metadata=metadata,
            )
        except Exception as exc:
            logger.error("Error during ML model inference: %s", exc)
            warnings.append(f"ERROR_INFERENCE_FAILURE: {exc}")
            return MLInferenceResult(
                status=MLModelStatus.ERROR,
                schema_version=features.schema_version,
                anomaly_score=None,
                is_anomaly=None,
                feature_vector=features.feature_vector,
                feature_summary=features.feature_dict,
                warnings=warnings,
                model_metadata=metadata,
            )
