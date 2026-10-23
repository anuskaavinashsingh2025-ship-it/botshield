"""
BotShield – Machine Learning Detection Package (Phase 4)

Provides reproducible behavioral feature extraction, unsupervised anomaly scoring
via Isolation Forest, model persistence, and evaluation service integration.
"""

from backend.ml.features import (
    FEATURE_CONTRACT,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    ExtractedFeatures,
    extract_session_features,
)
from backend.ml.model import (
    IsolationForestModel,
    MLInferenceResult,
    MLModelStatus,
    get_model_path,
    load_model,
    train_unsupervised_model,
)
from backend.ml.evaluation import (
    EvaluationMetrics,
    compute_classification_metrics,
    evaluate_model_pipeline,
    find_optimal_threshold,
    generate_evaluation_splits,
    generate_synthetic_scenario_vector,
)
from backend.ml.service import evaluate_session_ml

__all__ = [
    "FEATURE_CONTRACT",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "ExtractedFeatures",
    "extract_session_features",
    "IsolationForestModel",
    "MLInferenceResult",
    "MLModelStatus",
    "get_model_path",
    "load_model",
    "train_unsupervised_model",
    "evaluate_session_ml",
    "EvaluationMetrics",
    "compute_classification_metrics",
    "evaluate_model_pipeline",
    "find_optimal_threshold",
    "generate_evaluation_splits",
    "generate_synthetic_scenario_vector",
]

