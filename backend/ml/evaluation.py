"""
BotShield – Machine Learning Evaluation & Detection Quality Metrics (Phase 8)

Provides a reproducible evaluation pipeline for the Isolation Forest behavioral detector:
- Controlled synthetic dataset generation covering diverse behavioral scenarios.
- Strict train / validation / test dataset partitioning with zero test leakage.
- Validation-based decision threshold selection.
- Comprehensive metric calculations: Confusion Matrix, Precision, Recall, F1, FPR, FNR, Accuracy.
- Explicit labeling of dataset provenance (synthetic benchmark vs real labeled traffic).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.ml.features import (
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
    train_unsupervised_model,
)

logger = logging.getLogger("botshield.ml.evaluation")


@dataclass(frozen=True)
class EvaluationMetrics:
    """Documented performance metrics computed over a labeled dataset."""
    total_samples: int
    positive_samples: int  # Actual anomalies/bots (label=1)
    negative_samples: int  # Actual normal/humans (label=0)
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    threshold: float
    dataset_type: str
    is_synthetic: bool
    schema_version: str = FEATURE_SCHEMA_VERSION
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "class_distribution": {
                "anomalies_positive": self.positive_samples,
                "normal_negative": self.negative_samples,
            },
            "confusion_matrix": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "tn": self.true_negatives,
                "fn": self.false_negatives,
            },
            "metrics": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "false_positive_rate": round(self.false_positive_rate, 4),
                "false_negative_rate": round(self.false_negative_rate, 4),
                "accuracy": round(self.accuracy, 4),
            },
            "threshold": round(self.threshold, 4),
            "dataset_type": self.dataset_type,
            "is_synthetic": self.is_synthetic,
            "schema_version": self.schema_version,
            "warnings": self.warnings,
        }


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    threshold: float = 0.5,
    dataset_type: str = "synthetic_scenario_evaluation",
    is_synthetic: bool = True,
) -> EvaluationMetrics:
    """
    Compute binary classification metrics given ground-truth labels and continuous anomaly scores.

    y_true: Binary labels (0 = normal/inlier, 1 = anomaly/bot).
    scores: Continuous anomaly scores in [0.0, 1.0] where higher = more anomalous.
    threshold: Decision threshold; sample is predicted positive if score >= threshold.
    """
    y_arr = np.asarray(y_true, dtype=np.int32)
    s_arr = np.asarray(scores, dtype=np.float64)

    if len(y_arr) != len(s_arr):
        raise ValueError(f"Length mismatch: y_true has {len(y_arr)}, scores has {len(s_arr)}")

    if len(y_arr) == 0:
        raise ValueError("Cannot compute metrics on an empty dataset.")

    warnings: list[str] = []

    # Handle any non-finite scores safely
    if np.isnan(s_arr).any() or np.isinf(s_arr).any():
        warnings.append("WARNING_NON_FINITE_SCORES_REPLACED_WITH_ZERO")
        s_arr = np.nan_to_num(s_arr, nan=0.0, posinf=1.0, neginf=0.0)

    y_pred = (s_arr >= float(threshold)).astype(np.int32)

    tp = int(np.sum((y_arr == 1) & (y_pred == 1)))
    fp = int(np.sum((y_arr == 0) & (y_pred == 1)))
    tn = int(np.sum((y_arr == 0) & (y_pred == 0)))
    fn = int(np.sum((y_arr == 1) & (y_pred == 0)))

    total = tp + fp + tn + fn
    positives = tp + fn
    negatives = tn + fp

    # Safe metric calculations preventing division by zero
    if (tp + fp) > 0:
        precision = float(tp) / float(tp + fp)
    else:
        precision = 0.0
        warnings.append("WARNING_UNDEFINED_PRECISION_NO_POSITIVE_PREDICTIONS")

    if positives > 0:
        recall = float(tp) / float(positives)
        fnr = float(fn) / float(positives)
    else:
        recall = 0.0
        fnr = 0.0
        warnings.append("WARNING_UNDEFINED_RECALL_NO_POSITIVE_SAMPLES")

    if (precision + recall) > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    if negatives > 0:
        fpr = float(fp) / float(negatives)
    else:
        fpr = 0.0
        warnings.append("WARNING_UNDEFINED_FPR_NO_NEGATIVE_SAMPLES")

    accuracy = float(tp + tn) / float(total) if total > 0 else 0.0

    return EvaluationMetrics(
        total_samples=total,
        positive_samples=positives,
        negative_samples=negatives,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        accuracy=accuracy,
        threshold=threshold,
        dataset_type=dataset_type,
        is_synthetic=is_synthetic,
        schema_version=FEATURE_SCHEMA_VERSION,
        warnings=warnings,
    )


def generate_synthetic_scenario_vector(
    scenario_type: str,
    rng: np.random.RandomState,
) -> list[float]:
    """
    Generate a deterministic 21-dimensional feature vector for a specific behavioral scenario.

    Scenarios:
    - human_browse (class 0): Typical human with high telemetry, reasonable pacing, varied paths.
    - human_casual (class 0): Short human visit, modest telemetry, zero rapid traversals.
    - rapid_scraper (class 1): Fast automated scraper, zero telemetry, high request rate.
    - cycling_crawler (class 1): Bot looping over product paths with high repetition and low telemetry.
    - credential_stuffing (class 1): Rapid burst to login endpoint with high error status ratio.
    """
    if scenario_type == "human_browse":
        reqs = float(rng.randint(4, 20))
        telems = float(rng.randint(20, 90))
        duration = float(rng.uniform(30.0, 360.0))
        req_rate = reqs / duration
        telem_rate = telems / duration
        telem_ratio = telems / max(reqs, 1.0)
        req_ratio = reqs / (reqs + telems)
        unique_paths = float(rng.randint(3, min(int(reqs) + 1, 10)))
        path_rep = max(0.0, 1.0 - (unique_paths / reqs))
        pv_count = float(rng.randint(2, min(int(reqs), 7)))
        clicks = float(rng.randint(3, 20))
        scrolls = float(rng.randint(5, 30))
        mouse = float(rng.randint(10, 50))
        keyboard = float(rng.randint(0, 15))
        page_exits = 1.0
        mean_dt = duration / (reqs + telems)
        std_dt = float(rng.uniform(1.0, 6.0))
        min_dt = float(rng.uniform(0.15, 1.5))
        rapid_count = 0.0
        error_ratio = float(rng.choice([0.0, 0.0, 0.05]))

    elif scenario_type == "human_casual":
        reqs = float(rng.randint(2, 6))
        telems = float(rng.randint(4, 16))
        duration = float(rng.uniform(15.0, 90.0))
        req_rate = reqs / duration
        telem_rate = telems / duration
        telem_ratio = telems / max(reqs, 1.0)
        req_ratio = reqs / (reqs + telems)
        unique_paths = float(rng.randint(2, int(reqs) + 1))
        path_rep = max(0.0, 1.0 - (unique_paths / reqs))
        pv_count = float(rng.randint(1, int(reqs)))
        clicks = float(rng.randint(1, 4))
        scrolls = float(rng.randint(1, 6))
        mouse = float(rng.randint(2, 10))
        keyboard = float(rng.randint(0, 3))
        page_exits = 1.0
        mean_dt = duration / (reqs + telems)
        std_dt = float(rng.uniform(1.0, 8.0))
        min_dt = float(rng.uniform(0.3, 3.0))
        rapid_count = 0.0
        error_ratio = 0.0

    elif scenario_type == "rapid_scraper":
        reqs = float(rng.randint(35, 150))
        telems = float(rng.choice([0, 0, 1]))  # Nearly zero telemetry
        duration = float(rng.uniform(2.0, 15.0))  # Extremely rapid burst
        req_rate = reqs / duration
        telem_rate = telems / duration
        telem_ratio = telems / max(reqs, 1.0)
        req_ratio = reqs / (reqs + telems)
        unique_paths = float(rng.randint(15, min(int(reqs), 45)))
        path_rep = max(0.0, 1.0 - (unique_paths / reqs))
        pv_count = 0.0
        clicks = 0.0
        scrolls = 0.0
        mouse = 0.0
        keyboard = 0.0
        page_exits = 0.0
        mean_dt = duration / reqs
        std_dt = float(rng.uniform(0.01, 0.08))  # Machine-like uniformity
        min_dt = float(rng.uniform(0.005, 0.04))
        rapid_count = float(rng.randint(8, min(int(reqs), 30)))
        error_ratio = float(rng.choice([0.0, 0.02, 0.05]))

    elif scenario_type == "cycling_crawler":
        reqs = float(rng.randint(25, 80))
        telems = float(rng.choice([0, 1]))
        duration = float(rng.uniform(5.0, 25.0))
        req_rate = reqs / duration
        telem_rate = telems / duration
        telem_ratio = telems / max(reqs, 1.0)
        req_ratio = reqs / (reqs + telems)
        unique_paths = float(rng.randint(2, 4))  # High loop cycling
        path_rep = 1.0 - (unique_paths / reqs)
        pv_count = 0.0
        clicks = 0.0
        scrolls = 0.0
        mouse = 0.0
        keyboard = 0.0
        page_exits = 0.0
        mean_dt = duration / reqs
        std_dt = float(rng.uniform(0.01, 0.05))
        min_dt = float(rng.uniform(0.01, 0.05))
        rapid_count = float(rng.randint(5, 20))
        error_ratio = 0.0

    elif scenario_type == "credential_stuffing":
        reqs = float(rng.randint(20, 60))
        telems = float(rng.choice([0, 1]))
        duration = float(rng.uniform(4.0, 18.0))
        req_rate = reqs / duration
        telem_rate = telems / duration
        telem_ratio = telems / max(reqs, 1.0)
        req_ratio = reqs / (reqs + telems)
        unique_paths = 1.0  # Only targeting /login
        path_rep = 1.0 - (unique_paths / reqs)
        pv_count = 0.0
        clicks = 0.0
        scrolls = 0.0
        mouse = 0.0
        keyboard = 0.0
        page_exits = 0.0
        mean_dt = duration / reqs
        std_dt = float(rng.uniform(0.02, 0.12))
        min_dt = float(rng.uniform(0.01, 0.05))
        rapid_count = 0.0
        error_ratio = float(rng.uniform(0.40, 0.85))  # High HTTP 401/403 failures

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")

    return [
        reqs,
        telems,
        reqs + telems,
        duration,
        req_rate,
        telem_rate,
        telem_ratio,
        req_ratio,
        unique_paths,
        path_rep,
        pv_count,
        clicks,
        scrolls,
        mouse,
        keyboard,
        page_exits,
        mean_dt,
        std_dt,
        min_dt,
        rapid_count,
        error_ratio,
    ]


def generate_evaluation_splits(
    n_train_normal: int = 120,
    n_val_normal: int = 40,
    n_val_anomalies: int = 40,
    n_test_normal: int = 50,
    n_test_anomalies: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate clean, separated synthetic datasets for training, validation, and held-out testing.

    Returns:
        (X_train, X_val, y_val, X_test, y_test)

    - X_train: Normal human sessions only (for unsupervised Isolation Forest training).
    - X_val, y_val: Mixed normal and bot sessions for validation threshold selection.
    - X_test, y_test: Held-out mixed test sessions strictly reserved for final reporting.
    """
    rng = np.random.RandomState(random_state)

    # 1. Training set (normal behavior only, unsupervised baseline)
    train_vectors = []
    for _ in range(n_train_normal):
        stype = "human_browse" if rng.rand() > 0.3 else "human_casual"
        train_vectors.append(generate_synthetic_scenario_vector(stype, rng))
    X_train = np.asarray(train_vectors, dtype=np.float64)

    # 2. Validation set (mixed for threshold optimization)
    val_vectors = []
    y_val_list = []
    for _ in range(n_val_normal):
        stype = "human_browse" if rng.rand() > 0.3 else "human_casual"
        val_vectors.append(generate_synthetic_scenario_vector(stype, rng))
        y_val_list.append(0)
    for _ in range(n_val_anomalies):
        stype = rng.choice(["rapid_scraper", "cycling_crawler", "credential_stuffing"])
        val_vectors.append(generate_synthetic_scenario_vector(stype, rng))
        y_val_list.append(1)

    X_val = np.asarray(val_vectors, dtype=np.float64)
    y_val = np.asarray(y_val_list, dtype=np.int32)

    # 3. Held-out test set (strictly separate, unseen seed progression)
    test_vectors = []
    y_test_list = []
    for _ in range(n_test_normal):
        stype = "human_browse" if rng.rand() > 0.3 else "human_casual"
        test_vectors.append(generate_synthetic_scenario_vector(stype, rng))
        y_test_list.append(0)
    for _ in range(n_test_anomalies):
        stype = rng.choice(["rapid_scraper", "cycling_crawler", "credential_stuffing"])
        test_vectors.append(generate_synthetic_scenario_vector(stype, rng))
        y_test_list.append(1)

    X_test = np.asarray(test_vectors, dtype=np.float64)
    y_test = np.asarray(y_test_list, dtype=np.int32)

    return X_train, X_val, y_val, X_test, y_test


def find_optimal_threshold(
    model: IsolationForestModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold_candidates: list[float] | None = None,
) -> float:
    """
    Select the decision threshold that maximizes F1 score on the validation split.

    Guarantees no test set leakage during threshold tuning.
    """
    candidates = threshold_candidates or [
        round(t, 2) for t in np.arange(0.30, 0.75, 0.02)
    ]

    # Predict scores for validation set
    val_scores = []
    for vec in X_val:
        feat = ExtractedFeatures(
            feature_vector=list(vec),
            feature_names=FEATURE_NAMES,
            feature_dict={FEATURE_NAMES[i]: float(vec[i]) for i in range(len(FEATURE_NAMES))},
        )
        res = model.predict(feat)
        score = res.anomaly_score if res.anomaly_score is not None else 0.5
        val_scores.append(score)

    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in candidates:
        metrics = compute_classification_metrics(y_val, val_scores, threshold=thresh)
        if metrics.f1_score > best_f1:
            best_f1 = metrics.f1_score
            best_thresh = thresh

    logger.info("Selected optimal validation threshold: %.2f (Validation F1: %.4f)", best_thresh, best_f1)
    return best_thresh


def evaluate_model_pipeline(
    model_path: str | Path | None = None,
    random_state: int = 42,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Run full reproducible evaluation of a trained model against held-out synthetic test data.

    If threshold is None, selects threshold via validation set and tests on held-out test set.
    """
    target_path = get_model_path(model_path)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model artifact file does not exist at '{target_path}'. "
            f"Train a model first using: py -m backend.ml.train --train --synthetic --out {target_path}"
        )

    model = IsolationForestModel(model_path=target_path)
    if model.status != MLModelStatus.AVAILABLE:
        raise RuntimeError(
            f"Cannot evaluate model: status is '{model.status.value}' at '{target_path}'. "
            f"Reason: {model.init_warning}"
        )

    model_metadata = model.bundle.get("metadata", {}) if (model.bundle and isinstance(model.bundle, dict)) else {}

    # Generate isolated splits
    X_train, X_val, y_val, X_test, y_test = generate_evaluation_splits(
        random_state=random_state
    )

    # Threshold selection on validation set if not explicitly passed
    if threshold is None:
        eval_thresh = find_optimal_threshold(model, X_val, y_val)
    else:
        eval_thresh = float(threshold)

    # Evaluate strictly on held-out test set
    test_scores = []
    for vec in X_test:
        feat = ExtractedFeatures(
            feature_vector=list(vec),
            feature_names=FEATURE_NAMES,
            feature_dict={FEATURE_NAMES[i]: float(vec[i]) for i in range(len(FEATURE_NAMES))},
        )
        res = model.predict(feat)
        score = res.anomaly_score if res.anomaly_score is not None else 0.5
        test_scores.append(score)

    metrics = compute_classification_metrics(
        y_true=y_test,
        scores=test_scores,
        threshold=eval_thresh,
        dataset_type="synthetic_scenario_evaluation",
        is_synthetic=True,
    )

    report = {
        "evaluation_name": "Synthetic Behavioral Scenario Benchmark",
        "real_world_accuracy_claim": False,
        "disclaimer": (
            "NOTICE: This evaluation was conducted on controlled synthetic behavioral scenarios. "
            "It validates pipeline correctness, score semantics, and separability under synthetic "
            "distributions, but DOES NOT represent validated real-world bot detection accuracy."
        ),
        "model_artifact": str(target_path),
        "model_metadata": model_metadata,
        "random_state": random_state,
        "splits": {
            "validation_samples": len(y_val),
            "test_samples": len(y_test),
        },
        "operating_threshold": eval_thresh,
        "results": metrics.to_dict(),
    }

    return report
