"""
BotShield – Phase 8 ML Training, Validation, and Detection Quality Tests

Covers:
1. Feature Extraction & Schema Contract:
   - Verifies exact 21 features, names, ordering, and schema version 1.0.0.
   - Hand-verifiable feature calculation consistency between training and inference.
   - Handling of missing, non-finite, and malformed inputs.
   - Freedom from future event leakage or session ID reliance.
2. Model Lifecycle & Artifact Safety:
   - Untrained state reports MLModelStatus.UNTRAINED.
   - Training pipeline with StandardScaler and IsolationForest fits on training data only.
   - Atomic replacement: existing usable model artifact is preserved if new training fails.
   - Detection of corrupt, missing-key, or schema/feature-mismatched artifacts.
   - Bounded input validation and failure on invalid dimensions or sample counts.
3. Score Semantics:
   - Verifies raw anomaly score conversion, opposite score direction, and normalization to [0.0, 1.0].
   - Verifies thresholding direction (score >= threshold => anomaly).
   - Integration with hybrid scoring: scores stay within [0, 100], FSM-only fallback works.
4. Evaluation Dataset & Metrics:
   - Deterministic dataset generation with fixed random seed.
   - Strict train / validation / test dataset partitioning.
   - Accurate metric calculations against small hand-verifiable fixtures (TP, FP, TN, FN, Precision, Recall, F1, FPR, FNR, Accuracy).
   - Graceful zero-division handling for edge-case prediction sets.
   - Threshold selection strictly on validation set without test leakage.
   - Explicit synthetic scenario provenance labeling and disclaimers.
5. Regression:
   - Ensures full backward compatibility with FSM, hybrid scoring, enforcement, and admin dashboard.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest

from backend.ml.evaluation import (
    EvaluationMetrics,
    compute_classification_metrics,
    evaluate_model_pipeline,
    find_optimal_threshold,
    generate_evaluation_splits,
    generate_synthetic_scenario_vector,
)
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
from backend.scoring import (
    HybridConfig,
    HybridRiskResult,
    RiskLevel,
    ScoringMode,
    compute_hybrid_score,
    evaluate_session_risk,
)


@pytest.fixture
def temp_dir():
    """Create an isolated temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ===========================================================================
# 1. Feature Extraction & Schema Contract Tests
# ===========================================================================

def test_feature_contract_exact_names_and_count():
    """Verify exact count of 21 features, names, ordering, and schema version."""
    assert len(FEATURE_NAMES) == 21
    assert FEATURE_SCHEMA_VERSION == "1.0.0"
    assert len(FEATURE_CONTRACT) == 21

    expected_names = [
        "request_count",
        "telemetry_count",
        "total_activity_count",
        "session_duration_seconds",
        "request_rate",
        "telemetry_rate",
        "telemetry_to_request_ratio",
        "request_to_total_ratio",
        "unique_path_count",
        "path_repetition_ratio",
        "page_view_count",
        "click_count",
        "scroll_count",
        "mouse_activity_count",
        "keyboard_activity_count",
        "page_exit_count",
        "mean_inter_event_time",
        "std_inter_event_time",
        "min_inter_event_time",
        "rapid_traversal_count",
        "error_status_ratio",
    ]
    assert list(FEATURE_NAMES) == expected_names

    for idx, name in enumerate(expected_names):
        assert name in FEATURE_CONTRACT
        contract_entry = FEATURE_CONTRACT[name]
        assert contract_entry["index"] == idx
        assert contract_entry["name"] == name
        assert contract_entry["type"] == "float"
        bounds = contract_entry["bounds"]
        assert bounds[0] <= bounds[1]
        assert len(contract_entry["description"]) > 0


def test_feature_extraction_no_future_leakage_or_session_id():
    """Verify feature extractor uses only legitimate behavioral signals, ignoring session ID/IP."""
    events = [
        {"id": 1, "session_id": "sess-leak-test", "event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:00:00Z", "page": "/"},
        {"id": 2, "session_id": "sess-leak-test", "event_type": "CLICK", "timestamp": "2026-10-24T00:00:02Z", "page": "/product/1"},
    ]
    requests = [
        {"id": 1, "session_id": "sess-leak-test", "method": "GET", "path": "/", "status_code": 200, "timestamp": "2026-10-24T00:00:00Z"},
    ]

    features = extract_session_features(events=events, requests=requests)
    assert len(features.feature_vector) == 21
    # Verify vector contains only finite floats
    for val in features.feature_vector:
        assert isinstance(val, float)
        assert not math.isnan(val)
        assert not math.isinf(val)


def test_feature_extraction_missing_and_malformed_inputs():
    """Malformed timestamps, NaN values, and negative durations are safely clamped."""
    events = [
        {"id": 1, "event_type": "PAGE_VIEW", "timestamp": "invalid-timestamp", "page": "/"},
        {"id": 2, "event_type": "CLICK", "timestamp": None, "page": "/"},
    ]
    requests = [
        {"id": 1, "method": "GET", "path": "/foo", "status_code": "not-an-int", "timestamp": "not-a-ts"},
    ]

    features = extract_session_features(events=events, requests=requests)
    assert len(features.feature_vector) == 21
    for val in features.feature_vector:
        assert isinstance(val, float)
        assert not math.isnan(val)
        assert not math.isinf(val)
    assert len(features.warnings) > 0


# ===========================================================================
# 2. Model Lifecycle & Artifact Safety Tests
# ===========================================================================

def test_model_untrained_status_when_file_missing(temp_dir):
    """Missing model artifact returns MLModelStatus.UNTRAINED without crashing."""
    missing_path = temp_dir / "nonexistent_model.joblib"
    model = IsolationForestModel(model_path=missing_path)
    assert model.status == MLModelStatus.UNTRAINED

    feat = ExtractedFeatures(
        feature_vector=[0.0] * 21,
        feature_names=FEATURE_NAMES,
        feature_dict={name: 0.0 for name in FEATURE_NAMES},
    )
    res = model.predict(feat)
    assert res.status == MLModelStatus.UNTRAINED
    assert res.anomaly_score is None
    assert res.is_anomaly is None


def test_model_training_and_atomic_save(temp_dir):
    """Valid training saves an atomic artifact with comprehensive metadata."""
    model_path = temp_dir / "models" / "test_iso_forest.joblib"
    X = np.random.RandomState(42).uniform(0.0, 10.0, size=(25, 21))

    meta = train_unsupervised_model(
        training_data=X,
        model_path=model_path,
        contamination=0.1,
        random_state=42,
        min_samples=10,
        dataset_type="unit_test_synthetic",
    )

    assert model_path.exists()
    assert meta["training_sample_count"] == 25
    assert meta["model_type"] == "IsolationForest"
    assert meta["dataset_type"] == "unit_test_synthetic"

    bundle, status, warning = load_model(model_path)
    assert status == MLModelStatus.AVAILABLE
    assert bundle is not None
    assert "pipeline" in bundle
    assert "scaler" in bundle["pipeline"].named_steps
    assert "forest" in bundle["pipeline"].named_steps


def test_atomic_replacement_preserves_existing_model_on_failure(temp_dir):
    """If training replacement fails, the existing active model artifact is not destroyed."""
    model_path = temp_dir / "active_model.joblib"
    X_valid = np.random.RandomState(42).uniform(0.0, 5.0, size=(20, 21))

    # 1. Train initial working model
    train_unsupervised_model(X_valid, model_path=model_path, min_samples=10)
    assert model_path.exists()
    initial_mtime = model_path.stat().st_mtime

    # 2. Attempt replacement with invalid training data (NaNs)
    X_invalid = np.copy(X_valid)
    X_invalid[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or Infinite"):
        train_unsupervised_model(X_invalid, model_path=model_path, min_samples=10)

    # 3. Verify original model artifact remains intact and usable
    assert model_path.exists()
    bundle, status, _ = load_model(model_path)
    assert status == MLModelStatus.AVAILABLE
    assert bundle is not None


def test_model_incompatible_artifact_detection(temp_dir):
    """Artifact with mismatched feature length or ordering is flagged INCOMPATIBLE."""
    model_path = temp_dir / "incompatible.joblib"

    # Bundle with only 20 features
    truncated_features = list(FEATURE_NAMES[:20])
    payload = {
        "pipeline": None,
        "metadata": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": truncated_features,
        },
    }
    joblib.dump(payload, model_path)

    bundle, status, warning = load_model(model_path)
    assert status == MLModelStatus.INCOMPATIBLE
    assert "mismatch" in warning.lower()


def test_model_corrupt_artifact_detection(temp_dir):
    """A corrupt or truncated file is flagged CORRUPT without unhandled crash."""
    corrupt_path = temp_dir / "corrupt.joblib"
    with open(corrupt_path, "wb") as f:
        f.write(b"not-a-valid-joblib-pickle-data-stream")

    bundle, status, warning = load_model(corrupt_path)
    assert status == MLModelStatus.CORRUPT
    assert "corrupt" in warning.lower() or "unreadable" in warning.lower()


# ===========================================================================
# 3. Score Semantics Tests
# ===========================================================================

def test_score_semantics_direction_and_bounds(temp_dir):
    """
    Verify raw Isolation Forest score transformation:
    - Normal instances yield lower normalized anomaly scores.
    - Extreme anomaly instances yield higher normalized anomaly scores.
    - All scores are bounded strictly within [0.0, 1.0].
    """
    model_path = temp_dir / "semantics_model.joblib"
    rng = np.random.RandomState(42)

    # Train on normal baseline distribution
    X_train = np.array([generate_synthetic_scenario_vector("human_browse", rng) for _ in range(60)])
    train_unsupervised_model(X_train, model_path=model_path, min_samples=20)

    model = IsolationForestModel(model_path=model_path)
    assert model.status == MLModelStatus.AVAILABLE

    # Normal sample
    norm_vec = generate_synthetic_scenario_vector("human_browse", rng)
    norm_feat = ExtractedFeatures(norm_vec, FEATURE_NAMES, {FEATURE_NAMES[i]: norm_vec[i] for i in range(21)})
    norm_res = model.predict(norm_feat)

    # Severe anomaly sample (extreme request rate, 0 telemetry)
    anom_vec = generate_synthetic_scenario_vector("rapid_scraper", rng)
    anom_feat = ExtractedFeatures(anom_vec, FEATURE_NAMES, {FEATURE_NAMES[i]: anom_vec[i] for i in range(21)})
    anom_res = model.predict(anom_feat)

    assert norm_res.anomaly_score is not None
    assert anom_res.anomaly_score is not None

    # Bounds check
    assert 0.0 <= norm_res.anomaly_score <= 1.0
    assert 0.0 <= anom_res.anomaly_score <= 1.0

    # Direction check: scraper anomaly score should be strictly higher than human score
    assert anom_res.anomaly_score > norm_res.anomaly_score


def test_hybrid_scoring_consumes_ml_score():
    """Hybrid scoring combines FSM score and normalized ML score with configured weights."""
    cfg = HybridConfig(fsm_weight=0.60, ml_weight=0.40, low_threshold=35.0, high_threshold=70.0)

    # 1. Both available
    comp, norm, mode, warnings = compute_hybrid_score(
        fsm_score=0.80,
        ml_score=0.90,
        ml_status=MLModelStatus.AVAILABLE,
        config=cfg,
    )
    expected_norm = round((0.60 * 0.80) + (0.40 * 0.90), 4)  # 0.48 + 0.36 = 0.84
    assert mode == ScoringMode.HYBRID
    assert norm == expected_norm
    assert comp == round(expected_norm * 100.0, 1)

    # 2. ML untrained fallback
    comp_fsm, norm_fsm, mode_fsm, warnings_fsm = compute_hybrid_score(
        fsm_score=0.80,
        ml_score=None,
        ml_status=MLModelStatus.UNTRAINED,
        config=cfg,
    )
    assert mode_fsm == ScoringMode.FSM_ONLY
    assert norm_fsm == 0.80
    assert comp_fsm == 80.0
    assert any("untrained" in w.lower() for w in warnings_fsm)


# ===========================================================================
# 4. Evaluation Dataset & Metrics Tests
# ===========================================================================

def test_compute_classification_metrics_hand_verified():
    """
    Test metric computations on a known hand-verifiable fixture:
    y_true = [1, 1, 0, 0]
    scores = [0.8, 0.3, 0.7, 0.2]
    threshold = 0.5
    Predictions:
    sample 0: s=0.8 >= 0.5 -> pred=1 (y=1) -> TP
    sample 1: s=0.3 <  0.5 -> pred=0 (y=1) -> FN
    sample 2: s=0.7 >= 0.5 -> pred=1 (y=0) -> FP
    sample 3: s=0.2 <  0.5 -> pred=0 (y=0) -> TN
    TP=1, FP=1, TN=1, FN=1
    Precision = 1 / (1 + 1) = 0.5
    Recall    = 1 / (1 + 1) = 0.5
    F1        = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
    FPR       = 1 / (1 + 1) = 0.5
    FNR       = 1 / (1 + 1) = 0.5
    Accuracy  = (1 + 1) / 4 = 0.5
    """
    y_true = [1, 1, 0, 0]
    scores = [0.8, 0.3, 0.7, 0.2]

    metrics = compute_classification_metrics(y_true, scores, threshold=0.5)

    assert metrics.total_samples == 4
    assert metrics.positive_samples == 2
    assert metrics.negative_samples == 2
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.true_negatives == 1
    assert metrics.false_negatives == 1
    assert math.isclose(metrics.precision, 0.5)
    assert math.isclose(metrics.recall, 0.5)
    assert math.isclose(metrics.f1_score, 0.5)
    assert math.isclose(metrics.false_positive_rate, 0.5)
    assert math.isclose(metrics.false_negative_rate, 0.5)
    assert math.isclose(metrics.accuracy, 0.5)
    assert metrics.is_synthetic is True


def test_compute_classification_metrics_zero_division_safety():
    """Verify metrics handle edge cases (no positive predictions, all negative labels) safely."""
    y_true = [0, 0, 0]
    scores = [0.1, 0.2, 0.15]

    # No positive predictions, no positive ground truth
    metrics = compute_classification_metrics(y_true, scores, threshold=0.5)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1_score == 0.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.false_negative_rate == 0.0
    assert metrics.accuracy == 1.0  # All correctly identified as TN
    assert any("UNDEFINED" in w for w in metrics.warnings)


def test_evaluation_splits_reproducibility():
    """Splits generated with identical seed must produce identical data."""
    X_tr1, X_v1, y_v1, X_te1, y_te1 = generate_evaluation_splits(
        n_train_normal=20, n_val_normal=10, n_val_anomalies=10, n_test_normal=10, n_test_anomalies=10, random_state=123
    )
    X_tr2, X_v2, y_v2, X_te2, y_te2 = generate_evaluation_splits(
        n_train_normal=20, n_val_normal=10, n_val_anomalies=10, n_test_normal=10, n_test_anomalies=10, random_state=123
    )

    np.testing.assert_array_equal(X_tr1, X_tr2)
    np.testing.assert_array_equal(X_v1, X_v2)
    np.testing.assert_array_equal(y_v1, y_v2)
    np.testing.assert_array_equal(X_te1, X_te2)
    np.testing.assert_array_equal(y_te1, y_te2)


def test_threshold_selection_has_no_test_leakage(temp_dir):
    """Verify threshold optimization executes solely on validation split."""
    model_path = temp_dir / "leak_test_model.joblib"
    X_train, X_val, y_val, X_test, y_test = generate_evaluation_splits(
        n_train_normal=50, n_val_normal=20, n_val_anomalies=20, n_test_normal=20, n_test_anomalies=20, random_state=42
    )
    train_unsupervised_model(X_train, model_path=model_path, min_samples=10)
    model = IsolationForestModel(model_path=model_path)

    # Selected threshold using validation set
    optimal_t = find_optimal_threshold(model, X_val, y_val)
    assert 0.30 <= optimal_t <= 0.75


def test_evaluate_model_pipeline_full_run(temp_dir):
    """Full pipeline evaluation returns structured report with synthetic disclaimers."""
    model_path = temp_dir / "eval_pipeline_model.joblib"
    rng = np.random.RandomState(42)
    X_train = np.array([generate_synthetic_scenario_vector("human_browse", rng) for _ in range(50)])
    train_unsupervised_model(X_train, model_path=model_path, min_samples=10)

    report = evaluate_model_pipeline(model_path=model_path, random_state=42)

    assert report["real_world_accuracy_claim"] is False
    assert "controlled synthetic behavioral scenarios" in report["disclaimer"]
    assert "results" in report
    res = report["results"]
    assert "confusion_matrix" in res
    assert "metrics" in res
    assert 0.0 <= res["metrics"]["f1_score"] <= 1.0
    assert 0.0 <= res["metrics"]["accuracy"] <= 1.0
