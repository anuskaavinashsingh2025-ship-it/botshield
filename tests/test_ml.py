"""
BotShield – Machine Learning Detection & Feature Extraction Tests (Phase 4)

Covers:
- Feature vector length, schema version, and ordering.
- Deterministic repeatability.
- Empty sessions and partial telemetry.
- Malformed timestamps and metadata.
- Duplicate event and request handling.
- Extreme inputs, clamping, and boundary conditions.
- Training validation (sample counts, feature dimensionality, non-finite values).
- Untrained model cold-start handling (score is None, not 0.0).
- Model persistence, loading, corrupt file handling, and schema compatibility checks.
- Score normalization bounds ([0.0, 1.0]).
- Service integration with isolated SQLite databases.
- Preservation of existing FSM scores, risk scores, and reasons during ML updates.
- Privacy verification (no sensitive credentials or PII leaked).
"""

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.database.db import (
    get_connection,
    get_session_risk,
    init_db,
    insert_events,
    insert_request,
    update_session_ml_risk,
    upsert_session,
    upsert_session_risk,
)
from backend.ml import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    ExtractedFeatures,
    IsolationForestModel,
    MLInferenceResult,
    MLModelStatus,
    evaluate_session_ml,
    extract_session_features,
    load_model,
    train_unsupervised_model,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with all tables initialized."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def temp_model_path():
    """Provide a temporary file path for model serialization."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        model_path = f.name
    yield model_path
    try:
        Path(model_path).unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Feature Extraction Tests
# ---------------------------------------------------------------------------

def test_feature_ordering_and_vector_length():
    """Ensure feature vector length is 21 and ordering matches FEATURE_NAMES exactly."""
    assert len(FEATURE_NAMES) == 21
    features = extract_session_features([], [])
    assert len(features.feature_vector) == 21
    assert features.feature_names == FEATURE_NAMES
    assert list(features.feature_dict.keys()) == list(FEATURE_NAMES)
    assert features.schema_version == FEATURE_SCHEMA_VERSION


def test_deterministic_feature_extraction():
    """Running extraction multiple times on identical input yields identical output."""
    requests = [
        {"id": 1, "method": "GET", "path": "/", "status_code": 200, "timestamp": "2026-10-23T10:00:00Z"},
        {"id": 2, "method": "GET", "path": "/products", "status_code": 200, "timestamp": "2026-10-23T10:00:05Z"},
    ]
    events = [
        {"id": 1, "event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-10-23T10:00:00.100Z"},
        {"id": 2, "event_type": "CLICK", "page": "/", "timestamp": "2026-10-23T10:00:02Z"},
        {"id": 3, "event_type": "SCROLL", "page": "/", "timestamp": "2026-10-23T10:00:03Z"},
    ]
    feat1 = extract_session_features(events=events, requests=requests)
    feat2 = extract_session_features(events=events, requests=requests)

    assert feat1.feature_vector == feat2.feature_vector
    assert feat1.feature_dict == feat2.feature_dict
    assert feat1.warnings == feat2.warnings


def test_empty_session_and_incomplete_telemetry():
    """Empty sessions produce finite zero features with empty session warning."""
    feat = extract_session_features([], [])
    assert len(feat.feature_vector) == 21
    for val in feat.feature_vector:
        assert isinstance(val, float)
        assert not math.isnan(val)
        assert not math.isinf(val)
        assert val == 0.0
    assert "WARNING_EMPTY_SESSION" in feat.warnings

    # Incomplete telemetry: requests exist but zero browser telemetry
    requests = [
        {"id": 1, "method": "GET", "path": "/", "status_code": 200, "timestamp": "2026-10-23T10:00:00Z"},
        {"id": 2, "method": "GET", "path": "/about", "status_code": 200, "timestamp": "2026-10-23T10:00:04Z"},
    ]
    feat_req_only = extract_session_features(events=[], requests=requests)
    assert feat_req_only.feature_dict["request_count"] == 2.0
    assert feat_req_only.feature_dict["telemetry_count"] == 0.0
    assert feat_req_only.feature_dict["telemetry_to_request_ratio"] == 0.0
    assert feat_req_only.feature_dict["session_duration_seconds"] == 4.0


def test_malformed_metadata_and_timestamps():
    """Corrupt timestamps and invalid data types are handled safely without unhandled exceptions."""
    requests = [
        {"id": 1, "method": "GET", "path": "/", "status_code": "NOT_AN_INT", "timestamp": "invalid-timestamp"},
        {"id": 2, "method": "GET", "path": "/products?param=1", "status_code": 500, "timestamp": None},
    ]
    events = [
        {"id": 1, "event_type": None, "page": None, "timestamp": "not-a-date"},
    ]
    feat = extract_session_features(events=events, requests=requests)
    assert len(feat.feature_vector) == 21
    for val in feat.feature_vector:
        assert not math.isnan(val)
        assert not math.isinf(val)
    assert any("MALFORMED" in w for w in feat.warnings)


def test_duplicate_events_and_requests():
    """Exact duplicate HTTP requests and telemetry events are deduplicated."""
    requests = [
        {"id": 1, "method": "GET", "path": "/products", "status_code": 200, "timestamp": "2026-10-23T10:00:00Z"},
        {"id": 2, "method": "GET", "path": "/products", "status_code": 200, "timestamp": "2026-10-23T10:00:00Z"},
    ]
    events = [
        {"id": 1, "event_type": "CLICK", "page": "/products", "timestamp": "2026-10-23T10:00:01Z"},
        {"id": 2, "event_type": "CLICK", "page": "/products", "timestamp": "2026-10-23T10:00:01Z"},
    ]
    feat = extract_session_features(events=events, requests=requests)
    assert feat.feature_dict["request_count"] == 1.0
    assert feat.feature_dict["click_count"] == 1.0
    assert any("DUPLICATE" in w for w in feat.warnings)


def test_large_counts_and_extreme_intervals():
    """Adversarial/extreme counts and timestamps are clamped to valid bounds without overflow."""
    requests = [
        {"id": 1, "method": "GET", "path": "/", "status_code": 200, "timestamp": "1970-01-01T00:00:00Z"},
        {"id": 2, "method": "GET", "path": "/products", "status_code": 200, "timestamp": "2050-01-01T00:00:00Z"},
    ]
    feat = extract_session_features(events=[], requests=requests)
    assert feat.feature_dict["session_duration_seconds"] <= 2_592_000.0  # Clamped to 30 days
    assert feat.feature_dict["request_rate"] <= 10_000.0
    for val in feat.feature_vector:
        assert not math.isnan(val)
        assert not math.isinf(val)


# ---------------------------------------------------------------------------
# 2. Model Training & Persistence Tests
# ---------------------------------------------------------------------------

def test_valid_and_invalid_training_data(temp_model_path):
    """Test validation of training data inputs."""
    # 1. Invalid: None input
    with pytest.raises(ValueError, match="cannot be None"):
        train_unsupervised_model(None)

    # 2. Invalid: 1D array
    with pytest.raises(ValueError, match="must be a 2D array"):
        train_unsupervised_model([1.0, 2.0, 3.0])

    # 3. Invalid: Wrong feature count
    with pytest.raises(ValueError, match="Feature count mismatch"):
        train_unsupervised_model([[0.0] * 10 for _ in range(15)])

    # 4. Invalid: Insufficient samples
    with pytest.raises(ValueError, match="Insufficient training samples"):
        train_unsupervised_model([[0.0] * 21 for _ in range(4)], min_samples=10)

    # 5. Invalid: NaN in training data
    bad_data = [[0.0] * 21 for _ in range(15)]
    bad_data[0][0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Infinite"):
        train_unsupervised_model(bad_data)

    # 6. Valid training data
    valid_data = np.random.RandomState(42).randn(20, 21).tolist()
    meta = train_unsupervised_model(valid_data, model_path=temp_model_path, min_samples=10)
    assert meta["model_type"] == "IsolationForest"
    assert meta["training_sample_count"] == 20
    assert meta["schema_version"] == FEATURE_SCHEMA_VERSION
    assert Path(temp_model_path).exists()


def test_untrained_model_behavior():
    """When model file is missing, model returns UNTRAINED status and None score."""
    non_existent_path = Path("non_existent_model_12345.joblib")
    model = IsolationForestModel(model_path=non_existent_path)
    assert model.status == MLModelStatus.UNTRAINED

    features = extract_session_features([], [])
    result = model.predict(features)
    assert result.status == MLModelStatus.UNTRAINED
    assert result.anomaly_score is None
    assert result.is_anomaly is None
    assert any("UNTRAINED" in w for w in result.warnings)


def test_model_saving_and_loading(temp_model_path):
    """Model bundle saves and reloads cleanly with matching schema."""
    valid_data = np.random.RandomState(42).randn(15, 21).tolist()
    train_unsupervised_model(valid_data, model_path=temp_model_path, min_samples=10)

    bundle, status, warning = load_model(temp_model_path)
    assert status == MLModelStatus.AVAILABLE
    assert warning is None
    assert bundle is not None
    assert bundle["metadata"]["schema_version"] == FEATURE_SCHEMA_VERSION


def test_missing_corrupt_and_incompatible_model_files(temp_model_path):
    """Corrupt or incompatible files are caught without crashing."""
    # 1. Corrupt bytes
    with open(temp_model_path, "wb") as f:
        f.write(b"not a valid joblib or pickle file")
    _, status, warn = load_model(temp_model_path)
    assert status == MLModelStatus.CORRUPT
    assert "Corrupt" in warn

    # 2. Incompatible schema version
    import joblib
    fake_payload = {
        "pipeline": None,
        "metadata": {
            "schema_version": "99.0.0",
            "feature_names": list(FEATURE_NAMES),
        },
    }
    joblib.dump(fake_payload, temp_model_path)
    _, status, warn = load_model(temp_model_path)
    assert status == MLModelStatus.INCOMPATIBLE
    assert "incompatible" in warn

    # 3. Wrong feature length
    fake_payload_len = {
        "pipeline": None,
        "metadata": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": ["feat1", "feat2"],
        },
    }
    joblib.dump(fake_payload_len, temp_model_path)
    _, status, warn = load_model(temp_model_path)
    assert status == MLModelStatus.INCOMPATIBLE


def test_score_range_and_result_structure(temp_model_path):
    """Trained IsolationForest produces bounded anomaly scores in [0.0, 1.0]."""
    # Create synthetic training set of normal-looking sessions
    rng = np.random.RandomState(42)
    training_data = rng.uniform(0.1, 5.0, size=(30, 21)).tolist()
    train_unsupervised_model(training_data, model_path=temp_model_path, min_samples=10)

    model = IsolationForestModel(model_path=temp_model_path)
    assert model.status == MLModelStatus.AVAILABLE

    # Normal session
    features = extract_session_features(
        events=[{"id": 1, "event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-10-23T10:00:00Z"}],
        requests=[{"id": 1, "method": "GET", "path": "/", "status_code": 200, "timestamp": "2026-10-23T10:00:00Z"}],
    )
    result = model.predict(features)

    assert result.status == MLModelStatus.AVAILABLE
    assert isinstance(result.anomaly_score, float)
    assert 0.0 <= result.anomaly_score <= 1.0
    assert isinstance(result.is_anomaly, bool)
    assert result.schema_version == FEATURE_SCHEMA_VERSION
    assert len(result.feature_vector) == 21

    d = result.to_dict()
    assert d["status"] == "available"
    assert "anomaly_score" in d
    assert "feature_summary" in d


# ---------------------------------------------------------------------------
# 3. Database & Service Integration Tests
# ---------------------------------------------------------------------------

def test_service_evaluates_session_and_persists_ml_score(temp_db, temp_model_path):
    """Service loads DB rows, executes ML inference, and optionally persists ml_score."""
    # Train model
    rng = np.random.RandomState(42)
    train_unsupervised_model(rng.uniform(0.1, 2.0, size=(20, 21)).tolist(), model_path=temp_model_path)

    conn = get_connection(temp_db)
    session_id = "test-ml-session-001"
    upsert_session(conn, session_id)
    insert_request(conn, session_id, "GET", "/", 200, "2026-10-23T10:00:00Z")
    insert_request(conn, session_id, "GET", "/products", 200, "2026-10-23T10:00:05Z")
    insert_events(conn, session_id, [
        {"event_type": "PAGE_VIEW", "timestamp": "2026-10-23T10:00:00.100Z", "page": "/"},
        {"event_type": "CLICK", "timestamp": "2026-10-23T10:00:02Z", "page": "/"},
    ])
    conn.commit()
    conn.close()

    result = evaluate_session_ml(
        session_id=session_id,
        db_path=temp_db,
        model_path=temp_model_path,
        persist=True,
    )

    assert result.status == MLModelStatus.AVAILABLE
    assert result.anomaly_score is not None

    conn = get_connection(temp_db)
    row = get_session_risk(conn, session_id)
    conn.close()

    assert row is not None
    assert row["ml_score"] == result.anomaly_score
    assert row["fsm_score"] == 0.0
    assert row["risk_score"] == 0.0


def test_service_preserves_existing_fsm_score_and_reasons(temp_db, temp_model_path):
    """ML evaluation and persistence strictly preserves existing FSM score, risk score, and reasons."""
    rng = np.random.RandomState(42)
    train_unsupervised_model(rng.uniform(0.1, 2.0, size=(20, 21)).tolist(), model_path=temp_model_path)

    conn = get_connection(temp_db)
    session_id = "test-ml-preserve-002"
    upsert_session(conn, session_id)
    insert_request(conn, session_id, "GET", "/", 200, "2026-10-23T10:00:00Z")

    # Seed existing FSM risk evaluation
    upsert_session_risk(
        conn,
        session_id=session_id,
        fsm_score=0.85,
        risk_score=0.85,
        reasons=["ANOMALY_RAPID_TRAVERSAL: Fast navigation"],
    )
    conn.commit()
    conn.close()

    # Now run ML evaluation with persist=True
    result = evaluate_session_ml(
        session_id=session_id,
        db_path=temp_db,
        model_path=temp_model_path,
        persist=True,
    )
    assert result.status == MLModelStatus.AVAILABLE

    conn = get_connection(temp_db)
    row = get_session_risk(conn, session_id)
    conn.close()

    assert row is not None
    # FSM fields must be preserved exactly
    assert row["fsm_score"] == 0.85
    assert row["risk_score"] == 0.85
    reasons = json.loads(row["reasons"])
    assert reasons == ["ANOMALY_RAPID_TRAVERSAL: Fast navigation"]
    # Only ml_score should be updated
    assert row["ml_score"] == result.anomaly_score


def test_update_session_ml_risk_validation(temp_db):
    """update_session_ml_risk validates inputs properly."""
    conn = get_connection(temp_db)
    with pytest.raises(ValueError, match="session_id"):
        update_session_ml_risk(conn, "", 0.5)
    with pytest.raises(ValueError, match="ml_score"):
        update_session_ml_risk(conn, "sess-1", "bad_score")  # type: ignore
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        update_session_ml_risk(conn, "sess-1", 1.5)
    conn.close()


def test_no_leakage_of_sensitive_data_into_features_or_diagnostics():
    """Ensure sensitive data like passwords or tokens never appear in feature names or output."""
    sensitive_keywords = ("password", "token", "secret", "cookie", "user_agent", "ip_address")
    for name in FEATURE_NAMES:
        for kw in sensitive_keywords:
            assert kw not in name.lower()

    features = extract_session_features(
        events=[{"id": 1, "event_type": "CLICK", "page": "/login", "metadata": json.dumps({"password": "secret_password"})}],
        requests=[{"id": 1, "method": "POST", "path": "/login?token=abc", "status_code": 200}],
    )
    for kw in sensitive_keywords:
        assert kw not in str(features.feature_dict).lower()
        assert kw not in str(features.warnings).lower()


def test_train_cli_utility(temp_model_path):
    """Verify backend.ml.train generates baseline vectors and trains successfully."""
    from backend.ml.train import generate_synthetic_baseline
    vectors = generate_synthetic_baseline(15)
    assert len(vectors) == 15
    assert len(vectors[0]) == 21

    meta = train_unsupervised_model(vectors, model_path=temp_model_path, min_samples=10)
    assert meta["model_type"] == "IsolationForest"
    assert Path(temp_model_path).exists()

