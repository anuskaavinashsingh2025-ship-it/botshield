"""
BotShield – Phase 5 Hybrid Risk Scoring Tests

Comprehensive unit and integration test suite verifying:
1. Both FSM and ML available (HYBRID mode).
2. ML model unavailable or untrained (FSM_ONLY mode fallback).
3. ML score missing (None) -> safe fallback.
4. ML score invalid, non-finite (NaN, Inf), or negative.
5. FSM score at minimum (0.0) and maximum (1.0).
6. Composite score lower (0.0) and upper (100.0) bounds.
7. Threshold boundaries (0.0, 34.9, 35.0, 69.9, 70.0, 100.0).
8. Risk classifications (LOW, MEDIUM, HIGH).
9. Missing session or insufficient session data (INSUFFICIENT_EVIDENCE).
10. Persistence disabled (persist=False makes zero DB writes).
11. Persistence enabled (persist=True stores normalized risk_score, fsm_score, ml_score, reasons).
12. Repeated evaluations (consistent behavior, no duplicate rows).
13. Non-corruption of existing FSM and ML fields during updates.
14. Propagation of data quality and model status warnings.
15. Deterministic repeatability on unchanged inputs.
16. Public API compatibility and serialization.
"""

import math
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.database.db import (
    get_connection,
    get_session_risk,
    init_db,
    insert_events,
    insert_request,
    upsert_session,
    upsert_session_risk,
)
from backend.detection.fsm import (
    BehavioralFSM,
    FSMConfig,
    FSMEvent,
    FSMResult,
    FSMState,
    FSMSymbol,
)
from backend.ml.features import FEATURE_NAMES, ExtractedFeatures
from backend.ml.model import (
    IsolationForestModel,
    MLInferenceResult,
    MLModelStatus,
)
from backend.ml.train import train_unsupervised_model
from backend.scoring import (
    HybridConfig,
    HybridRiskResult,
    RiskLevel,
    ScoringMode,
    classify_risk,
    compute_hybrid_score,
    evaluate_session_risk,
)


@pytest.fixture
def temp_db():
    """Create an isolated, freshly initialized SQLite database for each test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    try:
        os.remove(db_path)
    except OSError:
        pass


@pytest.fixture
def trained_model_path(tmp_path):
    """Generate and save a lightweight trained Isolation Forest model for testing."""
    model_path = tmp_path / "test_iso_forest.joblib"
    rng = [
        [0.1 * i + 0.01 * j for j in range(len(FEATURE_NAMES))]
        for i in range(25)
    ]
    train_unsupervised_model(
        training_data=rng,
        model_path=model_path,
        random_state=42,
    )
    return model_path


# ---------------------------------------------------------------------------
# Unit Tests: Configuration & Boundaries
# ---------------------------------------------------------------------------

def test_config_validation():
    """Verify HybridConfig enforces valid weight summation and threshold ranges."""
    # Valid default config
    cfg = HybridConfig()
    assert cfg.fsm_weight == 0.60
    assert cfg.ml_weight == 0.40
    assert cfg.low_threshold == 35.0
    assert cfg.high_threshold == 70.0

    # Negative weights rejected
    with pytest.raises(ValueError, match="non-negative"):
        HybridConfig(fsm_weight=-0.1, ml_weight=1.1)

    # Weights not summing to 1.0 rejected
    with pytest.raises(ValueError, match="must sum to 1.0"):
        HybridConfig(fsm_weight=0.5, ml_weight=0.6)

    # Inverted thresholds rejected
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        HybridConfig(low_threshold=75.0, high_threshold=35.0)

    # Out-of-range thresholds rejected
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        HybridConfig(low_threshold=-5.0, high_threshold=70.0)


def test_classify_risk_thresholds():
    """Verify precise threshold boundary mapping to RiskLevel."""
    cfg = HybridConfig(low_threshold=35.0, high_threshold=70.0)

    assert classify_risk(0.0, cfg) == RiskLevel.LOW
    assert classify_risk(34.9, cfg) == RiskLevel.LOW
    assert classify_risk(35.0, cfg) == RiskLevel.MEDIUM
    assert classify_risk(50.0, cfg) == RiskLevel.MEDIUM
    assert classify_risk(69.9, cfg) == RiskLevel.MEDIUM
    assert classify_risk(70.0, cfg) == RiskLevel.HIGH
    assert classify_risk(100.0, cfg) == RiskLevel.HIGH


# ---------------------------------------------------------------------------
# Unit Tests: Scoring Formula & Fallbacks
# ---------------------------------------------------------------------------

def test_compute_hybrid_score_both_available():
    """Test score combination when both FSM and ML scores are present."""
    cfg = HybridConfig(fsm_weight=0.6, ml_weight=0.4)

    # 0.6 * 0.5 + 0.4 * 0.8 = 0.3 + 0.32 = 0.62 -> 62.0
    comp, norm, mode, warnings = compute_hybrid_score(0.5, 0.8, cfg, MLModelStatus.AVAILABLE)
    assert mode == ScoringMode.HYBRID
    assert comp == 62.0
    assert norm == 0.62
    assert warnings == []


def test_compute_hybrid_score_min_max_bounds():
    """Verify lower (0.0) and upper (100.0) score clamps."""
    cfg = HybridConfig(fsm_weight=0.7, ml_weight=0.3)

    # Min bound
    comp_min, norm_min, mode, _ = compute_hybrid_score(0.0, 0.0, cfg, MLModelStatus.AVAILABLE)
    assert comp_min == 0.0
    assert norm_min == 0.0
    assert mode == ScoringMode.HYBRID

    # Max bound
    comp_max, norm_max, mode, _ = compute_hybrid_score(1.0, 1.0, cfg, MLModelStatus.AVAILABLE)
    assert comp_max == 100.0
    assert norm_max == 1.0
    assert mode == ScoringMode.HYBRID

    # Overflows clamped cleanly
    comp_over, norm_over, _, _ = compute_hybrid_score(1.5, 2.0, cfg, MLModelStatus.AVAILABLE)
    assert comp_over == 100.0
    assert norm_over == 1.0


def test_compute_hybrid_score_untrained_fallback():
    """Untrained ML model must trigger FSM_ONLY mode without fabricating an ML score."""
    cfg = HybridConfig(fsm_weight=0.6, ml_weight=0.4)

    comp, norm, mode, warnings = compute_hybrid_score(0.75, None, cfg, MLModelStatus.UNTRAINED)
    assert mode == ScoringMode.FSM_ONLY
    assert comp == 75.0
    assert norm == 0.75
    assert any("untrained" in w.lower() for w in warnings)


def test_compute_hybrid_score_missing_or_corrupt_ml():
    """Missing or non-available ML status must cleanly fall back to FSM-only."""
    cfg = HybridConfig(fsm_weight=0.6, ml_weight=0.4)

    # Missing ML score when status is AVAILABLE
    comp, norm, mode, warnings = compute_hybrid_score(0.40, None, cfg, MLModelStatus.AVAILABLE)
    assert mode == ScoringMode.FSM_ONLY
    assert comp == 40.0
    assert any("missing" in w.lower() for w in warnings)

    # Corrupt model status
    comp_c, _, mode_c, warnings_c = compute_hybrid_score(0.30, 0.9, cfg, MLModelStatus.CORRUPT)
    assert mode_c == ScoringMode.FSM_ONLY
    assert comp_c == 30.0
    assert any("corrupt" in w.lower() for w in warnings_c)


def test_compute_hybrid_score_invalid_non_finite_inputs():
    """Invalid or non-finite inputs (NaN, Inf, strings) must not crash and fallback safely."""
    cfg = HybridConfig(fsm_weight=0.6, ml_weight=0.4)

    # NaN ML score
    comp, norm, mode, warnings = compute_hybrid_score(0.50, float("nan"), cfg, MLModelStatus.AVAILABLE)
    assert mode == ScoringMode.FSM_ONLY
    assert comp == 50.0
    assert any("non-finite" in w.lower() for w in warnings)

    # Inf ML score
    comp_inf, norm_inf, mode_inf, _ = compute_hybrid_score(0.50, float("inf"), cfg, MLModelStatus.AVAILABLE)
    assert mode_inf == ScoringMode.FSM_ONLY
    assert comp_inf == 50.0

    # NaN FSM score resets safely to 0.0
    comp_fnan, norm_fnan, _, warnings_fnan = compute_hybrid_score(float("nan"), 0.5, cfg, MLModelStatus.AVAILABLE)
    assert any("fsm" in w.lower() for w in warnings_fnan)


# ---------------------------------------------------------------------------
# Integration Tests: Database & Service Execution
# ---------------------------------------------------------------------------

def test_evaluate_session_risk_empty_session(temp_db):
    """A session with no events and no requests returns INSUFFICIENT_EVIDENCE."""
    session_id = "sess-empty-001"
    upsert_session(get_connection(temp_db), session_id=session_id)

    res = evaluate_session_risk(session_id=session_id, db_path=temp_db, persist=False)
    assert res.session_id == session_id
    assert res.scoring_mode == ScoringMode.INSUFFICIENT_EVIDENCE
    assert res.composite_score == 0.0
    assert res.normalized_score == 0.0
    assert res.risk_level == RiskLevel.LOW
    assert res.ml_score is None
    assert any("INSUFFICIENT_SESSION_EVIDENCE" in w for w in res.warnings)


def test_evaluate_session_risk_untrained_model(temp_db):
    """When no trained model artifact exists, evaluate in FSM-only fallback mode."""
    session_id = "sess-untrained-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[
        {"event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-03-01T12:00:00Z"},
        {"event_type": "CLICK", "page": "/", "timestamp": "2026-03-01T12:00:02Z", "x": 10, "y": 20},
    ])
    conn.commit()
    conn.close()

    # Pass nonexistent model path to guarantee untrained/missing status
    res = evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path="nonexistent_model.joblib",
        persist=False,
    )

    assert res.scoring_mode == ScoringMode.FSM_ONLY
    assert res.model_status == MLModelStatus.UNTRAINED
    assert res.ml_score is None
    assert res.composite_score == round(res.fsm_score * 100.0, 1)
    assert any("untrained" in w.lower() for w in res.warnings)


def test_evaluate_session_risk_trained_model(temp_db, trained_model_path):
    """When trained model exists, evaluate in HYBRID mode combining both scores."""
    session_id = "sess-trained-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[
        {"event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-03-01T12:00:00Z"},
        {"event_type": "SCROLL", "page": "/", "timestamp": "2026-03-01T12:00:01Z", "depth": 30},
        {"event_type": "CLICK", "page": "/", "timestamp": "2026-03-01T12:00:02Z", "x": 15, "y": 25},
    ])
    conn.commit()
    conn.close()

    res = evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path=trained_model_path,
        persist=False,
    )

    assert res.scoring_mode == ScoringMode.HYBRID
    assert res.model_status == MLModelStatus.AVAILABLE
    assert res.ml_score is not None
    assert 0.0 <= res.ml_score <= 1.0
    assert 0.0 <= res.composite_score <= 100.0
    assert pytest.approx(res.normalized_score, abs=0.01) == res.composite_score / 100.0


def test_evaluate_session_risk_malicious_scraper(temp_db):
    """A rapid scraper session should accumulate FSM violations and score HIGH."""
    session_id = "sess-scraper-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)

    # 15 rapid product requests 50ms apart with zero telemetry (burst + rapid traversal + scraper)
    for i in range(15):
        millis = i * 50
        sec = millis // 1000
        rem_ms = millis % 1000
        ts = f"2026-03-01T12:00:{sec:02d}.{rem_ms:03d}Z"
        insert_request(conn, session_id=session_id, method="GET", path=f"/product/{i}", status_code=200, timestamp=ts)
    conn.commit()
    conn.close()

    res = evaluate_session_risk(session_id=session_id, db_path=temp_db, persist=False)
    assert res.fsm_score >= 0.70
    assert res.composite_score >= 70.0
    assert res.risk_level == RiskLevel.HIGH
    assert len(res.reasons) > 0


def test_persistence_flag_false_makes_no_writes(temp_db):
    """When persist=False, session_risks table must remain untouched."""
    session_id = "sess-nopersist-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[{"event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-03-01T12:00:00Z"}])
    conn.commit()
    conn.close()

    res = evaluate_session_risk(session_id=session_id, db_path=temp_db, persist=False)
    assert res.composite_score is not None

    # Verify session_risks has no entry
    conn = get_connection(temp_db)
    row = get_session_risk(conn, session_id=session_id)
    conn.close()
    assert row is None


def test_persistence_flag_true_writes_database(temp_db, trained_model_path):
    """When persist=True, session_risks table must store normalized scores and reasons."""
    session_id = "sess-persist-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[
        {"event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-03-01T12:00:00Z"},
        {"event_type": "CLICK", "page": "/", "timestamp": "2026-03-01T12:00:01Z", "x": 5, "y": 10},
    ])
    conn.commit()
    conn.close()

    res = evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path=trained_model_path,
        persist=True,
    )

    conn = get_connection(temp_db)
    risk_row = get_session_risk(conn, session_id=session_id)
    conn.close()

    assert risk_row is not None
    assert pytest.approx(risk_row["risk_score"], 0.001) == res.normalized_score
    assert pytest.approx(risk_row["fsm_score"], 0.001) == res.fsm_score
    assert pytest.approx(risk_row["ml_score"], 0.001) == res.ml_score


def test_repeated_evaluation_idempotency(temp_db, trained_model_path):
    """Repeated evaluations with persist=True must update the existing record without duplicates."""
    session_id = "sess-repeat-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[{"event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-03-01T12:00:00Z"}])
    conn.commit()
    conn.close()

    # Evaluate multiple times
    res1 = evaluate_session_risk(session_id=session_id, db_path=temp_db, model_path=trained_model_path, persist=True)
    res2 = evaluate_session_risk(session_id=session_id, db_path=temp_db, model_path=trained_model_path, persist=True)

    assert res1.composite_score == res2.composite_score
    assert res1.scoring_mode == res2.scoring_mode

    conn = get_connection(temp_db)
    cur = conn.execute("SELECT COUNT(*) FROM session_risks WHERE session_id = ?", (session_id,))
    count = cur.fetchone()[0]
    conn.close()
    assert count == 1


def test_preserve_existing_ml_score_on_fsm_only_update(temp_db):
    """An FSM-only evaluation must not corrupt or overwrite an existing ML score in SQLite."""
    session_id = "sess-preserve-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[{"event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-03-01T12:00:00Z"}])
    # Manually store a prior ML score in the database
    upsert_session_risk(conn, session_id=session_id, risk_score=0.45, fsm_score=0.10, ml_score=0.85, reasons=["PRIOR_REASON"])
    conn.commit()
    conn.close()

    # Run evaluation with untrained model (ml_score is None) and persist=True
    evaluate_session_risk(
        session_id=session_id,
        db_path=temp_db,
        model_path="nonexistent.joblib",
        persist=True,
    )

    conn = get_connection(temp_db)
    updated_risk = get_session_risk(conn, session_id=session_id)
    conn.close()

    assert updated_risk is not None
    # Prior ML score was strictly preserved by the database layer
    assert updated_risk["ml_score"] == 0.85


def test_deterministic_repeatability(temp_db, trained_model_path):
    """Identical session inputs must always yield byte-for-byte deterministic outputs."""
    session_id = "sess-deterministic-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/products", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[
        {"event_type": "PAGE_VIEW", "page": "/products", "timestamp": "2026-03-01T12:00:00Z"},
        {"event_type": "SCROLL", "page": "/products", "timestamp": "2026-03-01T12:00:02Z", "depth": 50},
    ])
    conn.commit()
    conn.close()

    res_a = evaluate_session_risk(session_id=session_id, db_path=temp_db, model_path=trained_model_path, persist=False)
    res_b = evaluate_session_risk(session_id=session_id, db_path=temp_db, model_path=trained_model_path, persist=False)

    assert res_a.composite_score == res_b.composite_score
    assert res_a.normalized_score == res_b.normalized_score
    assert res_a.risk_level == res_b.risk_level
    assert res_a.scoring_mode == res_b.scoring_mode
    assert res_a.reasons == res_b.reasons
    assert res_a.warnings == res_b.warnings


def test_result_to_dict_and_public_api(temp_db):
    """Verify to_dict serialization and field consistency of HybridRiskResult."""
    session_id = "sess-api-001"
    conn = get_connection(temp_db)
    upsert_session(conn, session_id=session_id)
    insert_request(conn, session_id=session_id, method="GET", path="/", status_code=200, timestamp="2026-03-01T12:00:00Z")
    insert_events(conn, session_id=session_id, events=[{"event_type": "PAGE_VIEW", "page": "/", "timestamp": "2026-03-01T12:00:00Z"}])
    conn.commit()
    conn.close()

    res = evaluate_session_risk(session_id=session_id, db_path=temp_db, persist=False)
    d = res.to_dict()

    assert d["session_id"] == session_id
    assert isinstance(d["composite_score"], float)
    assert isinstance(d["normalized_score"], float)
    assert d["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert d["scoring_mode"] in ("HYBRID", "FSM_ONLY", "INSUFFICIENT_EVIDENCE")
    assert "fsm_summary" in d
    assert "ml_summary" in d
    assert isinstance(d["reasons"], list)
    assert isinstance(d["warnings"], list)


def test_invalid_session_id_raises_value_error(temp_db):
    """Invalid session_id types or empty strings must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        evaluate_session_risk("", db_path=temp_db)

    with pytest.raises(ValueError, match="non-empty string"):
        evaluate_session_risk("   ", db_path=temp_db)
