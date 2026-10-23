"""
BotShield – Phase 3 Finite-State Machine (FSM) Behavioral Detection Tests

Unit and integration tests for:
- Initial state and ordinary human browsing
- Supported routes and catalog/product views
- Real cart actions and login attempts
- Rapid traversal detection and boundary conditions
- Path cycling / repetitive loop detection
- High request / low telemetry scraper detection
- Burst rate anomaly detection
- Inconsistent ordering (direct cart action)
- Human recovery transition and penalty mitigation
- Normal low-interaction session protection (accessibility & mobile safety)
- Unknown events, invalid metadata, and malformed timestamps
- Timestamp tie-breaking and deterministic sorting
- Duplicate record deduplication
- Deterministic repeatability
- Score bounds and explainability contract
- Database service integration and preservation of existing ML scores
"""

import json
import os
import sqlite3
import sys

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.db import (
    get_connection,
    get_session_risk,
    init_db,
    insert_events,
    insert_request,
    upsert_session,
    upsert_session_risk,
)
from backend.detection import (
    BehavioralFSM,
    FSMConfig,
    FSMEvent,
    FSMResult,
    FSMState,
    FSMSymbol,
    evaluate_session_fsm,
)
from backend.detection.service import normalize_session_records


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Provide an isolated, freshly initialized SQLite database for each test."""
    db_file = str(tmp_path / "test_fsm.db")
    monkeypatch.setenv("BOTSHIELD_DB", db_file)
    init_db(db_file)
    return db_file


SESS_1 = "aaaaaaaa-bbbb-cccc-dddd-111111111111"
SESS_2 = "bbbbbbbb-cccc-dddd-eeee-222222222222"


# ---------------------------------------------------------------------------
# 1. Initial State & Ordinary Browsing
# ---------------------------------------------------------------------------

def test_fsm_initial_state_empty_session():
    """An empty session returns INITIAL state, 0.0 risk score, and quality warning."""
    fsm = BehavioralFSM()
    result = fsm.process_events([])

    assert result.final_state == FSMState.INITIAL
    assert result.risk_score == 0.0
    assert result.anomaly_codes == []
    assert result.reasons == []
    assert "WARNING_EMPTY_SESSION" in result.evaluation_metadata["data_quality_warnings"]


def test_fsm_ordinary_human_browsing():
    """
    Standard human browsing:
    Home (/) -> Products (/products) -> Product detail (/product/1) -> About (/about)
    with realistic human dwell times (3-5s) and interaction gestures.
    """
    fsm = BehavioralFSM()
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.0, iso_timestamp="2026-10-23T12:00:00Z", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.USER_INTERACTION, epoch_timestamp=1001.0, iso_timestamp="2026-10-23T12:00:01Z", source="TELEMETRY"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1004.0, iso_timestamp="2026-10-23T12:00:04Z", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.USER_INTERACTION, epoch_timestamp=1006.0, iso_timestamp="2026-10-23T12:00:06Z", source="TELEMETRY"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1010.0, iso_timestamp="2026-10-23T12:00:10Z", source="HTTP_REQUEST", path="/product/1"),
        FSMEvent(symbol=FSMSymbol.USER_INTERACTION, epoch_timestamp=1012.0, iso_timestamp="2026-10-23T12:00:12Z", source="TELEMETRY"),
        FSMEvent(symbol=FSMSymbol.NAV_ABOUT, epoch_timestamp=1016.0, iso_timestamp="2026-10-23T12:00:16Z", source="HTTP_REQUEST", path="/about"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.BROWSING
    assert result.risk_score == 0.0
    assert result.anomaly_codes == []
    assert len(result.transition_history) == len(events)


# ---------------------------------------------------------------------------
# 2. Supported Routes: Product View, Cart Action, Login Attempt
# ---------------------------------------------------------------------------

def test_fsm_product_view_and_cart_action():
    """Valid e-commerce flow: Catalog -> Product -> Click Add to Cart."""
    fsm = BehavioralFSM()
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.0, iso_timestamp="t0", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1003.0, iso_timestamp="t1", source="HTTP_REQUEST", path="/product/2"),
        FSMEvent(symbol=FSMSymbol.USER_INTERACTION, epoch_timestamp=1004.0, iso_timestamp="t2", source="TELEMETRY"),
        FSMEvent(symbol=FSMSymbol.ACTION_CART, epoch_timestamp=1006.0, iso_timestamp="t3", source="TELEMETRY", path="/product/2"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.CART_ACTION
    assert result.risk_score == 0.0
    assert result.anomaly_codes == []


def test_fsm_login_attempt_flow():
    """Navigating to /login and clicking Sign In transitions to LOGIN_ATTEMPT."""
    fsm = BehavioralFSM()
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.0, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_LOGIN, epoch_timestamp=1003.0, iso_timestamp="t1", source="HTTP_REQUEST", path="/login"),
        FSMEvent(symbol=FSMSymbol.ACTION_LOGIN, epoch_timestamp=1005.0, iso_timestamp="t2", source="HTTP_REQUEST", path="/login"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.LOGIN_ATTEMPT
    assert result.risk_score == 0.0
    assert result.anomaly_codes == []


# ---------------------------------------------------------------------------
# 3. Anomaly: Rapid Traversal & Boundary Conditions
# ---------------------------------------------------------------------------

def test_fsm_rapid_traversal_detection():
    """Navigating between distinct pages with only 25ms delay triggers ANOMALY_RAPID_TRAVERSAL."""
    fsm = BehavioralFSM(FSMConfig(rapid_traversal_threshold_ms=150.0))
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.000, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.025, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.SUSPICIOUS
    assert "ANOMALY_RAPID_TRAVERSAL" in result.anomaly_codes
    assert result.risk_score >= 0.35
    assert len(result.reasons) >= 1
    assert "rapid page traversal" in result.reasons[0].lower()


def test_fsm_rapid_traversal_boundary_condition():
    """Threshold boundary testing: 149ms triggers anomaly, 151ms does not."""
    fsm = BehavioralFSM(FSMConfig(rapid_traversal_threshold_ms=150.0))

    # Just below threshold (149ms) -> ANOMALY
    sub_threshold = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.000, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.149, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),
    ]
    res_sub = fsm.process_events(sub_threshold)
    assert "ANOMALY_RAPID_TRAVERSAL" in res_sub.anomaly_codes

    # Just above threshold (151ms) -> NO ANOMALY
    super_threshold = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.000, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.151, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),
    ]
    res_super = fsm.process_events(super_threshold)
    assert "ANOMALY_RAPID_TRAVERSAL" not in res_super.anomaly_codes


# ---------------------------------------------------------------------------
# 4. Anomaly: Path Cycling & Repeated Loops
# ---------------------------------------------------------------------------

def test_fsm_path_cycling_loop_detection():
    """Visiting identical path 4 times in 5 seconds without interaction triggers ANOMALY_PATH_CYCLING."""
    fsm = BehavioralFSM(FSMConfig(max_repeated_path_count=4, repeat_window_seconds=10.0))
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1000.0, iso_timestamp="t0", source="HTTP_REQUEST", path="/product/1"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1001.0, iso_timestamp="t1", source="HTTP_REQUEST", path="/product/1"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1002.0, iso_timestamp="t2", source="HTTP_REQUEST", path="/product/1"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1003.0, iso_timestamp="t3", source="HTTP_REQUEST", path="/product/1"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.SUSPICIOUS
    assert "ANOMALY_PATH_CYCLING" in result.anomaly_codes
    assert any("repetitive cyclic requests" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# 5. Anomaly: High Request Volume / Low Telemetry (Scraper)
# ---------------------------------------------------------------------------

def test_fsm_high_request_low_telemetry_scraper():
    """A crawler making 6 server requests with 0 telemetry events is flagged."""
    fsm = BehavioralFSM(FSMConfig(min_requests_for_telemetry_check=5, min_telemetry_ratio=0.15))
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.0, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1002.0, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1004.0, iso_timestamp="t2", source="HTTP_REQUEST", path="/product/1"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1006.0, iso_timestamp="t3", source="HTTP_REQUEST", path="/product/2"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1008.0, iso_timestamp="t4", source="HTTP_REQUEST", path="/product/3"),
        FSMEvent(symbol=FSMSymbol.NAV_ABOUT, epoch_timestamp=1010.0, iso_timestamp="t5", source="HTTP_REQUEST", path="/about"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.SUSPICIOUS
    assert "ANOMALY_HIGH_REQUEST_LOW_TELEMETRY" in result.anomaly_codes
    assert result.risk_score >= 0.30


# ---------------------------------------------------------------------------
# 6. Anomaly: Burst Rate & Invalid Ordering
# ---------------------------------------------------------------------------

def test_fsm_burst_request_rate():
    """10 requests in 0.8s (12.5 req/s) triggers burst rate anomaly."""
    fsm = BehavioralFSM(FSMConfig(burst_rps_threshold=6.0))
    events = [
        FSMEvent(
            symbol=FSMSymbol.NAV_CATALOG,
            epoch_timestamp=1000.0 + (i * 0.08),
            iso_timestamp=f"t{i}",
            source="HTTP_REQUEST",
            path="/products",
        )
        for i in range(10)
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.SUSPICIOUS
    assert "ANOMALY_RAPID_REQUEST_BURST" in result.anomaly_codes


def test_fsm_invalid_ordering_direct_cart_action():
    """Triggering cart action from INITIAL state without viewing any page triggers ANOMALY_INVALID_ORDERING."""
    fsm = BehavioralFSM()
    events = [
        FSMEvent(symbol=FSMSymbol.ACTION_CART, epoch_timestamp=1000.0, iso_timestamp="t0", source="TELEMETRY", path="/cart"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.SUSPICIOUS
    assert "ANOMALY_INVALID_ORDERING" in result.anomaly_codes
    assert result.risk_score >= 0.20


# ---------------------------------------------------------------------------
# 7. Recovery Transition & Penalty Mitigation
# ---------------------------------------------------------------------------

def test_fsm_human_recovery_transition():
    """
    Session triggers an initial rapid traversal (moves to SUSPICIOUS),
    followed by sustained human telemetry interactions with normal dwell times.
    Transitions to RECOVERY with penalty mitigation.
    """
    fsm = BehavioralFSM(
        FSMConfig(
            rapid_traversal_threshold_ms=150.0,
            recovery_interaction_min_count=3,
            recovery_dwell_seconds=2.0,
            recovery_mitigation_factor=0.40,
        )
    )
    events = [
        # 1. Suspicious rapid traversal
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.000, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.050, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),

        # 2. Sustained human interactions spaced > 2s apart
        FSMEvent(symbol=FSMSymbol.USER_INTERACTION, epoch_timestamp=1003.0, iso_timestamp="t2", source="TELEMETRY"),
        FSMEvent(symbol=FSMSymbol.USER_INTERACTION, epoch_timestamp=1006.0, iso_timestamp="t3", source="TELEMETRY"),
        FSMEvent(symbol=FSMSymbol.USER_INTERACTION, epoch_timestamp=1009.0, iso_timestamp="t4", source="TELEMETRY"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.RECOVERY
    assert "ANOMALY_RAPID_TRAVERSAL" in result.anomaly_codes
    # Original penalty was 0.35, mitigated by 0.40 -> 0.14
    assert pytest.approx(result.risk_score, 0.01) == 0.14


# ---------------------------------------------------------------------------
# 8. Protection for Low-Interaction Users (Accessibility / Mobile Safety)
# ---------------------------------------------------------------------------

def test_fsm_normal_low_interaction_session_not_marked_suspicious():
    """
    A visitor views 3 pages with normal dwell times (5s each) but performs no
    mouse or scroll interactions (e.g. screen reader or fast reader).
    Must NOT be marked SUSPICIOUS, risk score must remain 0.0.
    """
    fsm = BehavioralFSM()
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.0, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1005.0, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1010.0, iso_timestamp="t2", source="HTTP_REQUEST", path="/product/1"),
    ]

    result = fsm.process_events(events)

    assert result.final_state == FSMState.PRODUCT_VIEW
    assert result.risk_score == 0.0
    assert result.anomaly_codes == []


# ---------------------------------------------------------------------------
# 9. Data Normalization, Missing/Malformed Timestamps, and Deduplication
# ---------------------------------------------------------------------------

def test_normalizer_tie_breaking_order():
    """When timestamps are identical, tie-breaking orders HTTP requests before telemetry."""
    event_rows = [
        {"id": 1, "event_type": "PAGE_VIEW", "timestamp": "2026-10-23T12:00:00Z", "page": "/products", "metadata": "{}"},
        {"id": 2, "event_type": "CLICK", "timestamp": "2026-10-23T12:00:00Z", "page": "/products", "metadata": json.dumps({"element_id": "nav-link"})},
    ]
    request_rows = [
        {"id": 10, "method": "GET", "path": "/products", "timestamp": "2026-10-23T12:00:00Z", "status_code": 200},
    ]

    norm_events, _ = normalize_session_records(event_rows, request_rows)

    assert len(norm_events) == 3
    # Priority order: HTTP_REQUEST (1), PAGE_VIEW (2), CLICK (3)
    assert norm_events[0].source == "HTTP_REQUEST"
    assert norm_events[1].raw_type == "PAGE_VIEW"
    assert norm_events[2].raw_type == "CLICK"


def test_normalizer_malformed_timestamps_and_unknown_events():
    """Malformed timestamps and unknown event types are handled gracefully with warnings."""
    event_rows = [
        {"id": 1, "event_type": "UNKNOWN_CUSTOM_TYPE", "timestamp": "invalid-timestamp", "page": "/test", "metadata": "not-valid-json"},
    ]
    request_rows = [
        {"id": 2, "method": "GET", "path": "/strange", "timestamp": "", "status_code": 200},
    ]

    norm_events, warnings = normalize_session_records(event_rows, request_rows)

    assert len(norm_events) == 2
    assert any("WARNING_MALFORMED_TIMESTAMP" in w for w in warnings)
    assert any("WARNING_INVALID_METADATA" in w for w in warnings)


def test_normalizer_deduplicates_identical_records():
    """Identical duplicate records are filtered with a warning."""
    event_rows = [
        {"id": 1, "event_type": "PAGE_VIEW", "timestamp": "2026-10-23T12:00:00Z", "page": "/products", "metadata": "{}"},
        {"id": 2, "event_type": "PAGE_VIEW", "timestamp": "2026-10-23T12:00:00Z", "page": "/products", "metadata": "{}"},
    ]

    norm_events, warnings = normalize_session_records(event_rows, [])

    assert len(norm_events) == 1
    assert "WARNING_DUPLICATE_EVENT" in warnings


# ---------------------------------------------------------------------------
# 10. Deterministic Repeatability & Score Bounds
# ---------------------------------------------------------------------------

def test_fsm_deterministic_repeatability():
    """Executing the exact same event sequence 10 times produces bit-for-bit identical results."""
    fsm = BehavioralFSM()
    events = [
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.0, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.02, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_PRODUCT, epoch_timestamp=1000.04, iso_timestamp="t2", source="HTTP_REQUEST", path="/product/1"),
    ]

    runs = [fsm.process_events(events) for _ in range(10)]

    for r in runs[1:]:
        assert r.final_state == runs[0].final_state
        assert r.risk_score == runs[0].risk_score
        assert r.anomaly_codes == runs[0].anomaly_codes
        assert r.reasons == runs[0].reasons


def test_fsm_score_bounds_and_explanations():
    """Accumulating all possible anomalies must never exceed 1.0 risk score."""
    fsm = BehavioralFSM()
    events = [
        # Burst + rapid traversal + path cycling
        FSMEvent(symbol=FSMSymbol.NAV_HOME, epoch_timestamp=1000.00, iso_timestamp="t0", source="HTTP_REQUEST", path="/"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.02, iso_timestamp="t1", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.04, iso_timestamp="t2", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.06, iso_timestamp="t3", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.08, iso_timestamp="t4", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.NAV_CATALOG, epoch_timestamp=1000.10, iso_timestamp="t5", source="HTTP_REQUEST", path="/products"),
        FSMEvent(symbol=FSMSymbol.ACTION_CART, epoch_timestamp=1000.12, iso_timestamp="t6", source="TELEMETRY", path="/products"),
    ]

    result = fsm.process_events(events)

    assert 0.0 <= result.risk_score <= 1.0
    # Every anomaly code must have a non-empty explanation in reasons
    assert len(result.anomaly_codes) == len(result.reasons)
    for reason in result.reasons:
        assert isinstance(reason, str) and len(reason) > 10


# ---------------------------------------------------------------------------
# 11. Service Integration & Database Persistence
# ---------------------------------------------------------------------------

def test_service_evaluates_and_persists_session_risk(temp_db):
    """Integration test: evaluate_session_fsm loads from DB and updates session_risks."""
    conn = get_connection(temp_db)
    upsert_session(conn, SESS_1)

    # Insert requests and events
    insert_request(conn, SESS_1, "GET", "/", 200, timestamp="2026-10-23T12:00:00Z")
    insert_request(conn, SESS_1, "GET", "/products", 200, timestamp="2026-10-23T12:00:00.030Z")  # rapid traversal
    insert_events(
        conn,
        SESS_1,
        [{"event_type": "PAGE_VIEW", "timestamp": "2026-10-23T12:00:00Z", "page": "/"}],
    )
    conn.commit()
    conn.close()

    result = evaluate_session_fsm(SESS_1, db_path=temp_db, persist=True)

    assert result.final_state == FSMState.SUSPICIOUS
    assert "ANOMALY_RAPID_TRAVERSAL" in result.anomaly_codes

    # Verify database persistence in session_risks
    conn = get_connection(temp_db)
    risk_row = get_session_risk(conn, SESS_1)
    conn.close()

    assert risk_row is not None
    assert pytest.approx(risk_row["fsm_score"], 0.001) == result.risk_score
    reasons = json.loads(risk_row["reasons"])
    assert len(reasons) >= 1
    assert any("rapid page traversal" in r.lower() for r in reasons)


def test_service_preserves_existing_ml_score(temp_db):
    """Evaluating FSM must preserve pre-existing ML and combined scores in session_risks."""
    conn = get_connection(temp_db)
    upsert_session(conn, SESS_2)

    # Pre-populate risk record with existing ML score
    upsert_session_risk(
        conn,
        session_id=SESS_2,
        risk_score=0.75,
        fsm_score=0.10,
        ml_score=0.85,
        reasons=["pre_existing_ml_flag"],
    )
    insert_request(conn, SESS_2, "GET", "/", 200, timestamp="2026-10-23T12:00:00Z")
    conn.commit()
    conn.close()

    evaluate_session_fsm(SESS_2, db_path=temp_db, persist=True)

    conn = get_connection(temp_db)
    risk_row = get_session_risk(conn, SESS_2)
    conn.close()

    assert risk_row is not None
    # ml_score must be completely preserved
    assert pytest.approx(risk_row["ml_score"], 0.001) == 0.85
