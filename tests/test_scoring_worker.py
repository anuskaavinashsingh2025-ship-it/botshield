"""Tests for BotShield Asynchronous Evaluation Worker and Pluggable Rate Limiter (Phase 11).

Validates:
1. Discovery of pending/unassessed and updated sessions from SQLite.
2. Batch evaluation cycle execution and risk record persistence.
3. Threaded background worker startup, execution, and graceful shutdown.
4. Worker health metrics and error resilience.
5. Pluggable distributed rate limiter adapter functionality.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
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
from backend.enforcement.limiter import (
    BoundedRateLimiter,
    DistributedRateLimiterAdapter,
    RateLimiterBackend,
)
from backend.ml.model import train_unsupervised_model
from backend.ml.train import generate_synthetic_baseline
from backend.scoring.worker import (
    AsyncEvaluationWorker,
    get_pending_sessions,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with clean schema."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    init_db(db_path)
    yield db_path
    try:
        Path(db_path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def trained_model_path():
    """Create an isolated trained IsolationForest model."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "isolation_forest.joblib"
        X = generate_synthetic_baseline(sample_count=60, random_state=42)
        train_unsupervised_model(
            training_data=X,
            model_path=model_file,
            contamination=0.1,
            random_state=42,
        )
        yield str(model_file)


# ==============================================================================
# 1. Pending Sessions Query Tests
# ==============================================================================

def test_get_pending_sessions_unassessed(temp_db):
    """Sessions without session_risks rows should be returned as pending."""
    conn = get_connection(temp_db)
    upsert_session(conn, "sess-unassessed-1")
    upsert_session(conn, "sess-unassessed-2")
    conn.commit()

    pending = get_pending_sessions(conn, limit=10)
    conn.close()

    assert "sess-unassessed-1" in pending
    assert "sess-unassessed-2" in pending
    assert len(pending) == 2


def test_get_pending_sessions_already_evaluated(temp_db):
    """Sessions evaluated after their last activity should not be returned."""
    conn = get_connection(temp_db)
    upsert_session(conn, "sess-evaluated-1")
    conn.commit()

    # Artificially set updated_at in the future so last_seen < updated_at
    upsert_session_risk(
        conn,
        session_id="sess-evaluated-1",
        risk_score=0.1,
        fsm_score=0.1,
        ml_score=0.1,
        reasons=["Normal"],
        updated_at="2099-01-01T00:00:00Z",
    )
    conn.commit()

    pending = get_pending_sessions(conn, limit=10)
    conn.close()

    assert "sess-evaluated-1" not in pending


def test_get_pending_sessions_updated_after_evaluation(temp_db):
    """Sessions with new activity after an old evaluation should be returned."""
    conn = get_connection(temp_db)
    upsert_session(conn, "sess-updated-1")
    conn.commit()

    # Set an old evaluation timestamp
    upsert_session_risk(
        conn,
        session_id="sess-updated-1",
        risk_score=0.1,
        fsm_score=0.1,
        ml_score=0.1,
        reasons=["Normal"],
        updated_at="2000-01-01T00:00:00Z",
    )
    conn.commit()

    pending = get_pending_sessions(conn, limit=10)
    conn.close()

    assert "sess-updated-1" in pending


def test_get_pending_sessions_limit(temp_db):
    """Query should strictly honor the limit parameter."""
    conn = get_connection(temp_db)
    for i in range(15):
        upsert_session(conn, f"sess-batch-{i}")
    conn.commit()

    pending_5 = get_pending_sessions(conn, limit=5)
    pending_10 = get_pending_sessions(conn, limit=10)
    conn.close()

    assert len(pending_5) == 5
    assert len(pending_10) == 10


# ==============================================================================
# 2. Batch Evaluation Cycle Tests
# ==============================================================================

def test_async_worker_run_once(temp_db, trained_model_path):
    """run_once should evaluate all pending sessions and persist risk records."""
    conn = get_connection(temp_db)
    sess_id = "aaaaaaaa-1111-4444-8888-000000000001"
    upsert_session(conn, sess_id)
    insert_request(conn, sess_id, "GET", "/products", 200, "2026-10-24T00:00:00Z")
    insert_events(conn, sess_id, [
        {"event_type": "PAGE_VIEW", "timestamp": "2026-10-24T00:00:01Z", "page": "/products"},
    ])
    conn.commit()
    conn.close()

    worker = AsyncEvaluationWorker(
        db_path=temp_db,
        model_path=trained_model_path,
        batch_size=10,
    )
    res = worker.run_once()

    assert res["evaluated_count"] == 1
    assert res["errors_count"] == 0
    assert res["duration_ms"] >= 0.0

    # Verify risk row created in database
    conn = get_connection(temp_db)
    risk_row = get_session_risk(conn, sess_id)
    conn.close()

    assert risk_row is not None
    assert risk_row["session_id"] == sess_id
    assert risk_row["risk_score"] >= 0.0


def test_async_worker_run_once_empty_db(temp_db):
    """run_once on an empty database should complete cleanly with 0 evaluated."""
    worker = AsyncEvaluationWorker(db_path=temp_db, batch_size=10)
    res = worker.run_once()

    assert res["pending_count"] == 0
    assert res["evaluated_count"] == 0
    assert res["errors_count"] == 0


# ==============================================================================
# 3. Threaded Worker Lifecycle & Status Tests
# ==============================================================================

def test_async_worker_thread_lifecycle(temp_db):
    """Worker start and stop should manage background daemon thread cleanly."""
    worker = AsyncEvaluationWorker(
        db_path=temp_db,
        interval_seconds=0.1,  # Fast interval for test
        batch_size=10,
    )
    assert worker.is_running is False

    worker.start()
    assert worker.is_running is True

    time.sleep(0.3)  # Allow a couple cycles

    worker.stop(timeout=2.0)
    assert worker.is_running is False

    status = worker.get_status()
    assert status["running"] is False
    assert status["total_cycles"] >= 1
    assert status["last_run_time"] is not None


# ==============================================================================
# 4. Distributed Rate Limiter Adapter Tests
# ==============================================================================

class DummyStorage:
    """Mock storage simulating external key-value store."""
    def __init__(self):
        self._store = {}

    def clear(self):
        self._store.clear()

    def dbsize(self):
        return len(self._store)


def test_distributed_rate_limiter_adapter():
    """DistributedRateLimiterAdapter should accurately enforce sliding-window quotas."""
    storage = DummyStorage()
    adapter = DistributedRateLimiterAdapter(storage)

    # 1. Under limit: permit requests
    limited, retry = adapter.check("client_1", max_requests=3, window_seconds=60.0, now=100.0)
    assert limited is False
    assert retry == 0

    limited, retry = adapter.check("client_1", max_requests=3, window_seconds=60.0, now=101.0)
    assert limited is False

    limited, retry = adapter.check("client_1", max_requests=3, window_seconds=60.0, now=102.0)
    assert limited is False

    # 2. Exceeded limit: block and return retry-after
    limited, retry = adapter.check("client_1", max_requests=3, window_seconds=60.0, now=103.0)
    assert limited is True
    assert retry == 57  # (100.0 + 60 - 103.0) = 57s

    # 3. Tracked count and reset
    assert adapter.tracked_count == 1
    adapter.reset()
    assert adapter.tracked_count == 0

    # 4. Independent clients
    adapter.check("client_a", max_requests=2, window_seconds=10.0, now=100.0)
    adapter.check("client_b", max_requests=2, window_seconds=10.0, now=100.0)
    assert adapter.tracked_count == 2
