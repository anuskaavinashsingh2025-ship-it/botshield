"""BotShield – Asynchronous Background Session Risk Evaluation Worker (Phase 11).

Provides an asynchronous background evaluation engine that continuously
identifies unassessed or updated client sessions in SQLite and executes
hybrid risk scoring (FSM + ML) without delaying live HTTP request cycles.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from backend.database.db import get_connection
from backend.scoring.service import evaluate_session_risk

logger = logging.getLogger("botshield.scoring.worker")


def get_pending_sessions(conn: sqlite3.Connection, limit: int = 50) -> List[str]:
    """
    Retrieve session IDs that require risk evaluation or re-evaluation.

    A session requires evaluation if:
    1. It has never been evaluated (no entry in session_risks).
    2. It has observed new activity since its last evaluation (last_seen > updated_at).

    Parameters:
        conn: SQLite connection.
        limit: Maximum number of candidate session IDs to retrieve.

    Returns:
        List of session ID strings ordered by most recent activity.
    """
    query = """
        SELECT s.session_id
        FROM sessions s
        LEFT JOIN session_risks sr ON s.session_id = sr.session_id
        WHERE sr.session_id IS NULL OR s.last_seen > sr.updated_at
        ORDER BY s.last_seen DESC
        LIMIT ?
    """
    cur = conn.execute(query, (max(1, limit),))
    rows = cur.fetchall()
    return [str(row[0]) for row in rows]


class AsyncEvaluationWorker:
    """
    Thread-safe asynchronous background evaluator for BotShield sessions.
    
    Can run as a standalone worker process or as a background daemon thread
    within a Flask application instance.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        model_path: Optional[str | Path] = None,
        interval_seconds: float = 5.0,
        batch_size: int = 50,
    ) -> None:
        self.db_path = str(db_path) if db_path else None
        self.model_path = model_path
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.batch_size = max(1, int(batch_size))

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Telemetry & Status Metrics
        self._total_evaluated = 0
        self._total_cycles = 0
        self._total_errors = 0
        self._last_run_time: Optional[str] = None
        self._last_batch_duration_ms: float = 0.0

    @property
    def is_running(self) -> bool:
        """Check if background evaluation thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> Dict[str, Any]:
        """
        Execute a single batch evaluation cycle.

        Returns:
            Dictionary containing metrics for this evaluation cycle.
        """
        start_t = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        evaluated_in_cycle = 0
        errors_in_cycle = 0

        conn = get_connection(self.db_path)
        try:
            pending_ids = get_pending_sessions(conn, limit=self.batch_size)
        finally:
            conn.close()

        if pending_ids:
            logger.debug("[ASYNC_WORKER] Found %d pending sessions for risk evaluation", len(pending_ids))

        for session_id in pending_ids:
            if self._stop_event.is_set():
                break

            try:
                evaluate_session_risk(
                    session_id=session_id,
                    db_path=self.db_path,
                    model_path=self.model_path,
                    persist=True,
                )
                evaluated_in_cycle += 1
            except Exception as exc:
                errors_in_cycle += 1
                logger.error("[ASYNC_WORKER] Error evaluating session %s: %s", session_id, exc)

        duration_ms = round((time.time() - start_t) * 1000.0, 2)

        with self._lock:
            self._total_evaluated += evaluated_in_cycle
            self._total_errors += errors_in_cycle
            self._total_cycles += 1
            self._last_run_time = now_iso
            self._last_batch_duration_ms = duration_ms

        return {
            "cycle_timestamp": now_iso,
            "pending_count": len(pending_ids),
            "evaluated_count": evaluated_in_cycle,
            "errors_count": errors_in_cycle,
            "duration_ms": duration_ms,
        }

    def _worker_loop(self) -> None:
        """Main loop executed within the background daemon thread."""
        logger.info("[ASYNC_WORKER] Background evaluation thread started (interval: %.1fs)", self.interval_seconds)
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                logger.error("[ASYNC_WORKER] Unexpected error in worker loop: %s", exc)
                with self._lock:
                    self._total_errors += 1

            # Sleep in small increments to allow rapid responsive shutdown
            elapsed = 0.0
            sleep_chunk = 0.2
            while elapsed < self.interval_seconds and not self._stop_event.is_set():
                time.sleep(sleep_chunk)
                elapsed += sleep_chunk

        logger.info("[ASYNC_WORKER] Background evaluation thread stopped gracefully.")

    def start(self) -> None:
        """Start the background evaluation daemon thread."""
        with self._lock:
            if self.is_running:
                logger.warning("[ASYNC_WORKER] Worker thread is already running.")
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="BotShield-AsyncEvaluationWorker",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the worker to stop and wait for thread to terminate."""
        with self._lock:
            if not self.is_running:
                return
            self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)

    def get_status(self) -> Dict[str, Any]:
        """Return operational telemetry metrics for worker health monitoring."""
        with self._lock:
            return {
                "running": self.is_running,
                "interval_seconds": self.interval_seconds,
                "batch_size": self.batch_size,
                "total_cycles": self._total_cycles,
                "total_evaluated": self._total_evaluated,
                "total_errors": self._total_errors,
                "last_run_time": self._last_run_time,
                "last_batch_duration_ms": self._last_batch_duration_ms,
            }


_GLOBAL_WORKER: Optional[AsyncEvaluationWorker] = None
_GLOBAL_LOCK = threading.Lock()


def init_app_background_worker(
    app: Any,
    interval_seconds: float = 5.0,
    batch_size: int = 50,
) -> Optional[AsyncEvaluationWorker]:
    """
    Convenience initializer for embedding background evaluation in Flask.

    Respects app.config['ASYNC_EVALUATION_WORKER_ENABLED'] (default: True).
    """
    global _GLOBAL_WORKER
    if not app.config.get("ASYNC_EVALUATION_WORKER_ENABLED", True):
        logger.info("[ASYNC_WORKER] Background evaluation worker disabled by configuration.")
        return None

    with _GLOBAL_LOCK:
        if _GLOBAL_WORKER is not None and _GLOBAL_WORKER.is_running:
            return _GLOBAL_WORKER

        db_path = app.config.get("BOTSHIELD_DB")
        model_path = app.config.get("BOTSHIELD_MODEL_PATH")

        worker = AsyncEvaluationWorker(
            db_path=db_path,
            model_path=model_path,
            interval_seconds=interval_seconds,
            batch_size=batch_size,
        )
        worker.start()
        _GLOBAL_WORKER = worker
        return worker


def main() -> None:
    """CLI entry point for running the background risk evaluation worker."""
    parser = argparse.ArgumentParser(
        description="BotShield Asynchronous Session Risk Evaluation Worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds between evaluation cycles",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum candidate sessions to evaluate per cycle",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single evaluation cycle and exit immediately",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Optional custom SQLite database path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional custom Isolation Forest model artifact path",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    worker = AsyncEvaluationWorker(
        db_path=args.db,
        model_path=args.model,
        interval_seconds=args.interval,
        batch_size=args.batch_size,
    )

    if args.once:
        print(f"Running single evaluation cycle (batch-size: {args.batch_size})...")
        res = worker.run_once()
        print(f"Cycle completed in {res['duration_ms']}ms: {res['evaluated_count']} sessions evaluated ({res['errors_count']} errors).")
    else:
        print(f"Starting BotShield asynchronous evaluation worker (interval: {args.interval}s, batch-size: {args.batch_size})...")
        print("Press Ctrl+C to terminate.")
        worker.start()
        try:
            while worker.is_running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nShutting down worker...")
            worker.stop()
            print("Worker terminated cleanly.")


if __name__ == "__main__":
    main()
