"""
BotShield – Bounded Sliding-Window Rate Limiter (Phase 6)

Provides a thread-safe, memory-bounded, in-memory sliding-window rate limiter.
Tracks request timestamps per client key (session identifier or client IP)
and calculates exact Retry-After durations.

Multi-Worker Architectural Note:
This in-memory implementation operates on a per-process basis within the
active Flask application worker. In multi-worker deployments (e.g. Gunicorn/uWSGI
with multiple worker processes), rate-limiting state is maintained independently
in each worker unless backed by a shared distributed store (such as Redis).
For this application stage, this in-memory implementation provides zero external
dependencies, microsecond latency, deterministic behavior, and hard bounds
on memory usage.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from typing import Any, Callable, Deque, Dict, Optional, Protocol, Tuple


class RateLimiterBackend(Protocol):
    """Abstract protocol for pluggable rate limiter backends."""

    def check(
        self,
        key: str,
        max_requests: int,
        window_seconds: float,
        now: Optional[float] = None,
    ) -> Tuple[bool, int]:
        """Check whether client key exceeds quota."""
        ...

    def reset(self) -> None:
        """Clear all rate limit state."""
        ...

    @property
    def tracked_count(self) -> int:
        """Current number of tracked keys."""
        ...


class BoundedRateLimiter:
    """
    In-memory sliding-window rate limiter with thread synchronization and bounded capacity.
    """

    def __init__(self, max_tracked_clients: int = 5000) -> None:
        if max_tracked_clients <= 0:
            raise ValueError("max_tracked_clients must be positive.")
        self.max_tracked_clients = max_tracked_clients
        self._clients: dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def _prune_expired(self, now: float, window_seconds: float) -> None:
        """Internal helper to prune stale entries when capacity is approached."""
        cutoff = now - window_seconds
        stale_keys = [
            k for k, q in self._clients.items()
            if not q or q[-1] < cutoff
        ]
        for k in stale_keys:
            self._clients.pop(k, None)

        # If still over capacity, evict oldest entries to prevent memory exhaustion
        if len(self._clients) >= self.max_tracked_clients:
            excess = len(self._clients) - self.max_tracked_clients + 1
            keys_to_remove = list(self._clients.keys())[:excess]
            for k in keys_to_remove:
                self._clients.pop(k, None)

    def check(
        self,
        key: str,
        max_requests: int,
        window_seconds: float,
        now: float | None = None,
    ) -> tuple[bool, int]:
        """
        Record a request attempt and verify whether the client is rate-limited.

        Parameters:
            key: Unique client identifier (session ID or IP address).
            max_requests: Maximum allowed requests in the time window.
            window_seconds: Duration of the sliding window in seconds.
            now: Optional current timestamp (defaults to time.time()).

        Returns:
            (is_limited, retry_after_seconds)
            where is_limited is True if request exceeds limit, and retry_after_seconds
            is the integer duration until the oldest request in the window expires.
        """
        if not key or not isinstance(key, str):
            key = "unknown_client"

        current_time = time.time() if now is None else float(now)
        cutoff = current_time - window_seconds

        with self._lock:
            # Enforce bounded memory growth
            if key not in self._clients and len(self._clients) >= self.max_tracked_clients:
                self._prune_expired(current_time, window_seconds)

            if key not in self._clients:
                self._clients[key] = collections.deque()

            queue = self._clients[key]

            # Evict timestamps outside the active sliding window
            while queue and queue[0] <= cutoff:
                queue.popleft()

            if len(queue) >= max_requests:
                # Client has exhausted allowed quota
                oldest_ts = queue[0]
                retry_after = max(1, int(math.ceil(oldest_ts + window_seconds - current_time)))
                return True, retry_after

            # Within quota: record timestamp and permit request
            queue.append(current_time)
            return False, 0

    def reset(self) -> None:
        """Clear all tracked client states (primarily for testing)."""
        with self._lock:
            self._clients.clear()

    @property
    def tracked_count(self) -> int:
        """Return the current number of tracked clients."""
        with self._lock:
            return len(self._clients)


class DistributedRateLimiterAdapter:
    """
    Pluggable distributed sliding-window rate limiter adapter.

    Enables backing rate-limiting state by an external key-value store (e.g. Redis)
    or a custom shared driver for horizontal multi-worker WSGI deployments.
    """

    def __init__(self, storage: Any) -> None:
        """
        Initialize with a key-value or Redis-like storage client.
        The storage object can provide sliding_window_check or basic mapping.
        """
        self.storage = storage

    def check(
        self,
        key: str,
        max_requests: int,
        window_seconds: float,
        now: Optional[float] = None,
    ) -> Tuple[bool, int]:
        current_time = time.time() if now is None else float(now)
        cutoff = current_time - window_seconds

        # If storage provides an atomic sliding_window_check method:
        if hasattr(self.storage, "sliding_window_check"):
            return self.storage.sliding_window_check(key, max_requests, window_seconds, current_time)

        # Standard dictionary/mock-compatible sliding-window implementation
        if not hasattr(self.storage, "_store"):
            self.storage._store = {}

        store = self.storage._store
        if key not in store:
            store[key] = []

        timestamps = [ts for ts in store[key] if ts > cutoff]
        store[key] = timestamps

        if len(timestamps) >= max_requests:
            oldest_ts = timestamps[0]
            retry_after = max(1, int(math.ceil(oldest_ts + window_seconds - current_time)))
            return True, retry_after

        timestamps.append(current_time)
        return False, 0

    def reset(self) -> None:
        if hasattr(self.storage, "clear"):
            self.storage.clear()
        elif hasattr(self.storage, "_store"):
            self.storage._store.clear()

    @property
    def tracked_count(self) -> int:
        if hasattr(self.storage, "dbsize"):
            return self.storage.dbsize()
        if hasattr(self.storage, "_store"):
            return len(self.storage._store)
        return 0

