"""
BotShield – Behavioral Feature Extraction (Phase 4)

Extracts a deterministic, fixed-length behavioral feature vector from normalized
browser telemetry events and server-observed HTTP requests for a single session.

Feature Schema Version: 1.0.0
Number of Features: 21
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("botshield.ml.features")

FEATURE_SCHEMA_VERSION = "1.0.0"

FEATURE_NAMES: tuple[str, ...] = (
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
)

# Feature upper bounds to guard against adversarial/overflow inputs
_BOUNDS: dict[str, tuple[float, float]] = {
    "request_count": (0.0, 10_000_000.0),
    "telemetry_count": (0.0, 10_000_000.0),
    "total_activity_count": (0.0, 20_000_000.0),
    "session_duration_seconds": (0.0, 2_592_000.0),  # Max 30 days
    "request_rate": (0.0, 10_000.0),
    "telemetry_rate": (0.0, 10_000.0),
    "telemetry_to_request_ratio": (0.0, 100.0),
    "request_to_total_ratio": (0.0, 1.0),
    "unique_path_count": (0.0, 100_000.0),
    "path_repetition_ratio": (0.0, 1.0),
    "page_view_count": (0.0, 1_000_000.0),
    "click_count": (0.0, 1_000_000.0),
    "scroll_count": (0.0, 1_000_000.0),
    "mouse_activity_count": (0.0, 1_000_000.0),
    "keyboard_activity_count": (0.0, 1_000_000.0),
    "page_exit_count": (0.0, 1_000_000.0),
    "mean_inter_event_time": (0.0, 3600.0),
    "std_inter_event_time": (0.0, 3600.0),
    "min_inter_event_time": (0.0, 3600.0),
    "rapid_traversal_count": (0.0, 1_000_000.0),
    "error_status_ratio": (0.0, 1.0),
}

# Detailed metadata contract for all 21 features
FEATURE_CONTRACT: dict[str, dict[str, Any]] = {
    name: {
        "index": idx,
        "name": name,
        "type": "float",
        "bounds": _BOUNDS[name],
        "description": {
            "request_count": "Total server-observed HTTP requests in the session.",
            "telemetry_count": "Total browser telemetry events in the session.",
            "total_activity_count": "Sum of requests and telemetry events.",
            "session_duration_seconds": "Elapsed time in seconds between first and last recorded activity.",
            "request_rate": "Requests per second over effective session duration.",
            "telemetry_rate": "Telemetry events per second over effective session duration.",
            "telemetry_to_request_ratio": "Ratio of telemetry events to requests (bots often lack telemetry).",
            "request_to_total_ratio": "Proportion of total session activity represented by HTTP requests.",
            "unique_path_count": "Number of distinct URL paths visited.",
            "path_repetition_ratio": "Fraction of path visits that repeat previously visited paths.",
            "page_view_count": "Number of PAGE_VIEW telemetry events.",
            "click_count": "Number of CLICK telemetry events.",
            "scroll_count": "Number of SCROLL telemetry events.",
            "mouse_activity_count": "Number of MOUSE_ACTIVITY telemetry events.",
            "keyboard_activity_count": "Number of KEYBOARD_ACTIVITY telemetry events.",
            "page_exit_count": "Number of PAGE_EXIT telemetry events.",
            "mean_inter_event_time": "Mean time delta in seconds between consecutive activities.",
            "std_inter_event_time": "Standard deviation of time intervals between activities.",
            "min_inter_event_time": "Minimum observed time interval in seconds between activities.",
            "rapid_traversal_count": "Count of page transitions occurring in under 150 milliseconds.",
            "error_status_ratio": "Proportion of HTTP requests returning status codes >= 400.",
        }.get(name, ""),
    }
    for idx, name in enumerate(FEATURE_NAMES)
}


@dataclass(frozen=True)
class ExtractedFeatures:
    """Immutable representation of extracted behavioral session features."""
    feature_vector: list[float]
    feature_names: tuple[str, ...]
    feature_dict: dict[str, float]
    schema_version: str = FEATURE_SCHEMA_VERSION
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_vector": self.feature_vector,
            "feature_names": list(self.feature_names),
            "feature_dict": self.feature_dict,
            "warnings": self.warnings,
        }


def _parse_timestamp(ts: Any) -> float | None:
    """Parse any timestamp representation into a UTC epoch timestamp."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        val = float(ts)
        return val if not math.isnan(val) and not math.isinf(val) else None
    if isinstance(ts, str):
        cleaned = ts.strip()
        if not cleaned:
            return None
        # Try numeric string
        try:
            val = float(cleaned)
            return val if not math.isnan(val) and not math.isinf(val) else None
        except ValueError:
            pass
        # Try ISO8601
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(cleaned, fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
            return dt.timestamp()
        except ValueError:
            return None
    return None


def _clean_path(path: Any) -> str:
    """Normalize and clean URL paths without query strings or fragments."""
    if not isinstance(path, str):
        return ""
    cleaned = path.strip()
    if not cleaned:
        return ""
    if "?" in cleaned:
        cleaned = cleaned.split("?", 1)[0]
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0]
    cleaned = "/" + cleaned.lstrip("/")
    while "//" in cleaned:
        cleaned = cleaned.replace("//", "/")
    if len(cleaned) > 1 and cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    return cleaned


def _safe_float(val: float, name: str, warnings: list[str]) -> float:
    """Check for NaN/Inf and clamp value to defined feature boundaries."""
    if math.isnan(val) or math.isinf(val):
        warnings.append(f"WARNING_INVALID_VALUE_{name}")
        val = 0.0
    min_val, max_val = _BOUNDS.get(name, (0.0, 1e9))
    if val < min_val:
        warnings.append(f"WARNING_CLAMPED_MIN_{name}")
        return min_val
    if val > max_val:
        warnings.append(f"WARNING_CLAMPED_MAX_{name}")
        return max_val
    return float(val)


def extract_session_features(
    events: list[Any] | None = None,
    requests: list[Any] | None = None,
) -> ExtractedFeatures:
    """
    Extract a deterministic, fixed-length 21-element behavioral feature vector.

    Accepts lists of SQLite Row objects, dicts, or FSMEvent objects.
    Guarantees no NaN or Infinite values under any input conditions.
    """
    warnings: list[str] = []
    events_list = events or []
    requests_list = requests or []

    # 1. Process server-observed requests
    parsed_requests: list[dict[str, Any]] = []
    seen_req_signatures: set[tuple[str, str, int, float]] = set()

    for r in requests_list:
        r_id = r["id"] if hasattr(r, "keys") and "id" in r.keys() else 0
        method = str(r["method"]).upper() if hasattr(r, "keys") and "method" in r.keys() else (getattr(r, "method", "") or "")
        raw_path = str(r["path"]) if hasattr(r, "keys") and "path" in r.keys() else (getattr(r, "path", "") or "")
        path = _clean_path(raw_path)
        status_code = int(r["status_code"]) if hasattr(r, "keys") and "status_code" in r.keys() and str(r["status_code"]).isdigit() else 200

        raw_ts = r["timestamp"] if hasattr(r, "keys") and "timestamp" in r.keys() else getattr(r, "timestamp", None)
        epoch = _parse_timestamp(raw_ts)
        if epoch is None:
            warnings.append(f"WARNING_MALFORMED_REQUEST_TIMESTAMP_{r_id}")
            epoch = 0.0

        # Deduplicate identical request tuples
        sig = (method, path, status_code, epoch)
        if sig in seen_req_signatures:
            warnings.append(f"WARNING_DUPLICATE_REQUEST_{r_id}")
            continue
        seen_req_signatures.add(sig)

        parsed_requests.append({
            "id": r_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "epoch": epoch,
            "source": "REQUEST",
        })

    # 2. Process client browser telemetry events
    parsed_events: list[dict[str, Any]] = []
    seen_event_signatures: set[tuple[str, str, float]] = set()

    for e in events_list:
        e_id = e["id"] if hasattr(e, "keys") and "id" in e.keys() else 0
        raw_type = str(e["event_type"]).upper() if hasattr(e, "keys") and "event_type" in e.keys() else (getattr(e, "event_type", "") or getattr(e, "raw_type", "")).upper()
        raw_page = str(e["page"]) if hasattr(e, "keys") and "page" in e.keys() else (getattr(e, "page", "") or getattr(e, "path", "") or "")
        page = _clean_path(raw_page)

        raw_ts = e["timestamp"] if hasattr(e, "keys") and "timestamp" in e.keys() else getattr(e, "timestamp", None)
        if raw_ts is None and hasattr(e, "epoch_timestamp"):
            raw_ts = getattr(e, "epoch_timestamp")

        epoch = _parse_timestamp(raw_ts)
        if epoch is None:
            warnings.append(f"WARNING_MALFORMED_EVENT_TIMESTAMP_{e_id}")
            epoch = 0.0

        sig = (raw_type, page, epoch)
        if sig in seen_event_signatures:
            warnings.append(f"WARNING_DUPLICATE_EVENT_{e_id}")
            continue
        seen_event_signatures.add(sig)

        parsed_events.append({
            "id": e_id,
            "event_type": raw_type,
            "page": page,
            "epoch": epoch,
            "source": "EVENT",
        })

    # 3. Combine chronological activities
    all_activities = []
    for req in parsed_requests:
        all_activities.append((req["epoch"], 1, req))
    for ev in parsed_events:
        all_activities.append((ev["epoch"], 2, ev))
    # Sort deterministically
    all_activities.sort(key=lambda item: (item[0], item[1]))

    req_count = len(parsed_requests)
    telem_count = len(parsed_events)
    total_count = req_count + telem_count

    if total_count == 0:
        warnings.append("WARNING_EMPTY_SESSION")

    # 4. Compute temporal statistics
    epochs = [item[0] for item in all_activities if item[0] > 0.0]
    if len(epochs) >= 2:
        session_duration = max(0.0, max(epochs) - min(epochs))
        intervals = [max(0.0, epochs[i] - epochs[i - 1]) for i in range(1, len(epochs))]
        mean_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        if len(intervals) > 1:
            variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
            std_interval = math.sqrt(variance)
        else:
            std_interval = 0.0
    else:
        session_duration = 0.0
        mean_interval = 0.0
        min_interval = 0.0
        std_interval = 0.0

    # 5. Compute rates and ratios
    effective_duration = max(session_duration, 1.0)
    request_rate = req_count / effective_duration if total_count > 0 else 0.0
    telem_rate = telem_count / effective_duration if total_count > 0 else 0.0

    telem_to_req_ratio = telem_count / max(req_count, 1) if req_count > 0 else (float(telem_count) if telem_count > 0 else 0.0)
    req_to_total_ratio = req_count / max(total_count, 1) if total_count > 0 else 0.0

    # 6. Compute path metrics
    all_paths = [r["path"] for r in parsed_requests if r["path"]] + [e["page"] for e in parsed_events if e["page"]]
    unique_paths = set(all_paths)
    unique_path_count = len(unique_paths)
    total_path_visits = len(all_paths)
    if total_path_visits > 0:
        path_repetition_ratio = 1.0 - (unique_path_count / total_path_visits)
    else:
        path_repetition_ratio = 0.0

    # 7. Compute event-type distribution
    page_view_count = sum(1 for e in parsed_events if e["event_type"] == "PAGE_VIEW")
    click_count = sum(1 for e in parsed_events if e["event_type"] == "CLICK")
    scroll_count = sum(1 for e in parsed_events if e["event_type"] == "SCROLL")
    mouse_activity_count = sum(1 for e in parsed_events if e["event_type"] == "MOUSE_ACTIVITY")
    keyboard_activity_count = sum(1 for e in parsed_events if e["event_type"] == "KEYBOARD_ACTIVITY")
    page_exit_count = sum(1 for e in parsed_events if e["event_type"] == "PAGE_EXIT")

    # 8. Compute rapid page traversals (dwell time < 150ms between distinct pages)
    rapid_traversal_count = 0
    page_transitions = []
    for item in all_activities:
        act = item[2]
        if act["source"] == "REQUEST" and act["path"]:
            page_transitions.append((act["epoch"], act["path"]))
        elif act["source"] == "EVENT" and act["event_type"] == "PAGE_VIEW" and act["page"]:
            page_transitions.append((act["epoch"], act["page"]))

    for i in range(1, len(page_transitions)):
        dt = page_transitions[i][0] - page_transitions[i - 1][0]
        # Only count if elapsed time is positive and under 150ms
        if 0.0 <= dt < 0.150:
            rapid_traversal_count += 1

    # 9. Compute error status ratio (HTTP 4xx and 5xx)
    error_count = sum(1 for r in parsed_requests if r["status_code"] >= 400)
    error_status_ratio = error_count / max(req_count, 1) if req_count > 0 else 0.0

    # 10. Assemble and clamp feature map
    raw_feature_map: dict[str, float] = {
        "request_count": float(req_count),
        "telemetry_count": float(telem_count),
        "total_activity_count": float(total_count),
        "session_duration_seconds": float(session_duration),
        "request_rate": float(request_rate),
        "telemetry_rate": float(telem_rate),
        "telemetry_to_request_ratio": float(telem_to_req_ratio),
        "request_to_total_ratio": float(req_to_total_ratio),
        "unique_path_count": float(unique_path_count),
        "path_repetition_ratio": float(path_repetition_ratio),
        "page_view_count": float(page_view_count),
        "click_count": float(click_count),
        "scroll_count": float(scroll_count),
        "mouse_activity_count": float(mouse_activity_count),
        "keyboard_activity_count": float(keyboard_activity_count),
        "page_exit_count": float(page_exit_count),
        "mean_inter_event_time": float(mean_interval),
        "std_inter_event_time": float(std_interval),
        "min_inter_event_time": float(min_interval),
        "rapid_traversal_count": float(rapid_traversal_count),
        "error_status_ratio": float(error_status_ratio),
    }

    feature_dict: dict[str, float] = {}
    feature_vector: list[float] = []

    for name in FEATURE_NAMES:
        val = _safe_float(raw_feature_map.get(name, 0.0), name, warnings)
        feature_dict[name] = val
        feature_vector.append(val)

    return ExtractedFeatures(
        feature_vector=feature_vector,
        feature_names=FEATURE_NAMES,
        feature_dict=feature_dict,
        schema_version=FEATURE_SCHEMA_VERSION,
        warnings=warnings,
    )
