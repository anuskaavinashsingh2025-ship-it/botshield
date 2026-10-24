from datetime import datetime, timezone
from collections import deque
from automata.symbols import Symbol
from automata.history import SessionContext

# Configurable thresholds
RAPID_REQUEST_WINDOW_SEC = 2.0
RAPID_REQUEST_COUNT = 3

REPEATED_REQUEST_WINDOW_SEC = 10.0
REPEATED_REQUEST_COUNT = 3

def _parse_time(t_str: str) -> float:
    # Handle ISO format strings
    try:
        dt = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return 0.0

def interpret_event(raw_event: dict, context: SessionContext) -> Symbol:
    """
    Convert a raw telemetry event or HTTP request into a higher-level Symbol.
    """
    event_type = raw_event.get("event_type") or raw_event.get("type", "")
    
    # --- Behavioral Events ---
    if event_type == "PAGE_VIEW":
        return Symbol.PAGE_VIEW
    elif event_type == "CLICK":
        return Symbol.CLICK
    elif event_type == "SCROLL":
        return Symbol.SCROLL
    elif event_type == "MOUSE_ACTIVITY":
        return Symbol.MOUSE_ACTIVITY
    elif event_type == "KEYBOARD_ACTIVITY":
        return Symbol.KEYBOARD_ACTIVITY
    elif event_type == "PAGE_EXIT":
        return Symbol.WAIT
        
    # --- HTTP Request Events ---
    if event_type == "REQUEST":
        endpoint = raw_event.get("endpoint", "")
        status_code = raw_event.get("status_code", 200)
        timestamp_str = raw_event.get("timestamp", "")
        ts = _parse_time(timestamp_str)
        
        if status_code >= 400:
            return Symbol.FAILED_REQUEST
            
        if "login" in endpoint.lower():
            return Symbol.LOGIN_ATTEMPT

        # Tracking for rolling windows
        if not hasattr(context, "request_timestamps"):
            context.request_timestamps = deque()
            context.request_endpoints = deque()
            
        context.request_timestamps.append(ts)
        context.request_endpoints.append((ts, endpoint))
        
        # Prune old timestamps
        while context.request_timestamps and (ts - context.request_timestamps[0] > RAPID_REQUEST_WINDOW_SEC):
            context.request_timestamps.popleft()
            
        while context.request_endpoints and (ts - context.request_endpoints[0][0] > REPEATED_REQUEST_WINDOW_SEC):
            context.request_endpoints.popleft()
            
        # Detect Rapid Requests
        if len(context.request_timestamps) >= RAPID_REQUEST_COUNT:
            return Symbol.RAPID_REQUEST
            
        # Detect Repeated Requests
        # Check if the exact same endpoint occurs REPEATED_REQUEST_COUNT times
        endpoints_in_window = [e for _, e in context.request_endpoints]
        if endpoints_in_window.count(endpoint) >= REPEATED_REQUEST_COUNT:
            return Symbol.REPEATED_REQUEST
            
        return Symbol.REQUEST
        
    # Default fallback
    return Symbol.WAIT
