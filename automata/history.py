from typing import List, Dict, Any
from datetime import datetime, timezone
from automata.states import State

class SessionContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_state = State.START
        self.history: List[Dict[str, Any]] = []
        
        # Internal context for tracking behaviors
        self.interaction_count = 0
        self.failed_request_count = 0
        
    def add_transition(self, event_symbol: str, next_state: State, reason: str = ""):
        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": str(self.current_state),
            "event": str(event_symbol),
            "to": str(next_state),
            "reason": reason
        })
        self.current_state = next_state


# Global in-memory storage for active sessions (for Step 5 implementation)
_ACTIVE_SESSIONS: Dict[str, SessionContext] = {}

def get_session_context(session_id: str) -> SessionContext:
    if session_id not in _ACTIVE_SESSIONS:
        _ACTIVE_SESSIONS[session_id] = SessionContext(session_id)
    return _ACTIVE_SESSIONS[session_id]

def find_session_context(session_id: str) -> SessionContext | None:
    """Return the session's context if it exists, without creating one."""
    return _ACTIVE_SESSIONS.get(session_id)

def clear_session_context(session_id: str):
    if session_id in _ACTIVE_SESSIONS:
        del _ACTIVE_SESSIONS[session_id]
