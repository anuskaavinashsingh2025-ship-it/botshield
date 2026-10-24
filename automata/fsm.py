import logging
from automata.states import State
from automata.symbols import Symbol
from automata.history import SessionContext, get_session_context
from automata.interpreter import interpret_event

logger = logging.getLogger("botshield.automata.fsm")

def transition(state: State, symbol: Symbol, context: SessionContext) -> State:
    """Deterministic finite state machine transition logic."""
    # Always track interaction and anomalies
    if symbol in (Symbol.CLICK, Symbol.SCROLL, Symbol.KEYBOARD_ACTIVITY, Symbol.MOUSE_ACTIVITY):
        context.interaction_count += 1
    elif symbol == Symbol.FAILED_REQUEST:
        context.failed_request_count += 1
        
    # Helper to count anomalies in context
    if not hasattr(context, 'rapid_count'):
        context.rapid_count = 0
        context.repeated_count = 0
        
    if symbol == Symbol.RAPID_REQUEST:
        context.rapid_count += 1
    if symbol == Symbol.REPEATED_REQUEST:
        context.repeated_count += 1

    # Recovery / Decay logic (if interactions outweigh anomalies recently)
    # Simple rule: a genuine interaction in a suspicious state can lower the suspicion slightly
    if symbol in (Symbol.CLICK, Symbol.SCROLL, Symbol.KEYBOARD_ACTIVITY):
        if state == State.RAPID_REQUESTS:
            return State.INTERACTIVE_BROWSING
        elif state == State.REPEATED_REQUESTS:
            # Drop down one level
            return State.RAPID_REQUESTS

    # State transitions
    if state == State.START:
        if symbol in (Symbol.PAGE_VIEW, Symbol.REQUEST):
            return State.NORMAL_BROWSING
        elif symbol in (Symbol.CLICK, Symbol.SCROLL, Symbol.MOUSE_ACTIVITY):
            return State.INTERACTIVE_BROWSING
            
    elif state == State.NORMAL_BROWSING:
        if symbol in (Symbol.CLICK, Symbol.SCROLL, Symbol.MOUSE_ACTIVITY, Symbol.KEYBOARD_ACTIVITY):
            return State.INTERACTIVE_BROWSING
        elif symbol == Symbol.RAPID_REQUEST:
            return State.RAPID_REQUESTS
        elif symbol == Symbol.REPEATED_REQUEST:
            return State.REPEATED_REQUESTS
            
    elif state == State.INTERACTIVE_BROWSING:
        if symbol in (Symbol.PAGE_VIEW, Symbol.REQUEST):
            if context.interaction_count >= 3:
                return State.HUMAN_LIKE
        elif symbol == Symbol.RAPID_REQUEST:
            return State.RAPID_REQUESTS
        elif symbol == Symbol.REPEATED_REQUEST:
            return State.REPEATED_REQUESTS
            
    elif state == State.HUMAN_LIKE:
        if symbol in (Symbol.CLICK, Symbol.SCROLL, Symbol.MOUSE_ACTIVITY, Symbol.KEYBOARD_ACTIVITY, Symbol.PAGE_VIEW, Symbol.REQUEST):
            return State.HUMAN_LIKE
        elif symbol == Symbol.RAPID_REQUEST:
            # Even human-like sessions can have a burst, but drop state
            return State.RAPID_REQUESTS

    elif state == State.RAPID_REQUESTS:
        if symbol == Symbol.REPEATED_REQUEST:
            return State.REPEATED_REQUESTS
        elif symbol == Symbol.FAILED_REQUEST:
            return State.SUSPICIOUS_BEHAVIOR
        elif symbol == Symbol.RAPID_REQUEST and context.rapid_count >= 3:
            return State.SUSPICIOUS_BEHAVIOR
            
    elif state == State.REPEATED_REQUESTS:
        if symbol in (Symbol.FAILED_REQUEST, Symbol.RAPID_REQUEST) or context.repeated_count >= 3:
            return State.SUSPICIOUS_BEHAVIOR
            
    elif state == State.SUSPICIOUS_BEHAVIOR:
        # If we have rapid + repeated + failed and low interactions
        if symbol in (Symbol.RAPID_REQUEST, Symbol.REPEATED_REQUEST, Symbol.FAILED_REQUEST):
            if (context.rapid_count > 0 and context.repeated_count > 0) or context.failed_request_count >= 2:
                if context.interaction_count < 5:
                    return State.BOT_DETECTED
                    
    elif state == State.BOT_DETECTED:
        # Terminal state for now
        return State.BOT_DETECTED

    return state


def process_event(session_id: str, raw_event: dict) -> dict:
    """
    Main entry point to test the FSM. 
    Processes a raw event, updates the FSM, and calculates the score.
    """
    context = get_session_context(session_id)
    prev_state = context.current_state
    
    # 1. Interpret raw event into symbol
    symbol = interpret_event(raw_event, context)
    
    # 2. Compute next state
    next_state = transition(prev_state, symbol, context)
    
    # 3. Update history
    reason = f"Received symbol {symbol.name}"
    if next_state != prev_state:
        context.add_transition(symbol.name, next_state, reason)
    
    # 4. Generate Score (we will import from scoring module)
    from automata.scoring import calculate_score
    score, reasons = calculate_score(context)
    
    return {
        "session_id": session_id,
        "previous_state": str(prev_state),
        "event": str(symbol),
        "current_state": str(next_state),
        "automata_score": score,
        "reasons": reasons
    }
