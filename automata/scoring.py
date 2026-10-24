from automata.states import State
from automata.history import SessionContext

def calculate_score(context: SessionContext) -> tuple[float, list[str]]:
    score = 0.0
    reasons = []

    # Base score by state
    if context.current_state == State.START:
        score = 0.0
        reasons.append("FSM is in START state")
    elif context.current_state == State.NORMAL_BROWSING:
        score = 0.10
        reasons.append("FSM is in NORMAL_BROWSING")
    elif context.current_state == State.INTERACTIVE_BROWSING:
        score = 0.05
        reasons.append("FSM is in INTERACTIVE_BROWSING")
    elif context.current_state == State.HUMAN_LIKE:
        score = 0.02
        reasons.append("FSM remains HUMAN_LIKE")
    elif context.current_state == State.RAPID_REQUESTS:
        score = 0.40
        reasons.append("FSM entered RAPID_REQUESTS")
    elif context.current_state == State.REPEATED_REQUESTS:
        score = 0.60
        reasons.append("FSM entered REPEATED_REQUESTS")
    elif context.current_state == State.SUSPICIOUS_BEHAVIOR:
        score = 0.80
        reasons.append("FSM entered SUSPICIOUS_BEHAVIOR")
    elif context.current_state == State.BOT_DETECTED:
        score = 0.95
        reasons.append("FSM entered BOT_DETECTED")

    # Modifiers based on context anomalies
    rapid_count = getattr(context, 'rapid_count', 0)
    repeated_count = getattr(context, 'repeated_count', 0)
    
    if rapid_count > 0:
        score += min(0.20, rapid_count * 0.05)
        reasons.append(f"{rapid_count} rapid requests detected")
        
    if repeated_count > 0:
        score += min(0.20, repeated_count * 0.05)
        reasons.append(f"same endpoint repeated {repeated_count} times")
        
    if getattr(context, 'failed_request_count', 0) > 0:
        fails = context.failed_request_count
        score += min(0.15, fails * 0.05)
        reasons.append(f"{fails} failed requests")
        
    # Recovery / human behavior modifiers
    interactions = getattr(context, 'interaction_count', 0)
    if interactions > 0:
        reduction = min(0.30, interactions * 0.02)
        score -= reduction
        if interactions > 5:
            reasons.append("multiple scroll interactions / meaningful click activity")
        else:
            reasons.append("normal interaction activity")
    else:
        if context.current_state in (State.SUSPICIOUS_BEHAVIOR, State.BOT_DETECTED):
            reasons.append("very low interaction activity")
            
    # Clamp score
    final_score = max(0.0, min(1.0, score))
    return round(final_score, 2), reasons
