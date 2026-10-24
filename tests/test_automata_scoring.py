import pytest
from automata.history import SessionContext
from automata.states import State
from automata.scoring import calculate_score

def test_score_boundaries():
    ctx = SessionContext("test")
    
    # START state
    ctx.current_state = State.START
    score, reasons = calculate_score(ctx)
    assert score == 0.0
    assert "START" in "".join(reasons)
    
    # NORMAL_BROWSING state
    ctx.current_state = State.NORMAL_BROWSING
    score, reasons = calculate_score(ctx)
    assert 0.0 < score <= 0.20
    
    # BOT_DETECTED state
    ctx.current_state = State.BOT_DETECTED
    score, reasons = calculate_score(ctx)
    assert score >= 0.90
    
def test_score_modifiers():
    ctx = SessionContext("test")
    ctx.current_state = State.RAPID_REQUESTS
    ctx.rapid_count = 5
    ctx.interaction_count = 0
    score1, _ = calculate_score(ctx)
    
    # If interactions increase, score should decrease
    ctx.interaction_count = 10
    score2, _ = calculate_score(ctx)
    assert score2 < score1
