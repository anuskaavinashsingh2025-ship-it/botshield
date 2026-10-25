import pytest
from automata.states import State
from automata.history import get_session_context, clear_session_context
from automata.fsm import process_event

@pytest.fixture(autouse=True)
def reset_sessions():
    clear_session_context("human_session")
    clear_session_context("rapid_bot")
    clear_session_context("scraper")
    clear_session_context("strong_bot")
    clear_session_context("recovery")

def test_human_scenario():
    sid = "human_session"
    events = [
        {"event_type": "PAGE_VIEW"},
        {"event_type": "PAGE_EXIT"}, # WAIT
        {"event_type": "SCROLL"},
        {"event_type": "CLICK"},
        {"event_type": "PAGE_EXIT"}, # WAIT
        {"event_type": "MOUSE_ACTIVITY"},
        {"event_type": "PAGE_VIEW"},
        {"event_type": "SCROLL"},
        {"event_type": "CLICK"}
    ]
    
    last_res = None
    for evt in events:
        last_res = process_event(sid, evt)
        
    assert last_res["current_state"] == "HUMAN_LIKE"
    assert last_res["automata_score"] < 0.25

def test_rapid_bot_scenario():
    sid = "rapid_bot"
    events = [
        {"event_type": "PAGE_VIEW"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00Z", "endpoint": "/a"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00.5Z", "endpoint": "/b"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:01Z", "endpoint": "/c"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:01.5Z", "endpoint": "/d"},
    ]
    
    last_res = None
    for evt in events:
        last_res = process_event(sid, evt)
        
    assert last_res["current_state"] == "RAPID_REQUESTS"
    assert last_res["automata_score"] >= 0.35

def test_scraper_scenario():
    sid = "scraper"
    events = [
        {"event_type": "PAGE_VIEW"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00Z", "endpoint": "/products"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:05Z", "endpoint": "/products"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:09Z", "endpoint": "/products"}, # Repeated
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:10Z", "endpoint": "/products"}, # Rapid & Repeated
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:10.5Z", "endpoint": "/products"}, # Rapid & Repeated
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:11Z", "endpoint": "/products"}, # Rapid & Repeated
    ]
    
    last_res = None
    for evt in events:
        last_res = process_event(sid, evt)
        
    # The scraper should hit REPEATED_REQUESTS and eventually SUSPICIOUS_BEHAVIOR
    assert last_res["current_state"] in ("REPEATED_REQUESTS", "SUSPICIOUS_BEHAVIOR", "BOT_DETECTED")
    assert last_res["automata_score"] >= 0.50

def test_strong_bot_scenario():
    sid = "strong_bot"
    events = [
        {"event_type": "PAGE_VIEW"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00Z", "endpoint": "/products"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00.1Z", "endpoint": "/products"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00.2Z", "endpoint": "/products"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00.3Z", "endpoint": "/products", "status_code": 404},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00.4Z", "endpoint": "/products", "status_code": 404},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00.5Z", "endpoint": "/products", "status_code": 404},
    ]
    
    last_res = None
    for evt in events:
        last_res = process_event(sid, evt)
        
    assert last_res["current_state"] == "BOT_DETECTED"
    assert last_res["automata_score"] >= 0.90

def test_recovery_scenario():
    sid = "recovery"
    # First a rapid burst
    events = [
        {"event_type": "PAGE_VIEW"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00Z", "endpoint": "/a"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:00.5Z", "endpoint": "/b"},
        {"type": "REQUEST", "timestamp": "2026-01-01T10:00:01Z", "endpoint": "/c"},
    ]
    
    for evt in events:
        res = process_event(sid, evt)
        
    assert res["current_state"] == "RAPID_REQUESTS"
    
    # Now sustained normal interaction
    recovery_events = [
        {"event_type": "SCROLL"},
        {"event_type": "CLICK"},
        {"event_type": "PAGE_EXIT"},
        {"event_type": "PAGE_VIEW"},
        {"event_type": "MOUSE_ACTIVITY"},
        {"event_type": "CLICK"}
    ]
    
    for evt in recovery_events:
        res = process_event(sid, evt)
        
    # Should recover back to INTERACTIVE_BROWSING or HUMAN_LIKE
    assert res["current_state"] in ("INTERACTIVE_BROWSING", "HUMAN_LIKE")
    assert res["automata_score"] < 0.50
