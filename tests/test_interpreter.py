import pytest
from automata.interpreter import interpret_event, Symbol
from automata.history import SessionContext

def test_behavioral_events():
    ctx = SessionContext("test")
    assert interpret_event({"event_type": "PAGE_VIEW"}, ctx) == Symbol.PAGE_VIEW
    assert interpret_event({"event_type": "CLICK"}, ctx) == Symbol.CLICK
    assert interpret_event({"event_type": "PAGE_EXIT"}, ctx) == Symbol.WAIT

def test_failed_requests():
    ctx = SessionContext("test")
    assert interpret_event({"type": "REQUEST", "status_code": 404}, ctx) == Symbol.FAILED_REQUEST

def test_rapid_requests():
    ctx = SessionContext("test")
    # Base timestamp
    t0 = "2026-09-21T10:00:00Z"
    t1 = "2026-09-21T10:00:00.5Z"
    t2 = "2026-09-21T10:00:01.0Z"
    
    # 1st request -> normal
    assert interpret_event({"type": "REQUEST", "timestamp": t0, "endpoint": "/a"}, ctx) == Symbol.REQUEST
    # 2nd request -> normal
    assert interpret_event({"type": "REQUEST", "timestamp": t1, "endpoint": "/b"}, ctx) == Symbol.REQUEST
    # 3rd request within 2 seconds -> RAPID_REQUEST
    assert interpret_event({"type": "REQUEST", "timestamp": t2, "endpoint": "/c"}, ctx) == Symbol.RAPID_REQUEST

def test_repeated_requests():
    ctx = SessionContext("test")
    t0 = "2026-09-21T10:00:00Z"
    t1 = "2026-09-21T10:00:05Z"
    t2 = "2026-09-21T10:00:09Z"
    
    assert interpret_event({"type": "REQUEST", "timestamp": t0, "endpoint": "/products"}, ctx) == Symbol.REQUEST
    assert interpret_event({"type": "REQUEST", "timestamp": t1, "endpoint": "/products"}, ctx) == Symbol.REQUEST
    # 3rd request to same endpoint within 10s
    # Note: If it's also rapid (within 2s), RAPID has priority. But here it's spread out (5s, 4s).
    assert interpret_event({"type": "REQUEST", "timestamp": t2, "endpoint": "/products"}, ctx) == Symbol.REPEATED_REQUEST
