from enum import Enum, auto

class Symbol(Enum):
    PAGE_VIEW = auto()
    CLICK = auto()
    SCROLL = auto()
    MOUSE_ACTIVITY = auto()
    KEYBOARD_ACTIVITY = auto()
    WAIT = auto()
    REQUEST = auto()
    RAPID_REQUEST = auto()
    REPEATED_REQUEST = auto()
    LOGIN_ATTEMPT = auto()
    FAILED_REQUEST = auto()

    def __str__(self):
        return self.name
