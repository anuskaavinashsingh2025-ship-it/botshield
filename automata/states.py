from enum import Enum, auto

class State(Enum):
    START = auto()
    NORMAL_BROWSING = auto()
    INTERACTIVE_BROWSING = auto()
    RAPID_REQUESTS = auto()
    REPEATED_REQUESTS = auto()
    SUSPICIOUS_BEHAVIOR = auto()
    HUMAN_LIKE = auto()
    BOT_DETECTED = auto()

    def __str__(self):
        return self.name
