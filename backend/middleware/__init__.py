"""
BotShield – Middleware Package
"""

from backend.middleware.request_logger import init_request_logger

__all__ = ["init_request_logger"]
