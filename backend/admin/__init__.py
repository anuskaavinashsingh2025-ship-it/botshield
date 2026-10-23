"""
BotShield – Security Administrator Dashboard Package (Phase 7)

Exports administrative blueprint and authentication decorators.
"""

from backend.admin.auth import (
    get_configured_admin_token,
    require_admin_auth,
    verify_admin_token,
)
from backend.admin.routes import admin_bp

__all__ = [
    "admin_bp",
    "get_configured_admin_token",
    "require_admin_auth",
    "verify_admin_token",
]
