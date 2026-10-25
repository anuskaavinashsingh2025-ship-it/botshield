"""
BotShield – shared test configuration.

Keeps every test off the real botshield.db.
"""

import os
import sys
import tempfile

import pytest

# Ensure the project root is on the path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# app.py calls init_db() at import time, so point it at a throwaway database
# before any test module imports the app.
os.environ["BOTSHIELD_DB"] = os.path.join(tempfile.mkdtemp(prefix="botshield-test-"), "import.db")

# Keep results independent of any locally trained models/isolation_forest.joblib;
# tests that need an active model train their own and pass its path explicitly.
os.environ.setdefault("BOTSHIELD_MODEL_PATH", os.path.join(tempfile.gettempdir(), "botshield-no-model.joblib"))

# Authentication tests must see production behavior unless a test opts in.
os.environ.pop("BOTSHIELD_DEMO_MODE", None)
os.environ.pop("BOTSHIELD_ENV", None)

from backend.database.db import init_db  # noqa: E402


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Each test gets its own temporary SQLite database."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("BOTSHIELD_DB", db_file)
    init_db(db_file)
    yield db_file


@pytest.fixture(autouse=True)
def _restore_app_config():
    """
    Undo per-test changes to the shared Flask app's config.

    Several fixtures set keys such as BOTSHIELD_DB or ENFORCEMENT_* on the
    global `app`; without this they leak into every later test.
    """
    flask_app = getattr(sys.modules.get("app"), "app", None)
    saved = dict(flask_app.config) if flask_app is not None else None
    yield
    if flask_app is not None:
        flask_app.config.clear()
        flask_app.config.update(saved)
