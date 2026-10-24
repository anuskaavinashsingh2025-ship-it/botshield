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

from backend.database.db import init_db  # noqa: E402


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path, monkeypatch):
    """Each test gets its own temporary SQLite database."""
    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("BOTSHIELD_DB", db_file)
    init_db(db_file)
    yield db_file
