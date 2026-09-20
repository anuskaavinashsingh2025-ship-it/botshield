"""
BotShield – Step 1 tests
Verify that all routes return HTTP 200.
"""

import sys
import os
import pytest

# Ensure the project root is on the path so `app` can be imported.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app  # noqa: E402


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200


def test_products(client):
    response = client.get("/products")
    assert response.status_code == 200


def test_product_detail(client):
    response = client.get("/product/1")
    assert response.status_code == 200


def test_login(client):
    response = client.get("/login")
    assert response.status_code == 200


def test_about(client):
    response = client.get("/about")
    assert response.status_code == 200


def test_product_not_found(client):
    """A non-existent product should return 404."""
    response = client.get("/product/9999")
    assert response.status_code == 404
