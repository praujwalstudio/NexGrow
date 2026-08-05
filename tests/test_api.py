"""Tests for the FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient

from server.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_pipeline():
    import server.main as api

    api._pipeline = None
    yield
    api._pipeline = None


def test_health_reports_model_unavailable(monkeypatch):
    monkeypatch.setattr("server.main.MODEL_PATH", "models/nonexistent.pth")

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is False


def test_models_endpoint_lists_taxonomy():
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["diseases"]) == 15
    assert body["severity_levels"] == ["Low", "Moderate", "High"]


def test_diagnose_returns_503_without_model(monkeypatch):
    monkeypatch.setattr("server.main.MODEL_PATH", "models/nonexistent.pth")

    resp = client.post(
        "/diagnose",
        files={"image": ("leaf.jpg", b"not-an-image", "image/jpeg")},
    )
    assert resp.status_code == 503
