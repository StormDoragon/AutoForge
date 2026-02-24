"""Tests for FastAPI Deploy node."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoforge.nodes.fastapi_deploy import (
    PredictionRequest,
    create_fastapi_app,
    fastapi_deploy_node,
)


@pytest.fixture
def trained_classification_model(tmp_path: Path) -> str:
    """Create and save a trained classification model."""
    x = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]
    )
    pipeline.fit(x, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    return str(model_path)


@pytest.fixture
def trained_regression_model(tmp_path: Path) -> str:
    """Create and save a trained regression model."""
    x = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    y = pd.Series([15.0, 31.0, 33.0, 53.0, 55.0, 75.0])

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=5, random_state=42)),
        ]
    )
    pipeline.fit(x, y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    return str(model_path)


def test_create_fastapi_app_classification(trained_classification_model: str) -> None:
    """Test creating FastAPI app for classification."""
    app, model = create_fastapi_app(
        trained_classification_model,
        "classification",
        ["feature1", "feature2"],
    )

    assert app is not None
    assert model is not None
    assert app.task_type == "classification"
    assert app.feature_names == ["feature1", "feature2"]


def test_create_fastapi_app_regression(trained_regression_model: str) -> None:
    """Test creating FastAPI app for regression."""
    app, model = create_fastapi_app(
        trained_regression_model,
        "regression",
        ["feature1", "feature2"],
    )

    assert app is not None
    assert model is not None
    assert app.task_type == "regression"


def test_create_fastapi_app_model_not_found() -> None:
    """Test error when model file not found."""
    with pytest.raises(FileNotFoundError):
        create_fastapi_app("/nonexistent/model.joblib", "classification")


def test_health_endpoint(trained_classification_model: str) -> None:
    """Test health check endpoint."""
    app, _ = create_fastapi_app(trained_classification_model, "classification")
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_type"] == "classification"


def test_info_endpoint(trained_classification_model: str) -> None:
    """Test model info endpoint."""
    app, _ = create_fastapi_app(
        trained_classification_model,
        "classification",
        ["feature1", "feature2"],
    )
    client = TestClient(app)

    response = client.get("/info")

    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "classification"
    assert data["feature_count"] == 2
    assert data["feature_names"] == ["feature1", "feature2"]


def test_predict_endpoint_classification(trained_classification_model: str) -> None:
    """Test prediction endpoint for classification."""
    app, _ = create_fastapi_app(trained_classification_model, "classification")
    client = TestClient(app)

    request_data = {"records": [{"feature1": 1.0, "feature2": 8.0}]}
    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "n_samples" in data
    assert data["n_samples"] == 1
    assert isinstance(data["predictions"], list)


def test_predict_endpoint_regression(trained_regression_model: str) -> None:
    """Test prediction endpoint for regression."""
    app, _ = create_fastapi_app(trained_regression_model, "regression")
    client = TestClient(app)

    request_data = {"records": [{"feature1": 2.0, "feature2": 20.0}]}
    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert data["n_samples"] == 1
    assert isinstance(data["predictions"][0], (int, float))


def test_predict_batch(trained_classification_model: str) -> None:
    """Test batch prediction."""
    app, _ = create_fastapi_app(trained_classification_model, "classification")
    client = TestClient(app)

    request_data = {
        "records": [
            {"feature1": 1.0, "feature2": 8.0},
            {"feature1": 5.0, "feature2": 4.0},
        ]
    }
    response = client.post("/predict", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["n_samples"] == 2
    assert len(data["predictions"]) == 2


def test_predict_invalid_features(trained_classification_model: str) -> None:
    """Test prediction with invalid feature names."""
    app, _ = create_fastapi_app(trained_classification_model, "classification")
    client = TestClient(app)

    request_data = {"records": [{"invalid_feature": 1.0}]}
    response = client.post("/predict", json=request_data)

    # Should get an error because features don't match
    assert response.status_code == 400


def test_prediction_request_schema() -> None:
    """Test PredictionRequest schema validation."""
    # Valid request
    req = PredictionRequest(records=[{"a": 1}, {"b": 2}])
    assert len(req.records) == 2

    # Empty records should be allowed by schema (validation handled by endpoint)
    req_empty = PredictionRequest(records=[])
    assert len(req_empty.records) == 0


def test_fastapi_deploy_node(trained_classification_model: str) -> None:
    """Test fastapi_deploy_node end-to-end."""
    state = {
        "model_path": trained_classification_model,
        "task_type": "classification",
        "feature_columns": ["feature1", "feature2"],
    }

    result = fastapi_deploy_node(state)

    assert "fastapi_app" in result
    assert "deployment_app" in result
    assert result["model_loaded"] is True
    assert "api_endpoints" in result
    assert "/health" in result["api_endpoints"]
    assert "/predict" in result["api_endpoints"]
    assert result["deployment_status"] == "ready"
    assert result["server_port"] == 8000


def test_fastapi_deploy_node_missing_model_path() -> None:
    """Test error when model_path missing."""
    state = {
        "task_type": "classification",
    }

    with pytest.raises(KeyError):
        fastapi_deploy_node(state)


def test_fastapi_deploy_node_missing_task_type(trained_classification_model: str) -> None:
    """Test error when task_type missing."""
    state = {
        "model_path": trained_classification_model,
    }

    with pytest.raises(ValueError):
        fastapi_deploy_node(state)


def test_fastapi_deploy_node_default_features(trained_classification_model: str) -> None:
    """Test deployment with default (empty) feature list."""
    state = {
        "model_path": trained_classification_model,
        "task_type": "classification",
    }

    result = fastapi_deploy_node(state)

    assert result["model_loaded"] is True
    assert result["deployment_status"] == "ready"


def test_docs_endpoint(trained_classification_model: str) -> None:
    """Test that OpenAPI docs are available."""
    app, _ = create_fastapi_app(trained_classification_model, "classification")
    client = TestClient(app)

    # FastAPI auto-generates docs
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
