"""Tests for Model Registry + Export node."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoforge.nodes.model_registry import (
    create_model_info_card,
    export_model_formats,
    model_registry_node,
    register_model_in_registry,
)


@pytest.fixture
def trained_model() -> Pipeline:
    """Create a simple trained model."""
    x = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]
    )
    pipeline.fit(x, y)
    return pipeline


@pytest.fixture
def model_registry_state(tmp_path: Path, trained_model: Pipeline) -> dict:
    """Create a sample model registry state."""
    np.random.seed(42)
    n_samples = 50
    x_test = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        }
    )
    y_test = pd.Series(np.random.randint(0, 2, n_samples))

    return {
        "best_pipeline": trained_model,
        "x_test": x_test,
        "y_test": y_test,
        "task_type": "classification",
        "best_params": {"model": "rf", "n_estimators": 5},
        "test_metrics": {"accuracy": 0.85, "f1": 0.82},
        "train_metrics": {"accuracy": 0.90, "f1": 0.88},
        "feature_columns": ["feature1", "feature2"],
        "mlflow_run_id": "test_run_id",
        "model_name": "test_model",
        "export_dir": str(tmp_path / "exports"),
    }


def test_export_model_formats(trained_model: Pipeline, tmp_path: Path) -> None:
    """Test exporting model to multiple formats."""
    export_dir = tmp_path / "exports"
    exports = export_model_formats(trained_model, str(export_dir), "test_model")

    assert "joblib" in exports
    assert "sklearn" in exports
    assert Path(exports["joblib"]).exists()


def test_export_model_formats_creates_directory(trained_model: Pipeline, tmp_path: Path) -> None:
    """Test that export directory is created if it doesn't exist."""
    export_dir = tmp_path / "new_export_dir"
    assert not export_dir.exists()

    export_model_formats(trained_model, str(export_dir), "test_model")

    assert export_dir.exists()


def test_register_model_in_registry() -> None:
    """Test model registry registration."""
    result = register_model_in_registry(
        "test_model",
        "runs:/test_run_id/model",
        description="Test model",
        tags={"test": "true"},
    )

    assert "model_name" in result
    assert "version" in result
    assert "uri" in result
    assert result["model_name"] == "test_model"


def test_create_model_info_card() -> None:
    """Test model info card creation."""
    params = {"n_estimators": 100, "max_depth": 5}
    metrics = {"accuracy": 0.95}
    features = ["feature1", "feature2"]

    card = create_model_info_card(
        "test_model",
        "classification",
        params,
        metrics,
        features,
    )

    assert card["model_name"] == "test_model"
    assert card["task_type"] == "classification"
    assert card["hyperparameters"] == params
    assert card["metrics"] == metrics
    assert card["feature_names"] == features
    assert card["framework"] == "scikit-learn"


def test_create_model_info_card_without_features() -> None:
    """Test model info card creation without features."""
    card = create_model_info_card(
        "test_model",
        "regression",
        {"n_estimators": 50},
        {"r2": 0.92},
        feature_names=None,
    )

    assert card["feature_names"] == []


def test_model_registry_node(model_registry_state: dict) -> None:
    """Test model_registry_node end-to-end."""
    result = model_registry_node(model_registry_state)

    assert "export_formats" in result
    assert "model_card" in result
    assert "model_card_path" in result
    assert "registry_info" in result
    assert "deployment_ready" in result
    assert result["deployment_ready"] is True

    # Verify files exist
    assert Path(result["model_card_path"]).exists()

    # Verify model card content
    with open(result["model_card_path"]) as f:
        card_data = json.load(f)
    assert card_data["model_name"] == "test_model"
    assert card_data["task_type"] == "classification"


def test_model_registry_node_exports_joblib(model_registry_state: dict) -> None:
    """Test that joblib format is always exported."""
    result = model_registry_node(model_registry_state)

    assert "joblib" in result["export_formats"]
    joblib_path = result["export_formats"]["joblib"]
    assert Path(joblib_path).exists()


def test_model_registry_node_missing_pipeline(model_registry_state: dict) -> None:
    """Test error when pipeline missing."""
    state = {**model_registry_state}
    del state["best_pipeline"]

    with pytest.raises(KeyError):
        model_registry_node(state)


def test_model_registry_node_missing_task_type(model_registry_state: dict) -> None:
    """Test error when task_type missing."""
    state = {**model_registry_state}
    del state["task_type"]

    with pytest.raises(ValueError):
        model_registry_node(state)


def test_model_registry_node_default_model_name(
    model_registry_state: dict,
) -> None:
    """Test default model name is used if not provided."""
    state = {**model_registry_state}
    del state["model_name"]

    result = model_registry_node(state)

    assert result["model_name"] == "autoforge-model"


def test_model_registry_node_custom_names(model_registry_state: dict) -> None:
    """Test custom model name and export directory."""
    state = {**model_registry_state}
    state["model_name"] = "custom_model"
    state["export_dir"] = state["export_dir"].replace("exports", "custom_exports")

    result = model_registry_node(state)

    assert result["model_name"] == "custom_model"
    assert Path(result["export_dir"]).exists()
