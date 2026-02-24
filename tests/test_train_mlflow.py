"""Tests for Train + MLflow node."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from autoforge.nodes.train_mlflow import compute_metrics, log_model_artifacts, train_mlflow_node


@pytest.fixture
def classification_pipeline_state(tmp_path: Path) -> dict:
    """Create a sample classification pipeline state."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    x = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        }
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create simple pipeline
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    return {
        "best_pipeline": pipeline,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "task_type": "classification",
        "best_params": {"model": "rf", "n_estimators": 10, "max_depth": 5},
        "artifact_dir": str(tmp_path / "artifacts"),
        "tracking_uri": f"file://{(tmp_path / 'mlruns').resolve()}",
    }


@pytest.fixture
def regression_pipeline_state(tmp_path: Path) -> dict:
    """Create a sample regression pipeline state."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    x = pd.DataFrame(
        {
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        }
    )
    y = pd.Series(np.random.randn(n_samples) * 10 + 50)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Create simple pipeline
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=10, random_state=42)),
        ]
    )
    pipeline.fit(x_train, y_train)

    return {
        "best_pipeline": pipeline,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "task_type": "regression",
        "best_params": {"model": "rf", "n_estimators": 10, "max_depth": 5},
        "artifact_dir": str(tmp_path / "artifacts"),
        "tracking_uri": f"file://{(tmp_path / 'mlruns').resolve()}",
    }


def test_compute_metrics_classification() -> None:
    """Test metric computation for classification."""
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = compute_metrics(y_true, y_pred, "classification")

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_compute_metrics_regression() -> None:
    """Test metric computation for regression."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

    metrics = compute_metrics(y_true, y_pred, "regression")

    assert "mae" in metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0


def test_log_model_artifacts(tmp_path: Path) -> None:
    """Test saving model and metadata artifacts."""
    # Create simple model
    x = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=5, random_state=42)),
        ]
    )
    pipeline.fit(x, y)

    metrics = {"accuracy": 0.8}
    params = {"model": "rf"}

    artifact_dir = tmp_path / "artifacts"

    model_path, metadata_path = log_model_artifacts(pipeline, str(artifact_dir), metrics, params)

    # Verify files exist
    assert Path(model_path).exists()
    assert Path(metadata_path).exists()

    # Verify metadata structure
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert "best_params" in metadata
    assert "metrics" in metadata
    assert metadata["best_params"]["model"] == "rf"


def test_train_mlflow_node_classification(classification_pipeline_state: dict) -> None:
    """Test train_mlflow_node for classification."""
    result = train_mlflow_node(classification_pipeline_state)

    assert "mlflow_run_id" in result
    assert "train_metrics" in result
    assert "test_metrics" in result
    assert "model_path" in result
    assert "metadata_path" in result
    assert "train_metrics" in result and "accuracy" in result["train_metrics"]
    assert "test_metrics" in result and "accuracy" in result["test_metrics"]
    assert result["train_samples"] > 0
    assert result["test_samples"] > 0


def test_train_mlflow_node_regression(regression_pipeline_state: dict) -> None:
    """Test train_mlflow_node for regression."""
    result = train_mlflow_node(regression_pipeline_state)

    assert "mlflow_run_id" in result
    assert "train_metrics" in result
    assert "test_metrics" in result
    assert "model_path" in result
    assert "metadata_path" in result
    assert "train_metrics" in result and "mae" in result["train_metrics"]
    assert "test_metrics" in result and "mae" in result["test_metrics"]
    assert result["train_samples"] > 0
    assert result["test_samples"] > 0


def test_train_mlflow_node_missing_pipeline(classification_pipeline_state: dict) -> None:
    """Test error when pipeline missing."""
    state = {**classification_pipeline_state}
    del state["best_pipeline"]

    with pytest.raises(KeyError):
        train_mlflow_node(state)


def test_train_mlflow_node_missing_data(classification_pipeline_state: dict) -> None:
    """Test error when data missing."""
    state = {**classification_pipeline_state}
    del state["x_test"]

    with pytest.raises(KeyError):
        train_mlflow_node(state)


def test_train_mlflow_node_missing_task_type(classification_pipeline_state: dict) -> None:
    """Test error when task_type missing."""
    state = {**classification_pipeline_state}
    del state["task_type"]

    with pytest.raises(ValueError):
        train_mlflow_node(state)
