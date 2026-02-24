"""Training and MLflow experiment tracking node.

Handles model training, evaluation, and MLflow experiment logging for AutoML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from autoforge.types import AutoForgeState


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    task_type: str,
) -> dict[str, float]:
    """Compute evaluation metrics based on task type.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        task_type: 'classification' or 'regression'.

    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}

    if task_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    else:  # regression
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["r2"] = float(r2_score(y_true, y_pred))

    return metrics


def log_model_artifacts(
    model: Pipeline,
    artifact_dir: str,
    metrics: dict[str, float],
    best_params: dict[str, Any],
) -> tuple[str, str]:
    """Save model and metadata to disk and log to MLflow.

    Args:
        model: Trained pipeline.
        artifact_dir: Directory to save artifacts.
        metrics: Dictionary of metrics.
        best_params: Best hyperparameters from optimization.

    Returns:
        Tuple of (model_path, metadata_path).
    """
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = artifact_path / "model.joblib"
    joblib.dump(model, model_path)

    # Log model to MLflow
    mlflow.sklearn.log_model(model, "model", input_example=None)

    # Save metadata
    metadata = {
        "best_params": best_params,
        "metrics": metrics,
        "model_path": str(model_path),
    }

    import json

    metadata_path = artifact_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(model_path), str(metadata_path)


def train_mlflow_node(state: AutoForgeState) -> AutoForgeState:
    """Train model and log experiments with MLflow.

    This node:
    - Retrieves trained pipeline from previous nodes
    - Evaluates on full training + test sets
    - Logs metrics, parameters, and model to MLflow
    - Saves artifacts (model, metadata)
    - Returns results summary

    Args:
        state: AutoForgeState containing pipeline, data, and task info.

    Returns:
        Updated state with MLflow run info, metrics, and artifact paths.

    Raises:
        KeyError: If required state keys missing.
        ValueError: If data validation fails.
    """
    best_pipeline = state.get("best_pipeline")
    x_train = state.get("x_train")
    y_train = state.get("y_train")
    x_test = state.get("x_test")
    y_test = state.get("y_test")
    task_type = state.get("task_type")
    best_params = state.get("best_params", {})
    artifact_dir = state.get("artifact_dir", "artifacts")
    tracking_uri = state.get("tracking_uri", "file:./mlruns")

    if best_pipeline is None:
        raise KeyError("best_pipeline required in state")
    if x_train is None or y_train is None or x_test is None or y_test is None:
        raise KeyError("x_train, y_train, x_test, y_test required in state")
    if task_type is None:
        raise ValueError("task_type required in state")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("autoforge_automl")

    with mlflow.start_run():
        # Log hyperparameters
        for key, value in best_params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)

        # Evaluate on training set
        y_train_pred = best_pipeline.predict(x_train)
        train_metrics = compute_metrics(y_train, y_train_pred, task_type)

        # Evaluate on test set
        y_test_pred = best_pipeline.predict(x_test)
        test_metrics = compute_metrics(y_test, y_test_pred, task_type)

        # Log metrics
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        # Log additional info
        mlflow.log_metric("train_samples", len(x_train))
        mlflow.log_metric("test_samples", len(x_test))
        mlflow.log_metric("n_features", x_train.shape[1])

        # Save and log artifacts
        model_path, metadata_path = log_model_artifacts(
            best_pipeline,
            artifact_dir,
            {**train_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}},
            best_params,
        )

        # Get run info
        run = mlflow.active_run()
        run_id = run.info.run_id if run else "unknown"

        return {
            **state,
            "mlflow_run_id": run_id,
            "mlflow_tracking_uri": tracking_uri,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "train_samples": len(x_train),
            "test_samples": len(x_test),
        }
