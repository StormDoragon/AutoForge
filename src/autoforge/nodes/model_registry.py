"""Model Registry and Export node.

Handles model registration, versioning, exporting to multiple formats,
and deployment-ready artifact packaging for AutoML models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn

from autoforge.types import AutoForgeState


def export_model_formats(
    model: Any,
    export_dir: str,
    model_name: str = "autoforge_model",
) -> dict[str, str]:
    """Export model to multiple formats for deployment.

    Args:
        model: Trained sklearn pipeline.
        export_dir: Directory to save exported models.
        model_name: Name of the model for exports.

    Returns:
        Dictionary mapping format names to file paths.
    """
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    exports = {}

    # Export as joblib (default pickle format for sklearn)
    joblib_path = export_path / f"{model_name}.joblib"
    joblib.dump(model, joblib_path)
    exports["joblib"] = str(joblib_path)

    # Try to export as sklearn format (MLflow native)
    try:
        sklearn_path = export_path / f"{model_name}_sklearn"
        mlflow.sklearn.save_model(model, str(sklearn_path), input_example=None)
        exports["sklearn"] = str(sklearn_path)
    except Exception:
        pass

    return exports


def register_model_in_registry(
    model_name: str,
    model_uri: str,
    description: str = "",
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Register model in MLflow Model Registry.

    Args:
        model_name: Name for the registered model.
        model_uri: MLflow URI of the model artifact (e.g., "runs:/<run_id>/model").
        description: Description of the model.
        tags: Dictionary of tags to attach to model.

    Returns:
        Dictionary with registration details (name, version, uri).
    """
    tags = tags or {}

    try:
        # Register model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version
        name = result.name

        # Update metadata
        if description:
            mlflow.set_model_tag(
                model_name,
                "description",
                description,
            )

        for tag_key, tag_value in tags.items():
            mlflow.set_model_tag(model_name, tag_key, tag_value)

        return {
            "model_name": name,
            "version": version,
            "uri": model_uri,
            "stage": "None",
        }
    except Exception as e:
        # Model may already exist, return version info
        return {
            "model_name": model_name,
            "version": "unknown",
            "uri": model_uri,
            "error": str(e),
        }


def create_model_info_card(
    model_name: str,
    task_type: str,
    best_params: dict[str, Any],
    metrics: dict[str, float],
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Create metadata card for the model.

    Args:
        model_name: Name of the model.
        task_type: 'classification' or 'regression'.
        best_params: Hyperparameters used.
        metrics: Model performance metrics.
        feature_names: List of input feature names.

    Returns:
        Dictionary with comprehensive model information.
    """
    return {
        "model_name": model_name,
        "task_type": task_type,
        "hyperparameters": best_params,
        "metrics": metrics,
        "feature_names": feature_names or [],
        "framework": "scikit-learn",
    }


def model_registry_node(state: AutoForgeState) -> AutoForgeState:
    """Register and export model to MLflow Model Registry.

    This node:
    - Exports model to multiple formats (joblib, sklearn)
    - Registers model in MLflow Model Registry
    - Creates model info card with metadata
    - Prepares deployment artifacts
    - Returns registry and export information

    Args:
        state: AutoForgeState containing model, metrics, and task info.

    Returns:
        Updated state with registry info, export paths, and model card.

    Raises:
        KeyError: If required state keys missing.
        ValueError: If data validation fails.
    """
    best_pipeline = state.get("best_pipeline")
    model_name = state.get("model_name", "autoforge-model")
    export_dir = state.get("export_dir", "exports")
    task_type = state.get("task_type")
    best_params = state.get("best_params", {})
    test_metrics = state.get("test_metrics", {})
    train_metrics = state.get("train_metrics", {})
    feature_columns = state.get("feature_columns", [])
    mlflow_run_id = state.get("mlflow_run_id")

    if best_pipeline is None:
        raise KeyError("best_pipeline required in state")
    if task_type is None:
        raise ValueError("task_type required in state")

    # Export model to multiple formats
    exports = export_model_formats(best_pipeline, export_dir, model_name)

    # Create model info card
    combined_metrics = {**train_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}
    model_card = create_model_info_card(
        model_name,
        task_type,
        best_params,
        combined_metrics,
        feature_names=feature_columns,
    )

    # Save model card
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    model_card_path = export_path / "model_card.json"
    with open(model_card_path, "w") as f:
        json.dump(model_card, f, indent=2)

    # Register in MLflow if run_id available
    registry_info = {}
    if mlflow_run_id:
        model_uri = f"runs:/{mlflow_run_id}/model"
        registry_info = register_model_in_registry(
            model_name,
            model_uri,
            description=f"AutoML {task_type} model",
            tags={
                "task_type": task_type,
                "framework": "scikit-learn",
                "automl": "true",
            },
        )

    return {
        **state,
        "export_formats": exports,
        "model_card": model_card,
        "model_card_path": str(model_card_path),
        "registry_info": registry_info,
        "model_name": model_name,
        "export_dir": export_dir,
        "deployment_ready": True,
    }
