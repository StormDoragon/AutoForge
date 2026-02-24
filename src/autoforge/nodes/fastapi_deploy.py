"""FastAPI deployment and endpoint creation node.

Handles creation of FastAPI application with prediction endpoints,
input validation, and model serving capabilities for AutoML models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autoforge.types import AutoForgeState


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""

    records: list[dict[str, Any]] = Field(..., description="List of feature dictionaries")

    model_config = {"json_schema_extra": {"example": {"records": [{"feature1": 1.0, "feature2": 2.0}]}}}


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    predictions: list[int | float] = Field(..., description="Predicted values")
    probabilities: list[list[float]] | None = Field(None, description="Class probabilities for classification")
    n_samples: int = Field(..., description="Number of samples predicted")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(..., description="Type of model (classification/regression)")


def create_fastapi_app(
    model_path: str,
    task_type: str,
    feature_names: list[str] | None = None,
) -> tuple[FastAPI, Any]:
    """Create FastAPI application with prediction endpoints.

    Args:
        model_path: Path to saved model (joblib format).
        task_type: 'classification' or 'regression'.
        feature_names: List of expected feature names.

    Returns:
        Tuple of (FastAPI app, loaded model).

    Raises:
        FileNotFoundError: If model file not found.
        ValueError: If model cannot be loaded.
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load model from {model_path}: {e}") from e

    app = FastAPI(
        title="AutoForge Model Server",
        description=f"AutoML {task_type} model serving API",
        version="1.0.0",
    )

    # Store model and config in app state
    app.model = model
    app.task_type = task_type
    app.feature_names = feature_names or []

    @app.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=app.task_type,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        """Make predictions on input data."""
        try:
            # Convert records to DataFrame
            df = pd.DataFrame(request.records)

            # Make predictions
            predictions = app.model.predict(df)

            result = PredictionResponse(
                predictions=predictions.tolist(),
                probabilities=None,
                n_samples=len(predictions),
            )

            # For classification, try to get probabilities
            if app.task_type == "classification":
                try:
                    if hasattr(app.model, "predict_proba"):
                        probs = app.model.predict_proba(df)
                        result.probabilities = probs.tolist()
                except Exception:
                    pass

            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}") from e

    @app.get("/info")
    def model_info() -> dict[str, Any]:
        """Get model information."""
        return {
            "model_type": app.task_type,
            "feature_count": len(app.feature_names),
            "feature_names": app.feature_names,
            "framework": "scikit-learn",
        }

    return app, model


def fastapi_deploy_node(state: AutoForgeState) -> AutoForgeState:
    """Create and prepare FastAPI deployment application.

    This node:
    - Loads trained model from path
    - Creates FastAPI application with endpoints
    - Configures prediction, health, and info endpoints
    - Prepares for deployment

    Args:
        state: AutoForgeState containing model path and task info.

    Returns:
        Updated state with FastAPI app and deployment info.

    Raises:
        KeyError: If required state keys missing.
        FileNotFoundError: If model file not found.
    """
    model_path = state.get("model_path")
    task_type = state.get("task_type")
    feature_columns = state.get("feature_columns", [])

    if not model_path:
        raise KeyError("model_path required in state")
    if not task_type:
        raise ValueError("task_type required in state")

    # Create FastAPI app
    app, model = create_fastapi_app(model_path, task_type, feature_columns)

    return {
        **state,
        "fastapi_app": app,
        "deployment_app": app,
        "model_loaded": True,
        "api_endpoints": [
            "/health",
            "/predict",
            "/info",
            "/docs",
            "/openapi.json",
        ],
        "deployment_status": "ready",
        "server_host": "0.0.0.0",
        "server_port": 8000,
    }
