"""Integration tests for LangGraph orchestration.

Tests the complete AutoML pipeline end-to-end with all nodes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoforge.graph import create_automl_graph, run_automl_pipeline


@pytest.fixture
def sample_classification_csv(tmp_path: Path) -> str:
    """Create sample classification dataset CSV."""
    data = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [8, 7, 6, 5, 4, 3, 2, 1],
            "feature3": [2, 4, 6, 8, 10, 12, 14, 16],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    csv_path = tmp_path / "classification.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_regression_csv(tmp_path: Path) -> str:
    """Create sample regression dataset CSV."""
    data = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "feature3": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "target": [15.0, 35.0, 55.0, 75.0, 95.0, 115.0, 135.0, 155.0],
        }
    )
    csv_path = tmp_path / "regression.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)


def test_create_automl_graph() -> None:
    """Test creating and compiling the AutoML pipeline graph."""
    graph = create_automl_graph()

    assert graph is not None
    nodes = list(graph.nodes)
    assert "data_loader" in nodes
    assert "data_split" in nodes
    assert "preprocess_optuna" in nodes
    assert "train_mlflow" in nodes
    assert "model_registry" in nodes
    assert "prepare_deployment" in nodes
    assert "fastapi_deploy" in nodes


def test_automl_pipeline_classification(sample_classification_csv: str) -> None:
    """Test complete AutoML pipeline with classification task."""
    result = run_automl_pipeline(
        data_path=sample_classification_csv,
        target_column="target",
        task_type="classification",
        optuna_trials=3,
        random_state=42,
    )

    # Verify deployment stage completed
    assert result["deployment_status"] == "ready"
    assert result["model_loaded"] is True
    assert "fastapi_app" in result


def test_automl_pipeline_regression(sample_regression_csv: str) -> None:
    """Test complete AutoML pipeline with regression task."""
    result = run_automl_pipeline(
        data_path=sample_regression_csv,
        target_column="target",
        task_type="regression",
        optuna_trials=3,
        random_state=42,
    )

    # Verify completion
    assert result["deployment_status"] == "ready"
    assert result["task_type"] == "regression"


def test_automl_pipeline_missing_file() -> None:
    """Test pipeline fails gracefully with missing data file."""
    with pytest.raises(FileNotFoundError):
        run_automl_pipeline(
            data_path="/nonexistent/file.csv",
            target_column="target",
            task_type="classification",
        )


def test_automl_pipeline_invalid_target_column(sample_classification_csv: str) -> None:
    """Test pipeline fails with nonexistent target column."""
    with pytest.raises(ValueError):
        run_automl_pipeline(
            data_path=sample_classification_csv,
            target_column="nonexistent_column",
            task_type="classification",
        )


def test_automl_pipeline_state_flow(sample_classification_csv: str) -> None:
    """Test that state flows properly through all pipeline stages."""
    result = run_automl_pipeline(
        data_path=sample_classification_csv,
        target_column="target",
        task_type="classification",
        optuna_trials=2,
    )

    # Verify key outputs from all stages
    assert "dataframe" in result  # data_loader
    assert "x_train" in result  # data_split
    assert "best_pipeline" in result  # preprocess_optuna
    assert "mlflow_run_id" in result  # train_mlflow
    assert "export_formats" in result  # model_registry
    assert "model_path" in result  # prepare_deployment
    assert "deployment_status" in result  # fastapi_deploy


def test_graph_stream_execution(sample_classification_csv: str) -> None:
    """Test graph streaming execution."""
    graph = create_automl_graph()
    
    state = {
        "csv_path": sample_classification_csv,
        "target_column": "target",
        "task_type": "classification",
        "optuna_trials": 2,
    }
    
    # Test streaming execution
    chunks = []
    for chunk in graph.stream(state):
        chunks.append(chunk)
    
    # Verify we got output from multiple nodes
    assert len(chunks) > 0
