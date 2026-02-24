"""Tests for Preprocess + Optuna node."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from autoforge.nodes.preprocess_optuna import (
    build_preprocessor,
    create_objective,
    make_estimator,
    preprocess_optuna_node,
)


@pytest.fixture
def classification_data(tmp_path: Path) -> dict:
    """Create sample classification dataset."""
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "num2": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "cat": ["a", "b", "a", "b", "a", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    x = df.drop(columns=["target"])
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": "classification",
    }


@pytest.fixture
def regression_data(tmp_path: Path) -> dict:
    """Create sample regression dataset."""
    # Create larger dataset to avoid NaN/inf issues
    np.random.seed(42)
    n_samples = 50
    num1 = np.random.uniform(1, 10, n_samples)
    num2 = np.random.uniform(10, 100, n_samples)
    cat = np.random.choice(["x", "y"], n_samples)
    target = 2 * num1 + 0.5 * num2 + np.random.normal(0, 5, n_samples)

    df = pd.DataFrame(
        {
            "num1": num1,
            "num2": num2,
            "cat": cat,
            "target": target,
        }
    )

    x = df.drop(columns=["target"])
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "task_type": "regression",
    }


def test_build_preprocessor_numeric_only() -> None:
    """Test preprocessor with only numeric features."""
    x = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    preprocessor = build_preprocessor(x)

    assert preprocessor is not None
    # Fit and transform
    x_transformed = preprocessor.fit_transform(x)
    assert x_transformed.shape[0] == 3


def test_build_preprocessor_mixed() -> None:
    """Test preprocessor with numeric and categorical features."""
    x = pd.DataFrame(
        {
            "num": [1, 2, 3],
            "cat": ["a", "b", "a"],
        }
    )
    preprocessor = build_preprocessor(x)
    x_transformed = preprocessor.fit_transform(x)
    assert x_transformed.shape[0] == 3


def test_make_estimator_classification() -> None:
    """Test estimator creation for classification."""
    params_rf = {"model": "rf", "n_estimators": 10, "max_depth": 5}
    est = make_estimator("classification", params_rf)
    assert est is not None
    assert hasattr(est, "predict")

    params_logreg = {"model": "logreg", "C": 1.0}
    est2 = make_estimator("classification", params_logreg)
    assert est2 is not None


def test_make_estimator_regression() -> None:
    """Test estimator creation for regression."""
    params_rf = {"model": "rf", "n_estimators": 10, "max_depth": 5}
    est = make_estimator("regression", params_rf)
    assert est is not None
    assert hasattr(est, "predict")

    params_gbr = {"model": "gbr", "n_estimators": 10, "learning_rate": 0.1}
    est2 = make_estimator("regression", params_gbr)
    assert est2 is not None


def test_create_objective(classification_data: dict) -> None:
    """Test objective function creation."""
    preprocessor = build_preprocessor(classification_data["x_train"])
    objective = create_objective(
        "classification",
        classification_data["x_train"],
        classification_data["y_train"],
        classification_data["x_test"],
        classification_data["y_test"],
        preprocessor,
    )

    assert callable(objective)


def test_preprocess_optuna_node_classification(classification_data: dict) -> None:
    """Test preprocess_optuna_node for classification."""
    state = {
        **classification_data,
        "n_trials": 3,
    }

    result = preprocess_optuna_node(state)

    assert "best_params" in result
    assert "best_score" in result
    assert "best_pipeline" in result
    assert "test_score" in result
    assert result["test_metric"] == "accuracy"
    assert result["n_trials_completed"] == 3
    assert result["best_params"]["model"] in ["rf", "logreg"]


def test_preprocess_optuna_node_regression(regression_data: dict) -> None:
    """Test preprocess_optuna_node for regression."""
    state = {
        **regression_data,
        "n_trials": 3,
    }

    result = preprocess_optuna_node(state)

    assert "best_params" in result
    assert "best_score" in result
    assert "best_pipeline" in result
    assert "test_score" in result
    assert result["test_metric"] == "mean_absolute_error"
    assert result["n_trials_completed"] > 0
    assert result["best_params"]["model"] in ["rf", "gbr"]


def test_preprocess_optuna_node_missing_data() -> None:
    """Test error handling when required data missing."""
    state = {
        "task_type": "classification",
        "n_trials": 3,
        # Missing x_train, y_train
    }

    with pytest.raises(KeyError):
        preprocess_optuna_node(state)


def test_preprocess_optuna_node_missing_task_type(classification_data: dict) -> None:
    """Test error handling when task_type missing."""
    state = {
        "x_train": classification_data["x_train"],
        "y_train": classification_data["y_train"],
        "x_test": classification_data["x_test"],
        "y_test": classification_data["y_test"],
        # Missing task_type
        "n_trials": 3,
    }

    with pytest.raises(ValueError):
        preprocess_optuna_node(state)
