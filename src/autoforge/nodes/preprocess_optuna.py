"""Preprocessing and Optuna hyperparameter optimization node.

Handles feature engineering, preprocessing pipeline construction, and
Optuna-based hyperparameter tuning for AutoML.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autoforge.types import AutoForgeState


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical features.

    Args:
        x: Input feature DataFrame.

    Returns:
        ColumnTransformer with pipelines for numeric and categorical columns.
    """
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers)


def make_estimator(task_type: str, params: dict[str, Any]):
    """Create a scikit-learn estimator based on task type and parameters.

    Args:
        task_type: Either 'classification' or 'regression'.
        params: Dictionary of model parameters including 'model' key.

    Returns:
        Initialized estimator.
    """
    model_name = params.get("model", "rf")

    if task_type == "classification":
        if model_name == "logreg":
            return LogisticRegression(max_iter=1000, C=params.get("C", 1.0), random_state=42)
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            random_state=42,
        )

    if model_name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 100),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=42,
        )

    return RandomForestRegressor(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 10),
        random_state=42,
    )


def create_objective(
    task_type: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    preprocessor: ColumnTransformer,
) -> Callable[[Trial], float]:
    """Create an Optuna objective function for hyperparameter optimization.

    Args:
        task_type: 'classification' or 'regression'.
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features.
        y_val: Validation labels.
        preprocessor: ColumnTransformer for preprocessing.

    Returns:
        Objective function callable by Optuna.
    """

    def objective(trial: Trial) -> float:
        # Suggest hyperparameters
        if task_type == "classification":
            model_choice = trial.suggest_categorical("model", ["rf", "logreg"])
            if model_choice == "logreg":
                params = {"model": "logreg", "C": trial.suggest_float("C", 0.01, 100, log=True)}
            else:
                params = {
                    "model": "rf",
                    "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                }
        else:  # regression
            model_choice = trial.suggest_categorical("model", ["rf", "gbr"])
            if model_choice == "gbr":
                params = {
                    "model": "gbr",
                    "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
                }
            else:
                params = {
                    "model": "rf",
                    "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                }

        # Create and train model
        estimator = make_estimator(task_type, params)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        try:
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_val)

            # Compute metric
            if task_type == "classification":
                score = accuracy_score(y_val, y_pred)
            else:
                # Use mean_absolute_error for regression to avoid NaN with small datasets
                mae = mean_absolute_error(y_val, y_pred)
                score = -mae  # Negate so higher is better
            
            # Check for NaN/inf
            if not np.isfinite(score):
                return -1.0 if task_type == "regression" else 0.0
                
            return score
        except Exception:
            # Return worst possible score on error
            return -1.0 if task_type == "regression" else 0.0

    return objective


def preprocess_optuna_node(state: AutoForgeState) -> AutoForgeState:
    """Preprocess data and optimize hyperparameters with Optuna.

    This node:
    - Builds a preprocessing pipeline
    - Splits training data into train/val for hyperparameter tuning
    - Runs Optuna hyperparameter optimization
    - Trains final model with best parameters
    - Returns best parameters and trained pipeline

    Args:
        state: AutoForgeState containing train/test data and task info.

    Returns:
        Updated state with 'best_params', 'best_score', and 'best_pipeline'.

    Raises:
        KeyError: If required state keys missing.
        ValueError: If data validation fails.
    """
    x_train = state.get("x_train")
    y_train = state.get("y_train")
    x_test = state.get("x_test")
    y_test = state.get("y_test")
    task_type = state.get("task_type")
    n_trials = state.get("n_trials", 10)

    if x_train is None or y_train is None:
        raise KeyError("x_train and y_train required in state")
    if task_type is None:
        raise ValueError("task_type required in state")

    # Build preprocessor
    preprocessor = build_preprocessor(x_train)

    # Split training data for hyperparameter tuning (80/20 split)
    n_train = int(0.8 * len(x_train))
    x_hp_train = x_train.iloc[:n_train]
    y_hp_train = y_train.iloc[:n_train]
    x_hp_val = x_train.iloc[n_train:]
    y_hp_val = y_train.iloc[n_train:]

    # Create and run Optuna study
    objective = create_objective(task_type, x_hp_train, y_hp_train, x_hp_val, y_hp_val, preprocessor)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Handle case where all trials failed
    completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed_trials:
        # Use default params if no successful trials
        if task_type == "classification":
            best_params = {"model": "rf", "n_estimators": 100, "max_depth": 10}
        else:
            best_params = {"model": "rf", "n_estimators": 100, "max_depth": 10}
        best_score = 0.0
        best_trial_number = -1
    else:
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        best_trial_number = best_trial.number

    # Train final model with best parameters on full training set
    estimator = make_estimator(task_type, best_params)
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    best_pipeline.fit(x_train, y_train)

    # Evaluate on test set
    y_pred = best_pipeline.predict(x_test)
    if task_type == "classification":
        test_score = accuracy_score(y_test, y_pred)
        test_metric = "accuracy"
    else:
        test_score = mean_absolute_error(y_test, y_pred)
        test_metric = "mean_absolute_error"

    return {
        **state,
        "best_params": best_params,
        "best_score": best_score,
        "test_score": test_score,
        "test_metric": test_metric,
        "best_pipeline": best_pipeline,
        "best_trial_number": best_trial_number,
        "n_trials_completed": len(completed_trials),
    }
