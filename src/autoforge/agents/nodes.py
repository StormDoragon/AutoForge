from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from autoforge.types import AutoForgeState


def _infer_task_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    return "regression"


def _build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    return ColumnTransformer(
        transformers=[("numeric", numeric_pipe, numeric_cols), ("categorical", categorical_pipe, categorical_cols)]
    )


def _make_estimator(task_type: str, params: Dict[str, Any]):
    model_name = params["model"]
    if task_type == "classification":
        if model_name == "logreg":
            return LogisticRegression(max_iter=1000, C=params["C"])
        return RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42
        )

    if model_name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], random_state=42
        )
    return RandomForestRegressor(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42
    )


def load_data_node(state: AutoForgeState) -> AutoForgeState:
    df = pd.read_csv(state["csv_path"])
    target_col = state["target_column"]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV")

    y = df[target_col]
    x = df.drop(columns=[target_col])

    task_type = state.get("task_type") or _infer_task_type(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=state.get("test_size", 0.2),
        random_state=state.get("random_state", 42),
        stratify=y if task_type == "classification" else None,
    )

    return {
        **state,
        "task_type": task_type,
        "dataframe_shape": df.shape,
        "feature_columns": x.columns.tolist(),
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def optimize_node(state: AutoForgeState) -> AutoForgeState:
    x_train = state["x_train"]
    y_train = state["y_train"]
    x_test = state["x_test"]
    y_test = state["y_test"]
    task_type = state["task_type"]
    preprocessor = _build_preprocessor(x_train)

    def objective(trial: optuna.Trial) -> float:
        if task_type == "classification":
            model_choice = trial.suggest_categorical("model", ["rf", "logreg"])
            if model_choice == "logreg":
                params = {"model": "logreg", "C": trial.suggest_float("C", 1e-2, 10.0, log=True)}
            else:
                params = {
                    "model": "rf",
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                }
        else:
            model_choice = trial.suggest_categorical("model", ["rf", "gbr"])
            if model_choice == "gbr":
                params = {
                    "model": "gbr",
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                }
            else:
                params = {
                    "model": "rf",
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 2, 20),
                }

        estimator = _make_estimator(task_type, params)
        model = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        if task_type == "classification":
            return f1_score(y_test, preds, average="weighted")
        return r2_score(y_test, preds)

    direction = "maximize"
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=state.get("n_trials", 20))

    return {**state, "best_params": study.best_params, "best_score": float(study.best_value)}


def train_eval_mlflow_node(state: AutoForgeState) -> AutoForgeState:
    x_train = state["x_train"]
    y_train = state["y_train"]
    x_test = state["x_test"]
    y_test = state["y_test"]

    tracking_uri = state.get("tracking_uri") or f"file://{Path('mlruns').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("autoforge")

    preprocessor = _build_preprocessor(x_train)
    estimator = _make_estimator(state["task_type"], state["best_params"])
    model = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    if state["task_type"] == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
        }
        signature = mlflow.models.infer_signature(x_train, model.predict(x_train))
        flavor = mlflow.sklearn
    else:
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
        }
        signature = mlflow.models.infer_signature(x_train, model.predict(x_train))
        flavor = mlflow.sklearn

    with mlflow.start_run() as run:
        mlflow.log_params(state["best_params"])
        mlflow.log_metrics(metrics)
        flavor.log_model(model, "model", signature=signature)

    return {
        **state,
        "model": model,
        "metrics": metrics,
        "run_id": run.info.run_id,
        "model_uri": f"runs:/{run.info.run_id}/model",
    }


def registry_export_node(state: AutoForgeState) -> AutoForgeState:
    artifact_dir = Path(state.get("artifact_dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    export_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"

    joblib.dump(state["model"], export_path)
    metadata = {
        "task_type": state["task_type"],
        "target_column": state["target_column"],
        "feature_columns": state["feature_columns"],
        "metrics": state["metrics"],
        "model_uri": state["model_uri"],
        "mlflow_run_id": state["run_id"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {**state, "export_path": str(export_path)}
