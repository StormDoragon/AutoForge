from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


TaskType = Literal["classification", "regression"]


class AutoForgeState(TypedDict, total=False):
    # Inputs
    csv_path: str
    data_path: str
    target_column: str
    task_type: TaskType
    test_size: float
    random_state: int
    n_trials: int
    optuna_trials: int
    artifact_dir: str
    tracking_uri: Optional[str]
    mlflow_experiment_name: str
    model_name: str
    export_dir: str

    # Data loading & preprocessing
    dataframe: Any  # DataFrame
    data_stats: Dict[str, Any]
    dataframe_shape: tuple[int, int]
    feature_columns: List[str]

    # Train/test splits
    x_train: Any  # DataFrame
    x_test: Any  # DataFrame
    X_train: Any  # DataFrame (capitalized variant)
    X_test: Any  # DataFrame (capitalized variant)
    y_train: Any  # Series
    y_test: Any  # Series

    # Optimization & training
    best_params: Dict[str, Any]
    best_score: float
    test_score: float
    test_metric: str
    best_pipeline: Any  # Pipeline
    best_trial_number: int
    n_trials_completed: int

    # Model evaluation
    model: Any  # Trained model
    pipeline: Any  # Fitted pipeline
    metrics: Dict[str, float]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]

    # MLflow logging
    mlflow_run_id: str
    run_id: str
    mlflow_experiment_name: str

    # Model registry & export
    model_uri: str
    export_path: str
    export_formats: Dict[str, str]  # {"joblib": path, "sklearn": path}
    model_card: Dict[str, Any]
    model_card_path: str
    registry_info: Dict[str, Any]
    deployment_ready: bool

    # Deployment
    model_path: str
    fastapi_app: Any
    deployment_app: Any
    model_loaded: bool
    api_endpoints: List[str]
    deployment_status: str
    server_host: str
    server_port: int
