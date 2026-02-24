from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict


TaskType = Literal["classification", "regression"]


class AutoForgeState(TypedDict, total=False):
    csv_path: str
    target_column: str
    task_type: TaskType
    test_size: float
    random_state: int
    n_trials: int
    artifact_dir: str
    tracking_uri: Optional[str]

    dataframe_shape: tuple[int, int]
    feature_columns: List[str]

    x_train: Any
    x_test: Any
    y_train: Any
    y_test: Any
    best_params: Dict[str, Any]
    best_score: float

    model: Any
    metrics: Dict[str, float]
    run_id: str
    model_uri: str
    export_path: str
