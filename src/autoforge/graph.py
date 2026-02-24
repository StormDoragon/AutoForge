"""LangGraph orchestration for AutoML pipeline.

Defines the complete AutoML workflow combining all nodes:
- DataLoader: Load and validate input data
- PreprocessOptuna: Feature preprocessing and hyperparameter tuning
- TrainMLflow: Model training with MLflow logging
- ModelRegistry: Model versioning and export
- FastAPIDeploy: REST API deployment
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from langgraph.graph import StateGraph

from autoforge.nodes.data_loader import data_loader_node
from autoforge.nodes.fastapi_deploy import fastapi_deploy_node
from autoforge.nodes.model_registry import model_registry_node
from autoforge.nodes.preprocess_optuna import preprocess_optuna_node
from autoforge.nodes.train_mlflow import train_mlflow_node
from autoforge.types import AutoForgeState


def data_split_node(state: AutoForgeState) -> AutoForgeState:
    """Split loaded data into train/test sets.

    This intermediate node:
    - Extracts target column from dataframe
    - Splits data into train/test
    - Returns x_train, x_test, y_train, y_test

    Args:
        state: AutoForgeState with dataframe and target_column

    Returns:
        Updated state with train/test splits
    """
    dataframe = state.get("dataframe")
    target_column = state.get("target_column")
    test_size = state.get("test_size", 0.2)
    random_state = state.get("random_state", 42)

    if dataframe is None:
        raise ValueError("dataframe not found in state")
    if not target_column:
        raise ValueError("target_column not found in state")

    # Extract features and target
    y = dataframe[target_column]
    X = dataframe.drop(columns=[target_column])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return {
        **state,
        "x_train": X_train,
        "x_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,  # Also store capitalized for convention
        "X_test": X_test,
    }


def prepare_deployment_node(state: AutoForgeState) -> AutoForgeState:
    """Prepare model path and features for deployment.

    This intermediate node:
    - Extracts model path from exports
    - Prepares feature column names
    - Prepares for FastAPI deployment

    Args:
        state: AutoForgeState with export_formats and feature info

    Returns:
        Updated state with model_path for deployment
    """
    export_formats = state.get("export_formats", {})
    
    # Get model path from joblib export
    model_path = export_formats.get("joblib")
    if not model_path:
        # Fallback to sklearn if joblib not available
        model_path = export_formats.get("sklearn")
    
    if not model_path:
        raise ValueError("No model export format available")
    
    # Extract feature column names from X_train if available
    feature_columns = state.get("feature_columns", [])
    X_train = state.get("x_train")
    if X_train is None:
        X_train = state.get("X_train")
    
    if X_train is not None and not feature_columns:
        feature_columns = X_train.columns.tolist()
    
    return {
        **state,
        "model_path": model_path,
        "feature_columns": feature_columns,
    }



def create_automl_graph() -> StateGraph:
    """Create LangGraph StateGraph for complete AutoML pipeline.

    Workflow stages:
    1. data_loader_node: Load CSV, validate, extract statistics
    2. data_split_node: Split data into train/test sets
    3. preprocess_optuna_node: Feature preprocessing and Optuna tuning
    4. train_mlflow_node: Model training with MLflow experiment tracking
    5. model_registry_node: Export formats and MLflow Model Registry
    6. prepare_deployment_node: Prepare model path for deployment
    7. fastapi_deploy_node: REST API deployment setup

    Returns:
        StateGraph: Compiled workflow graph ready for invocation

    Example:
        >>> graph = create_automl_graph()
        >>> state = {
        ...     "csv_path": "data.csv",
        ...     "target_column": "target",
        ...     "task_type": "classification"
        ... }
        >>> result = graph.invoke(state)
    """
    # Create state graph
    graph = StateGraph(AutoForgeState)

    # Add all nodes in pipeline sequence
    graph.add_node("data_loader", data_loader_node)
    graph.add_node("data_split", data_split_node)
    graph.add_node("preprocess_optuna", preprocess_optuna_node)
    graph.add_node("train_mlflow", train_mlflow_node)
    graph.add_node("model_registry", model_registry_node)
    graph.add_node("prepare_deployment", prepare_deployment_node)
    graph.add_node("fastapi_deploy", fastapi_deploy_node)

    # Define edges (sequential workflow)
    graph.add_edge("data_loader", "data_split")
    graph.add_edge("data_split", "preprocess_optuna")
    graph.add_edge("preprocess_optuna", "train_mlflow")
    graph.add_edge("train_mlflow", "model_registry")
    graph.add_edge("model_registry", "prepare_deployment")
    graph.add_edge("prepare_deployment", "fastapi_deploy")

    # Set entry and exit points
    graph.set_entry_point("data_loader")
    graph.set_finish_point("fastapi_deploy")

    # Compile the graph
    return graph.compile()




def run_automl_pipeline(
    data_path: str,
    target_column: str,
    task_type: str = "classification",
    feature_columns: list[str] | None = None,
    test_size: float = 0.2,
    optuna_trials: int = 10,
    random_state: int = 42,
    mlflow_experiment_name: str | None = None,
    model_name: str = "automl_model",
) -> dict:
    """Execute complete AutoML pipeline end-to-end.

    Orchestrates data loading, preprocessing, model training, registry, and deployment.

    Args:
        data_path: Path to CSV file
        target_column: Name of target column
        task_type: "classification" or "regression"
        feature_columns: Optional list of feature column names
        test_size: Fraction of data for testing (0.0-1.0)
        optuna_trials: Number of Optuna optimization trials
        random_state: Random seed for reproducibility
        mlflow_experiment_name: Custom MLflow experiment name
        model_name: Name for model registry and exports

    Returns:
        dict: Complete pipeline state with:
        - dataframe: Input data
        - data_stats: Statistics (rows, features, nulls)
        - X_train/X_test: Training/testing features
        - y_train/y_test: Training/testing targets
        - pipeline: Final trained model pipeline
        - train_metrics: Training performance metrics
        - test_metrics: Testing performance metrics
        - export_formats: Saved model paths
        - model_card: Metadata card
        - registry_info: MLflow registry information
        - fastapi_app: Deployment FastAPI application
        - deployment_status: Ready/failed

    Raises:
        FileNotFoundError: If data file not found
        ValueError: If target_column not in data or invalid task_type
        KeyError: If required state keys missing during execution

    Example:
        >>> result = run_automl_pipeline(
        ...     data_path="data.csv",
        ...     target_column="target",
        ...     task_type="classification",
        ...     optuna_trials=20
        ... )
        >>> print(f"Model saved to: {result['model_path']}")
        >>> print(f"API deployed: {result['deployment_status']}")
    """
    # Build initial state
    initial_state: AutoForgeState = {
        "csv_path": data_path,  # Map data_path to csv_path for data_loader_node
        "target_column": target_column,
        "task_type": task_type,
        "feature_columns": feature_columns or [],
        "test_size": test_size,
        "optuna_trials": optuna_trials,
        "random_state": random_state,
        "model_name": model_name,
    }

    # Add optional MLflow experiment name if provided
    if mlflow_experiment_name:
        initial_state["mlflow_experiment_name"] = mlflow_experiment_name

    # Create and invoke graph
    graph = create_automl_graph()
    final_state = graph.invoke(initial_state)

    return final_state


def get_pipeline_graph() -> StateGraph:
    """Get compiled AutoML pipeline graph for inspection/visualization.

    Returns:
        StateGraph: Compiled workflow graph

    Example:
        >>> graph = get_pipeline_graph()
        >>> print(graph.get_graph().draw_ascii())
    """
    return create_automl_graph()
