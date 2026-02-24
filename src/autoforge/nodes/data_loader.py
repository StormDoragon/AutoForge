"""DataLoader node for AutoForge pipeline.

Handles reading CSV files, validating data, and preparing datasets for AutoML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from autoforge.types import AutoForgeState


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV file and return DataFrame.

    Args:
        csv_path: Path to CSV file.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        pd.errors.ParserError: If CSV is malformed.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse CSV {csv_path}: {e}") from e

    return df


def validate_data(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Validate DataFrame and extract basic statistics.

    Args:
        df: Input DataFrame.
        target_column: Name of target column.

    Returns:
        Dictionary with validation results.

    Raises:
        ValueError: If target column missing or no valid features.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df) < 2:
        raise ValueError("DataFrame has fewer than 2 rows")

    # Get feature columns (exclude target)
    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise ValueError("No feature columns found (only target column exists)")

    # Check for rows with all NaN features
    df_features = df[feature_cols]
    all_nan_rows = df_features.isna().all(axis=1).sum()

    return {
        "n_rows": len(df),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "target_column": target_column,
        "all_nan_rows": all_nan_rows,
        "missing_per_column": df.isna().sum().to_dict(),
    }


def data_loader_node(state: AutoForgeState) -> AutoForgeState:
    """Load and validate data from CSV.

    This node:
    - Loads CSV file from the provided path
    - Validates the data contains target column
    - Checks data has sufficient rows/columns
    - Returns validation stats and the loaded DataFrame

    Args:
        state: AutoForgeState containing 'csv_path' and 'target_column'.

    Returns:
        Updated state with 'dataframe' and 'data_stats' keys.

    Raises:
        FileNotFoundError: If CSV path does not exist.
        ValueError: If data validation fails.
    """
    csv_path = state.get("csv_path")
    target_column = state.get("target_column")

    if not csv_path:
        raise ValueError("'csv_path' not found in state")
    if not target_column:
        raise ValueError("'target_column' not found in state")

    # Load CSV
    df = load_csv(csv_path)

    # Validate data
    stats = validate_data(df, target_column)

    return {
        **state,
        "dataframe": df,
        "data_stats": stats,
    }
