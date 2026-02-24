"""Tests for DataLoader node."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from autoforge.nodes.data_loader import data_loader_node, load_csv, validate_data


def test_load_csv_valid(tmp_path: Path) -> None:
    """Test loading a valid CSV file."""
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df.to_csv(csv_path, index=False)

    loaded = load_csv(str(csv_path))
    assert loaded.shape == (3, 2)
    assert list(loaded.columns) == ["x", "y"]


def test_load_csv_file_not_found() -> None:
    """Test loading non-existent CSV file raises error."""
    with pytest.raises(FileNotFoundError):
        load_csv("/nonexistent/path.csv")


def test_validate_data_valid(tmp_path: Path) -> None:
    """Test data validation with valid DataFrame."""
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [5, 6, 7, 8],
            "target": [0, 1, 0, 1],
        }
    )
    stats = validate_data(df, "target")

    assert stats["n_rows"] == 4
    assert stats["n_features"] == 2
    assert set(stats["feature_columns"]) == {"feature1", "feature2"}
    assert stats["target_column"] == "target"
    assert stats["all_nan_rows"] == 0


def test_validate_data_missing_target() -> None:
    """Test validation fails when target column missing."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column"):
        validate_data(df, "missing_target")


def test_validate_data_empty() -> None:
    """Test validation fails on empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Target column"):
        validate_data(df, "target")


def test_data_loader_node(tmp_path: Path) -> None:
    """Test data_loader_node end-to-end."""
    csv_path = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4],
            "num2": [10, 11, 12, 13],
            "target": [0, 1, 0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    state = {
        "csv_path": str(csv_path),
        "target_column": "target",
    }
    result = data_loader_node(state)

    assert "dataframe" in result
    assert "data_stats" in result
    assert result["dataframe"].shape == (4, 3)
    assert result["data_stats"]["n_features"] == 2
