from __future__ import annotations

from pathlib import Path

import pandas as pd

from autoforge.pipeline import run_pipeline


def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5, 6, 7, 8],
            "num2": [10, 11, 12, 13, 14, 15, 16, 17],
            "cat": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "target": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    result = run_pipeline(
        {
            "csv_path": str(csv_path),
            "target_column": "target",
            "n_trials": 2,
            "artifact_dir": str(tmp_path / "artifacts"),
            "tracking_uri": f"file://{(tmp_path / 'mlruns').resolve()}",
        }
    )

    assert "metrics" in result
    assert Path(result["export_path"]).exists()
