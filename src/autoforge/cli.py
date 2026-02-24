from __future__ import annotations

import argparse
import json

from autoforge.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoForge end-to-end pipeline")
    parser.add_argument("csv_path")
    parser.add_argument("target_column")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()

    result = run_pipeline(
        {
            "csv_path": args.csv_path,
            "target_column": args.target_column,
            "n_trials": args.n_trials,
            "artifact_dir": args.artifact_dir,
        }
    )
    print(json.dumps({"metrics": result["metrics"], "export_path": result["export_path"]}, indent=2))


if __name__ == "__main__":
    main()
