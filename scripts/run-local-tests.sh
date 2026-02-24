#!/usr/bin/env bash
set -euo pipefail
# Run tests without installing the package (useful in offline/proxy environments)
export PYTHONPATH="${PYTHONPATH:-}""$(cd "$(dirname "$0")/.." && pwd)/src"
echo "Running pytest with PYTHONPATH=$PYTHONPATH"
python -m pytest -q "$@"
