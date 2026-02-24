# AutoForge

Single-repo LangGraph AutoML system: **raw CSV → Optuna tuning → MLflow tracking → export → FastAPI deploy**.

## Project structure

- `src/autoforge/agents/nodes.py` — agent nodes for each pipeline phase.
- `src/autoforge/pipeline.py` — LangGraph orchestration.
- `src/autoforge/api/app.py` — inference API (`/health`, `/predict`).
- `tests/` — end-to-end pipeline and API tests.
- `.github/workflows/ci.yml` — lint + test in CI.
- `Dockerfile` — containerized inference service.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run the full pipeline:

```bash
python -m autoforge.cli data/train.csv target --n-trials 25 --artifact-dir artifacts
```

This produces:
- `artifacts/model.joblib`
- `artifacts/metadata.json`
- local MLflow run data in `mlruns/` (or custom `tracking_uri`)

## FastAPI inference

```bash
uvicorn autoforge.api.app:app --reload
```

Prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"feature1": 1.2, "feature2": "A"}]}'
```

## Docker

```bash
docker build -t autoforge .
docker run -p 8000:8000 autoforge
```

## Tests

```bash
# If your environment cannot access PyPI (proxy/firewall), you can run tests without installing the package:
# Option 1: run via Makefile (sets PYTHONPATH to `src`)
make test-local

# Option 2: run the helper script
scripts/run-local-tests.sh

# In CI or on a machine with network access, run the full test matrix with installs:
pytest -q
```
If you need to pre-download wheels on a networked machine and transfer them to an offline environment, use `dev-requirements.txt`:

```bash
pip download -r dev-requirements.txt -d ./wheels
# then transfer ./wheels to the offline machine and install with:
# pip install --no-index --find-links ./wheels "autoforge[dev]"
```
