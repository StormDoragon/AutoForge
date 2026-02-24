from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


def create_app(model_path: str = "artifacts/model.joblib") -> FastAPI:
    app = FastAPI(title="AutoForge Inference API", version="0.1.0")
    resolved_model_path = Path(model_path)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: PredictRequest) -> Dict[str, Any]:
        if not resolved_model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found at {resolved_model_path}")
        model = joblib.load(resolved_model_path)
        predictions = model.predict(payload.records)
        return {"predictions": predictions.tolist()}

    return app


app = create_app()
