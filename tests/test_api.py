from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from autoforge.api.app import create_app


def test_predict_endpoint(tmp_path: Path) -> None:
    model = DummyClassifier(strategy="most_frequent")
    x = pd.DataFrame({"feature": [1, 2]})
    y = [0, 0]
    model.fit(x, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    app = create_app(str(model_path))
    client = TestClient(app)

    response = client.post("/predict", json={"records": [{"feature": 9}, {"feature": 10}]})
    assert response.status_code == 200
    assert "predictions" in response.json()
