"""Integration test: needs the pipeline to have run (CI runs it with --sample first)."""
import pytest

from src.config import PRICE_MODEL_PATH

pytestmark = pytest.mark.skipif(not PRICE_MODEL_PATH.exists(), reason="run `python -m src.pipeline --sample` first")


def test_prediction_is_sane():
    from src.models.predict import make_row, predict
    dub = predict(make_row("Dublin", False, False, dublin_area="D4"))["predicted_price_eur"]
    lei = predict(make_row("Leitrim", False, False))["predicted_price_eur"]
    assert 50_000 < lei < dub < 3_000_000


def test_api():
    from fastapi.testclient import TestClient

    from api.main import app
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/predict", json={"county": "Cork", "is_new": True})
    assert r.status_code == 200 and r.json()["predicted_price_eur"] > 0
    assert client.post("/predict", json={"county": "Atlantis"}).status_code == 422
