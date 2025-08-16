from src.ml.train import train_and_save
from pathlib import Path


def test_training(tmp_path: Path):
    out = tmp_path / "model.joblib"
    metrics = train_and_save(str(out))
    assert out.exists()
    # Basic sanity checks
    assert 0.0 <= metrics["r2"] <= 1.0
    assert metrics["rmse"] > 0.0
