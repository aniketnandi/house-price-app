from __future__ import annotations
import argparse
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from .data import load_data
from .model import build_pipeline, evaluate


def train_and_save(output_path: str) -> dict[str, float]:
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = build_pipeline(list(X.columns))
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    metrics = evaluate(y_val, preds)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/model_v1.joblib")
    args = parser.parse_args()
    metrics = train_and_save(args.model_path)
    print("Saved:", args.model_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
