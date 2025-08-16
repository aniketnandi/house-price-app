from __future__ import annotations
import argparse
import joblib
import pandas as pd
from pathlib import Path


def batch_predict(model_path: str, input_csv: str, output_csv: str) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(input_csv)
    preds = model.predict(df)
    out = df.copy()
    out["prediction"] = preds
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/model_v1.joblib")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="data/predictions.csv")
    args = parser.parse_args()
    batch_predict(args.model_path, args.input_csv, args.output_csv)
    print("Wrote:", args.output_csv)


if __name__ == "__main__":
    main()
