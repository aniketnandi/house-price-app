from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from .features import AddDerivedFeatures


def select_features(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    derived = [c for c in ["RoomsPerPerson", "Inc_x_Age"] if c in df.columns]
    return df[numeric_features + derived]


def build_pipeline(feature_names: list[str]) -> Pipeline:
    # California data is numeric-only
    numeric_features = feature_names

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("derive", AddDerivedFeatures()),
            (
                "select",
                FunctionTransformer(
                    select_features,
                    validate=False,
                    kw_args={"numeric_features": numeric_features},
                ),
            ),
            ("pre", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )
    return pipeline


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}
