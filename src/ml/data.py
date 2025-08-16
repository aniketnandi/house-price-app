from __future__ import annotations
from sklearn.datasets import fetch_california_housing
import pandas as pd


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return features X and target y as pandas objects."""
    ds = fetch_california_housing(as_frame=True)
    X: pd.DataFrame = ds.data
    y: pd.Series = ds.target  # median house value in $100,000s
    return X, y
