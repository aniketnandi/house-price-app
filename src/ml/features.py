from __future__ import annotations
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AddDerivedFeatures(BaseEstimator, TransformerMixin):
    """Example derived features for tabular numeric data."""

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Avoid divide-by-zero with small epsilon
        eps = 1e-6
        if {"AveRooms", "AveOccup"}.issubset(X.columns):
            X["RoomsPerPerson"] = X["AveRooms"] / (X["AveOccup"] + eps)
        if {"MedInc", "HouseAge"}.issubset(X.columns):
            X["Inc_x_Age"] = X["MedInc"] * X["HouseAge"]
        return X
