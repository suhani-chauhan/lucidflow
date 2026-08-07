"""Candidate imputation strategies for a classification-style target column
(company_size, state), sharing a uniform fit/predict interface so the same
objects can be used both for benchmarking (fit on a train fold, score on a
held-out fold) and for the final persisted imputer (fit on all known rows,
predict the real missing rows).

Predictors are frequency-encoded for KNN/MICE (which require pure numeric
input, fit on training rows only to avoid leakage) and passed as native
pandas categoricals to LightGBM, which handles them directly.
"""

import numpy as np
import pandas as pd
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401  (registers IterativeImputer)
)
from sklearn.impute import IterativeImputer, KNNImputer

RANDOM_STATE = 42


def _encode_predictors_fit(train_df: pd.DataFrame, predictor_cols: list[str]) -> dict:
    freq_maps = {}
    for col in predictor_cols:
        if not pd.api.types.is_numeric_dtype(train_df[col]):
            freq_maps[col] = train_df[col].value_counts(normalize=True)
    return freq_maps


def _encode_predictors_transform(df: pd.DataFrame, predictor_cols: list[str], freq_maps: dict) -> np.ndarray:
    encoded = pd.DataFrame(index=df.index)
    for col in predictor_cols:
        if col in freq_maps:
            encoded[col] = df[col].map(freq_maps[col]).fillna(0.0)
        else:
            encoded[col] = df[col].astype(float)
    return encoded.to_numpy(dtype=float)


class MedianStrategy:
    """Naive baseline for ordinal columns: median of known codes, rounded to the nearest valid code."""

    name = "median"

    def fit(self, known_df: pd.DataFrame, target_col: str, predictor_cols: list[str]) -> None:
        self.value = round(known_df[target_col].median())

    def predict(self, df: pd.DataFrame, predictor_cols: list[str]) -> pd.Series:
        return pd.Series(self.value, index=df.index)


class ModeStrategy:
    """Naive baseline for nominal columns: most frequent known category."""

    name = "mode"

    def fit(self, known_df: pd.DataFrame, target_col: str, predictor_cols: list[str]) -> None:
        self.value = known_df[target_col].mode().iloc[0]

    def predict(self, df: pd.DataFrame, predictor_cols: list[str]) -> pd.Series:
        return pd.Series(self.value, index=df.index)


class KNNStrategy:
    """KNNImputer over a numeric-encoded [target_code, *predictors] matrix."""

    name = "knn"

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, known_df: pd.DataFrame, target_col: str, predictor_cols: list[str]) -> None:
        self.categories = sorted(known_df[target_col].unique())
        self.cat_to_code = {c: i for i, c in enumerate(self.categories)}
        self.code_to_cat = dict(enumerate(self.categories))
        self.freq_maps = _encode_predictors_fit(known_df, predictor_cols)

        X = _encode_predictors_transform(known_df, predictor_cols, self.freq_maps)
        y_code = known_df[target_col].map(self.cat_to_code).to_numpy(dtype=float).reshape(-1, 1)
        train_matrix = np.hstack([y_code, X])

        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer.fit(train_matrix)

    def predict(self, df: pd.DataFrame, predictor_cols: list[str]) -> pd.Series:
        X = _encode_predictors_transform(df, predictor_cols, self.freq_maps)
        missing_target = np.full((len(df), 1), np.nan)
        matrix = np.hstack([missing_target, X])
        filled = self.imputer.transform(matrix)
        codes = np.clip(np.rint(filled[:, 0]).astype(int), 0, len(self.categories) - 1)
        return pd.Series([self.code_to_cat[c] for c in codes], index=df.index)


class MICEStrategy:
    """IterativeImputer (MICE) over the same numeric-encoded matrix as KNNStrategy."""

    name = "mice"

    def fit(self, known_df: pd.DataFrame, target_col: str, predictor_cols: list[str]) -> None:
        self.categories = sorted(known_df[target_col].unique())
        self.cat_to_code = {c: i for i, c in enumerate(self.categories)}
        self.code_to_cat = dict(enumerate(self.categories))
        self.freq_maps = _encode_predictors_fit(known_df, predictor_cols)

        X = _encode_predictors_transform(known_df, predictor_cols, self.freq_maps)
        y_code = known_df[target_col].map(self.cat_to_code).to_numpy(dtype=float).reshape(-1, 1)
        train_matrix = np.hstack([y_code, X])

        self.imputer = IterativeImputer(random_state=RANDOM_STATE, max_iter=10)
        self.imputer.fit(train_matrix)

    def predict(self, df: pd.DataFrame, predictor_cols: list[str]) -> pd.Series:
        X = _encode_predictors_transform(df, predictor_cols, self.freq_maps)
        missing_target = np.full((len(df), 1), np.nan)
        matrix = np.hstack([missing_target, X])
        filled = self.imputer.transform(matrix)
        codes = np.clip(np.rint(filled[:, 0]).astype(int), 0, len(self.categories) - 1)
        return pd.Series([self.code_to_cat[c] for c in codes], index=df.index)


class LightGBMStrategy:
    """LGBMClassifier fit directly on predictors -> target, using native categorical support."""

    name = "lightgbm"

    def fit(self, known_df: pd.DataFrame, target_col: str, predictor_cols: list[str]) -> None:
        import lightgbm as lgb

        X = known_df[predictor_cols].copy()
        self.categorical_cols = [c for c in predictor_cols if not pd.api.types.is_numeric_dtype(X[c])]
        self.category_levels = {}
        for col in self.categorical_cols:
            X[col] = X[col].astype("category")
            self.category_levels[col] = X[col].cat.categories

        self.model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1, n_estimators=200)
        self.model.fit(X, known_df[target_col])

    def predict(self, df: pd.DataFrame, predictor_cols: list[str]) -> pd.Series:
        X = df[predictor_cols].copy()
        for col in self.categorical_cols:
            X[col] = X[col].astype("category").cat.set_categories(self.category_levels[col])
        return pd.Series(self.model.predict(X), index=df.index)


def build_strategies(column_kind: str) -> list:
    baseline = MedianStrategy() if column_kind == "ordinal" else ModeStrategy()
    return [baseline, KNNStrategy(), MICEStrategy(), LightGBMStrategy()]
