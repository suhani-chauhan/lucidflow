"""Runs the trained column-type classifier (Model 1) against columns from any CSV.

Model 1 is used exactly as trained -- no retraining, no fine-tuning. This module
only adds the inference path that was previously missing: `train.py` fits and
evaluates the classifier, but nothing before this consumed its predictions.
"""

from dataclasses import dataclass

import joblib
import polars as pl

from lucidflow.models.column_type_classifier.features import extract_column_features
from lucidflow.models.column_type_classifier.train import MODEL_PATH


@dataclass
class TypePrediction:
    column: str
    predicted_type: str
    confidence: float
    # Full probability distribution over all 8 trained labels, not just the winner --
    # lets callers see e.g. "categorical 0.55 / boolean 0.40" instead of just "categorical".
    proba: dict[str, float]


def _load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained column-type classifier not found at {MODEL_PATH} -- "
            "run `python -m lucidflow.models.column_type_classifier.train` first."
        )
    return joblib.load(MODEL_PATH)


def classify_columns(columns: dict[str, list[str | None]]) -> dict[str, TypePrediction]:
    """Predicts each column's semantic type from its statistical fingerprint.

    `columns` maps column name -> raw string values (None for nulls), same shape
    `extract_column_features` expects -- i.e. read the source CSV with every
    column as a string, same as `ingestion.loader.load_file` already does.
    """
    bundle = _load_bundle()
    clf = bundle["model"]
    feature_names = bundle["feature_names"]
    labels = bundle["labels"]

    predictions: dict[str, TypePrediction] = {}
    for column, raw_values in columns.items():
        features = extract_column_features(raw_values)
        x = [[features[name] for name in feature_names]]
        proba = clf.predict_proba(x)[0]
        proba_by_label = dict(zip(labels, (float(p) for p in proba)))
        predicted_type = max(proba_by_label, key=proba_by_label.get)
        predictions[column] = TypePrediction(
            column=column,
            predicted_type=predicted_type,
            confidence=proba_by_label[predicted_type],
            proba=proba_by_label,
        )
    return predictions


def classify_dataframe(df: pl.DataFrame) -> dict[str, TypePrediction]:
    """Convenience wrapper: classify every column of an all-string Polars DataFrame."""
    return classify_columns({column: df[column].to_list() for column in df.columns})
