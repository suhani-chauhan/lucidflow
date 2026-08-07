from pathlib import Path

import joblib
import pytest

from lucidflow.models.column_type_classifier.features import extract_column_features

MODEL_PATH = Path("src/lucidflow/models/column_type_classifier/column_type_classifier.joblib")

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model artifact not present")


def _predict(values: list[str]) -> str:
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    feature_names = bundle["feature_names"]
    features = extract_column_features(values)
    return clf.predict([[features[name] for name in feature_names]])[0]


def test_0_1_encoded_boolean_column_predicts_boolean():
    # This matches the only encoding the training data actually contains
    # (remote_allowed, sponsored, inferred are all "0"/"1").
    values = ["0", "1", "0", "0", "1", "0", "1", "0", "0", "1"] * 20

    assert _predict(values) == "boolean"


def test_true_false_text_boolean_column_is_misclassified_as_categorical():
    """Known, documented limitation — not a silent surprise.

    The training set's only boolean examples (remote_allowed, sponsored,
    inferred) are all numeric-coded ("0"/"1"), so the model never learned
    that text "true"/"false" values are also boolean. On text values it
    falls back to the closest thing it *has* seen: low-cardinality text,
    i.e. categorical. If a "true"/"false"-style boolean column shows up in
    a future dataset, expect it to be mislabeled until more diverse
    boolean training examples are added.
    """
    values = ["true", "false", "true", "true", "false", "false", "true"] * 10

    assert _predict(values) == "categorical"
