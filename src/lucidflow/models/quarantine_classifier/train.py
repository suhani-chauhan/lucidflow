"""Trains the quarantine classifier on the synthetic-corruption dataset.

    python -m lucidflow.models.quarantine_classifier.train

Splits stratified by corruption_type (clean + the 4 injected types), fits
Isolation Forest on the train split only (its score becomes a classifier
feature, not a separate detector), trains a LightGBM binary classifier, and
reports precision/recall/F1 per corruption type plus aggregate PR-AUC --
per-type breakdown is the point: it shows whether the model actually learned
all four failure modes or is coasting on the easiest one.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from lucidflow.models.column_type_classifier.split import stratified_min1_split
from lucidflow.models.quarantine_classifier.corruption import CORRUPTION_TYPES
from lucidflow.models.quarantine_classifier.features import FEATURE_NAMES, ISO_FOREST_INPUT_FEATURES

PACKAGE_DIR = Path(__file__).parent
DATASET_PATH = PACKAGE_DIR / "labeled_dataset.json"
MODEL_PATH = PACKAGE_DIR / "quarantine_classifier.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text())


def main() -> None:
    dataset = load_dataset()
    labels_for_split = [row["corruption_type"] for row in dataset]
    indices = list(range(len(dataset)))

    train_idx, test_idx, _, _ = stratified_min1_split(
        indices, labels_for_split, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_rows = [dataset[i] for i in train_idx]
    test_rows = [dataset[i] for i in test_idx]

    print(f"Total rows: {len(dataset)}  |  train: {len(train_rows)}  |  test: {len(test_rows)}")

    iso_train_X = np.array([[r["features"][f] for f in ISO_FOREST_INPUT_FEATURES] for r in train_rows])
    iso_forest = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_STATE)
    iso_forest.fit(iso_train_X)

    def build_matrix(rows: list[dict]) -> np.ndarray:
        base = np.array([[r["features"][f] for f in FEATURE_NAMES] for r in rows])
        iso_X = np.array([[r["features"][f] for f in ISO_FOREST_INPUT_FEATURES] for r in rows])
        iso_scores = iso_forest.decision_function(iso_X).reshape(-1, 1)
        return np.hstack([base, iso_scores])

    X_train = build_matrix(train_rows)
    X_test = build_matrix(test_rows)
    y_train = np.array([1 if r["is_synthetic"] else 0 for r in train_rows])
    y_test = np.array([1 if r["is_synthetic"] else 0 for r in test_rows])

    clf = lgb.LGBMClassifier(
        random_state=RANDOM_STATE, verbosity=-1, n_estimators=300, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nAggregate PR-AUC: {pr_auc:.4f}")
    print(f"Aggregate precision/recall/F1 @ threshold={THRESHOLD}: "
          f"{precision_score(y_test, y_pred, zero_division=0):.4f} / "
          f"{recall_score(y_test, y_pred, zero_division=0):.4f} / "
          f"{f1_score(y_test, y_pred, zero_division=0):.4f}")

    clean_mask = np.array([t == "clean" for t in [r["corruption_type"] for r in test_rows]])
    n_clean = int(clean_mask.sum())
    n_clean_false_positives = int(y_pred[clean_mask].sum())
    print(
        f"\nClean test rows: {n_clean}  |  false-positive rate among them: "
        f"{n_clean_false_positives}/{n_clean} ({100 * n_clean_false_positives / n_clean:.2f}%) "
        "-- this same pool of false positives is shared across every per-type subset below, "
        "so cross-type precision differences mostly reflect each type's own recall (denominator), "
        "not a fundamentally different false-positive rate. Recall is the cleaner per-type signal."
    )

    print("\nPer-corruption-type precision/recall/F1 (each type vs. all clean test rows,")
    print("isolated from the other types so cross-type confusion can't hide in the number):")
    test_types = [r["corruption_type"] for r in test_rows]
    for corruption_type in CORRUPTION_TYPES:
        subset_mask = np.array([t in (corruption_type, "clean") for t in test_types])
        y_true_subset = y_test[subset_mask]
        y_pred_subset = y_pred[subset_mask]
        n_positive = int(y_true_subset.sum())
        precision = precision_score(y_true_subset, y_pred_subset, zero_division=0)
        recall = recall_score(y_true_subset, y_pred_subset, zero_division=0)
        f1 = f1_score(y_true_subset, y_pred_subset, zero_division=0)
        print(f"  {corruption_type:20s} n={n_positive:4d}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")

    joblib.dump(
        {
            "model": clf,
            "iso_forest": iso_forest,
            "feature_names": [*FEATURE_NAMES, "iso_forest_score"],
            "iso_forest_input_features": ISO_FOREST_INPUT_FEATURES,
            "threshold": THRESHOLD,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
