"""Trains the quarantine classifier on the synthetic-corruption dataset.

    python -m lucidflow.models.quarantine_classifier.train

Splits stratified by corruption_type (clean + the 4 injected types), fits
Isolation Forest on the train split only (its score becomes a classifier
feature, not a separate detector), trains a LightGBM binary classifier, and
reports precision/recall/F1 per corruption type plus aggregate PR-AUC --
per-type breakdown is the point: it shows whether the model actually learned
all four failure modes or is coasting on the easiest one.

Every run is logged to MLflow and the LightGBM classifier is registered as a
new model-registry version (see mlflow_config.py). The registered model is
just the classifier -- the real deployable unit also needs the Isolation
Forest, feature ordering, and threshold, so the full joblib bundle
(MODEL_PATH) is attached to the same run as an artifact too.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from lucidflow.mlflow_config import configure_mlflow
from lucidflow.models.column_type_classifier.split import stratified_min1_split
from lucidflow.models.quarantine_classifier.corruption import CORRUPTION_TYPES
from lucidflow.models.quarantine_classifier.features import FEATURE_NAMES, ISO_FOREST_INPUT_FEATURES

PACKAGE_DIR = Path(__file__).parent
DATASET_PATH = PACKAGE_DIR / "labeled_dataset.json"
MODEL_PATH = PACKAGE_DIR / "quarantine_classifier.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD = 0.5

EXPERIMENT_NAME = "lucidflow-quarantine-classifier"
REGISTERED_MODEL_NAME = "lucidflow-quarantine-classifier"


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

    n_estimators = 300
    clf = lgb.LGBMClassifier(
        random_state=RANDOM_STATE, verbosity=-1, n_estimators=n_estimators, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)
    agg_precision = precision_score(y_test, y_pred, zero_division=0)
    agg_recall = recall_score(y_test, y_pred, zero_division=0)
    agg_f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\nAggregate PR-AUC: {pr_auc:.4f}")
    print(f"Aggregate precision/recall/F1 @ threshold={THRESHOLD}: "
          f"{agg_precision:.4f} / {agg_recall:.4f} / {agg_f1:.4f}")

    clean_mask = np.array([t == "clean" for t in [r["corruption_type"] for r in test_rows]])
    n_clean = int(clean_mask.sum())
    n_clean_false_positives = int(y_pred[clean_mask].sum())
    clean_fp_rate = n_clean_false_positives / n_clean
    print(
        f"\nClean test rows: {n_clean}  |  false-positive rate among them: "
        f"{n_clean_false_positives}/{n_clean} ({100 * clean_fp_rate:.2f}%) "
        "-- this same pool of false positives is shared across every per-type subset below, "
        "so cross-type precision differences mostly reflect each type's own recall (denominator), "
        "not a fundamentally different false-positive rate. Recall is the cleaner per-type signal."
    )

    print("\nPer-corruption-type precision/recall/F1 (each type vs. all clean test rows,")
    print("isolated from the other types so cross-type confusion can't hide in the number):")
    test_types = [r["corruption_type"] for r in test_rows]
    per_type_metrics: dict[str, dict[str, float]] = {}
    for corruption_type in CORRUPTION_TYPES:
        subset_mask = np.array([t in (corruption_type, "clean") for t in test_types])
        y_true_subset = y_test[subset_mask]
        y_pred_subset = y_pred[subset_mask]
        n_positive = int(y_true_subset.sum())
        precision = precision_score(y_true_subset, y_pred_subset, zero_division=0)
        recall = recall_score(y_true_subset, y_pred_subset, zero_division=0)
        f1 = f1_score(y_true_subset, y_pred_subset, zero_division=0)
        per_type_metrics[corruption_type] = {"precision": precision, "recall": recall, "f1": f1, "n": n_positive}
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

    configure_mlflow(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "class_weight": "balanced",
                "threshold": THRESHOLD,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "corruption_types": ",".join(CORRUPTION_TYPES),
                "n_total": len(dataset),
                "n_train": len(train_rows),
                "n_test": len(test_rows),
            }
        )
        mlflow.log_metrics(
            {
                "pr_auc": pr_auc,
                "precision": agg_precision,
                "recall": agg_recall,
                "f1": agg_f1,
                "clean_false_positive_rate": clean_fp_rate,
            }
        )
        for corruption_type, metrics in per_type_metrics.items():
            mlflow.log_metrics(
                {
                    f"{corruption_type}_precision": metrics["precision"],
                    f"{corruption_type}_recall": metrics["recall"],
                    f"{corruption_type}_f1": metrics["f1"],
                }
            )
        # The registered model is the LightGBM classifier alone -- the deployable unit also
        # needs the Isolation Forest, feature ordering, and threshold, so the full bundle is
        # attached as a plain artifact on the same run rather than registered separately.
        mlflow.lightgbm.log_model(clf, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="full_bundle")
        print(f"\nLogged to MLflow run {run.info.run_id}, registered as '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    main()
