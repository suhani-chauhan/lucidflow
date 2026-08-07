"""Trains the semantic column-type classifier on the fingerprint dataset.

    python -m lucidflow.models.column_type_classifier.train

Loads column_fingerprints.json (features) + confirmed_labels.csv (ground
truth), splits with a guaranteed minimum of one held-out example per class,
trains a RandomForestClassifier, reports macro-F1 and the confusion matrix,
and persists the fitted model + label list under this package.

Every run is logged to MLflow and registered as a new model-registry
version (see mlflow_config.py).
"""

import csv
import json
import tempfile
from pathlib import Path

import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

from lucidflow.mlflow_config import configure_mlflow
from lucidflow.models.column_type_classifier.features import FEATURE_NAMES
from lucidflow.models.column_type_classifier.split import stratified_min1_split

PACKAGE_DIR = Path(__file__).parent
FINGERPRINTS_PATH = PACKAGE_DIR / "column_fingerprints.json"
LABELS_PATH = PACKAGE_DIR / "confirmed_labels.csv"
MODEL_PATH = PACKAGE_DIR / "column_type_classifier.joblib"

EXPERIMENT_NAME = "lucidflow-column-type-classifier"
REGISTERED_MODEL_NAME = "lucidflow-column-type-classifier"


def load_training_data() -> tuple[list[list[float]], list[str], list[str]]:
    fingerprints = json.loads(FINGERPRINTS_PATH.read_text())
    by_key = {(row["source_file"], row["column"]): row["features"] for row in fingerprints}

    with LABELS_PATH.open(newline="") as f:
        label_rows = list(csv.DictReader(f))

    X: list[list[float]] = []
    y: list[str] = []
    keys: list[str] = []
    for row in label_rows:
        key = (row["source_file"], row["column"])
        features = by_key[key]
        X.append([features[name] for name in FEATURE_NAMES])
        y.append(row["label"])
        keys.append(f"{key[0]}/{key[1]}")

    return X, y, keys


def main() -> None:
    X, y, keys = load_training_data()
    indexed = list(range(len(X)))
    train_i, test_i, y_train, y_test = stratified_min1_split(indexed, y, test_size=0.25, random_state=42)
    X_train = [X[i] for i in train_i]
    X_test = [X[i] for i in test_i]
    test_keys = [keys[i] for i in test_i]

    print(f"Total columns: {len(X)}  |  train: {len(X_train)}  |  test: {len(X_test)}")

    n_estimators = 300
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print(f"\nMacro-F1 (held-out): {macro_f1:.4f}\n")
    print("Confusion matrix (rows=true, cols=predicted):")
    header = "".join(f"{lbl[:12]:>14}" for lbl in labels_sorted)
    print(f"{'':>18}{header}")
    cm_lines = [f"{'':>18}{header}"]
    for label, row in zip(labels_sorted, cm):
        row_str = "".join(f"{v:>14}" for v in row)
        line = f"{label[:18]:>18}{row_str}"
        print(line)
        cm_lines.append(line)

    misclassified = [
        (key, true, pred) for key, true, pred in zip(test_keys, y_test, y_pred) if true != pred
    ]
    if misclassified:
        print("\nMisclassified held-out columns:")
        for key, true, pred in misclassified:
            print(f"  {key}: true={true}  predicted={pred}")

    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "labels": labels_sorted}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    configure_mlflow(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "class_weight": "balanced",
                "random_state": 42,
                "test_size": 0.25,
                "n_total_columns": len(X),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_misclassified": len(misclassified),
            }
        )
        mlflow.log_metric("macro_f1", macro_f1)
        for label, row in zip(labels_sorted, cm):
            for pred_label, count in zip(labels_sorted, row):
                if count:
                    mlflow.log_metric(f"confusion_{label}_as_{pred_label}", int(count))

        with tempfile.TemporaryDirectory() as tmp_dir:
            confusion_matrix_path = Path(tmp_dir) / "confusion_matrix.txt"
            confusion_matrix_path.write_text("\n".join(cm_lines))
            mlflow.log_artifact(str(confusion_matrix_path))

        mlflow.sklearn.log_model(clf, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME)
        print(f"\nLogged to MLflow run {run.info.run_id}, registered as '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    main()
