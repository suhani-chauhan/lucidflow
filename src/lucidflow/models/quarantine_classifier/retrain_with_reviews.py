"""Retrains the quarantine classifier by combining the existing synthetic-corruption
dataset with human-reviewed real quarantine.records decisions (Phase 5, Task 3).

Triggered from the dashboard's Quarantine Review page once at least
MIN_REVIEWS_FOR_RETRAIN human decisions have been recorded -- not run
automatically or on a schedule.

Honesty note (same discipline as every synthetic/thin-data case earlier in
this project): Task 0's grounding found real companies.csv has 0 rows
failing the Pydantic contract, and the model's real-world flags are
overwhelmingly expected to be legitimate false positives rather than
organic corruption -- so the human-labeled 'confirmed_bad' count here may
be zero or very small. That is itself real information (what the model's
false-positive patterns on real data look like), not a failure of the
review process, and it's reported as such rather than papered over.

The registration decision is anchored ONLY on the fixed synthetic held-out
test split -- the same split, same random_state, that train.py itself
uses -- precisely because that split is comparable across runs regardless
of how many human-reviewed rows happen to exist. The human-reviewed
evaluation (when there are enough reviewed rows to hold any out) is
reported as a separate, explicitly small, supplementary signal and is
never used as the registration gate.

    python -m lucidflow.models.quarantine_classifier.retrain_with_reviews
"""

import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sqlalchemy import text

from lucidflow.loading.db import get_engine
from lucidflow.mlflow_config import configure_mlflow
from lucidflow.models.column_type_classifier.split import stratified_min1_split
from lucidflow.models.quarantine_classifier.build_dataset import build_dataset
from lucidflow.models.quarantine_classifier.features import FEATURE_NAMES, ISO_FOREST_INPUT_FEATURES
from lucidflow.models.quarantine_classifier.train import (
    EXPERIMENT_NAME,
    MODEL_PATH,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    TEST_SIZE,
    THRESHOLD,
    build_matrix,
    fit_isolation_forest,
    row_label,
)

PR_AUC_REGRESSION_TOLERANCE = 0.01
HUMAN_TEST_SIZE = 0.2


def load_human_reviewed_rows() -> list[dict]:
    """Every reviewed quarantine.records row, in the same {company_id, is_synthetic,
    label, corruption_type, features} shape build_dataset.py produces -- features are
    read back exactly as persisted at classification time (see pipeline_flow.py's
    quarantine_classify_task), never recomputed, so training sees precisely what the
    model actually scored.
    """
    engine = get_engine()
    query = text(
        """
        SELECT q.id, q.raw_data, q.reasons, r.decision
        FROM quarantine.records q
        JOIN quarantine.quarantine_reviews r ON r.record_id = q.id
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()

    labeled = [reviewed_row_to_labeled_row(row.raw_data, row.reasons, row.decision) for row in rows]
    return [row for row in labeled if row is not None]


def reviewed_row_to_labeled_row(raw_data: dict, reasons: list[dict], decision: str) -> dict | None:
    """Builds one labeled training row from a reviewed quarantine.records row, or None if
    it isn't a reviewable ML flag (defensive -- the review queue only ever offers enriched
    ML flags, so this shouldn't happen in practice). Pure function, split out from
    load_human_reviewed_rows for unit testing.
    """
    ml_reason = next(
        (rr for rr in reasons if rr.get("rule") == "quarantine_classifier" and "features" in rr), None
    )
    if ml_reason is None:
        return None
    is_confirmed_bad = decision == "confirmed_bad"
    return {
        "company_id": raw_data.get("company_id"),
        "is_synthetic": False,
        "label": 1 if is_confirmed_bad else 0,
        "corruption_type": "human_confirmed_bad" if is_confirmed_bad else "human_false_positive",
        "features": ml_reason["features"],
    }


def _registered_model_pr_auc() -> float | None:
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    except MlflowException:
        return None
    latest = max(versions, key=lambda v: int(v.version), default=None)
    if latest is None:
        return None
    run = client.get_run(latest.run_id)
    return run.data.metrics.get("pr_auc")


def main() -> dict:
    synthetic_dataset = build_dataset()
    human_rows = load_human_reviewed_rows()
    if not human_rows:
        raise ValueError("No human-reviewed rows found in quarantine.quarantine_reviews -- nothing to retrain on.")

    # Synthetic split: identical params to train.py's own split, so the resulting synthetic
    # test set -- and its PR-AUC -- is directly comparable to the currently registered
    # model's synthetic PR-AUC.
    synthetic_labels = [row["corruption_type"] for row in synthetic_dataset]
    syn_train_idx, syn_test_idx, _, _ = stratified_min1_split(
        list(range(len(synthetic_dataset))), synthetic_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    synthetic_train = [synthetic_dataset[i] for i in syn_train_idx]
    synthetic_test = [synthetic_dataset[i] for i in syn_test_idx]

    # Human split: a small, separately-reported eval -- see module docstring for why it's
    # never used for the registration decision. A class with <2 members (e.g. 0 or 1
    # confirmed_bad rows) goes entirely to train -- stratified_min1_split handles that.
    human_labels = [row["corruption_type"] for row in human_rows]
    hum_train_idx, hum_test_idx, _, _ = stratified_min1_split(
        list(range(len(human_rows))), human_labels, test_size=HUMAN_TEST_SIZE, random_state=RANDOM_STATE
    )
    human_train = [human_rows[i] for i in hum_train_idx]
    human_test = [human_rows[i] for i in hum_test_idx]

    train_rows = synthetic_train + human_train
    test_rows = synthetic_test  # kept pure-synthetic for the primary, comparable metric

    print(
        f"Synthetic: {len(synthetic_train)} train / {len(synthetic_test)} test  |  "
        f"Human-reviewed: {len(human_train)} train / {len(human_test)} test "
        f"({sum(1 for r in human_rows if r['label'] == 1)} confirmed_bad, "
        f"{sum(1 for r in human_rows if r['label'] == 0)} false_positive)"
    )

    iso_forest = fit_isolation_forest(train_rows)
    X_train = build_matrix(train_rows, iso_forest)
    X_test = build_matrix(test_rows, iso_forest)
    y_train = np.array([row_label(r) for r in train_rows])
    y_test = np.array([row_label(r) for r in test_rows])

    n_estimators = 300
    clf = lgb.LGBMClassifier(
        random_state=RANDOM_STATE, verbosity=-1, n_estimators=n_estimators, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    synthetic_pr_auc = average_precision_score(y_test, y_proba)
    synthetic_precision = precision_score(y_test, y_pred, zero_division=0)
    synthetic_recall = recall_score(y_test, y_pred, zero_division=0)
    synthetic_f1 = f1_score(y_test, y_pred, zero_division=0)
    print(
        f"\nSynthetic held-out PR-AUC: {synthetic_pr_auc:.4f}  |  "
        f"precision/recall/F1: {synthetic_precision:.4f} / {synthetic_recall:.4f} / {synthetic_f1:.4f}"
    )

    human_eval = None
    if human_test:
        X_human_test = build_matrix(human_test, iso_forest)
        y_human_test = np.array([row_label(r) for r in human_test])
        y_human_pred = (clf.predict_proba(X_human_test)[:, 1] >= THRESHOLD).astype(int)
        human_eval = {
            "n": len(human_test),
            "precision": precision_score(y_human_test, y_human_pred, zero_division=0),
            "recall": recall_score(y_human_test, y_human_pred, zero_division=0),
        }
        print(
            f"Human-reviewed held-out eval (n={human_eval['n']}, small-sample, NOT used for "
            f"registration): precision={human_eval['precision']:.4f} recall={human_eval['recall']:.4f}"
        )
    else:
        print("Human-reviewed held-out eval: skipped (too few reviewed rows to hold any out for testing).")

    registered_pr_auc = _registered_model_pr_auc()
    should_register = (
        registered_pr_auc is None or synthetic_pr_auc >= registered_pr_auc - PR_AUC_REGRESSION_TOLERANCE
    )
    print(
        f"\nCurrently registered model's synthetic PR-AUC: "
        f"{registered_pr_auc:.4f}" if registered_pr_auc is not None else "n/a (nothing registered yet)"
    )
    print(f"Register this retrained candidate? {should_register}")

    configure_mlflow(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "class_weight": "balanced",
                "threshold": THRESHOLD,
                "retrain_trigger": "human_review",
                "pr_auc_regression_tolerance": PR_AUC_REGRESSION_TOLERANCE,
                "n_synthetic_total": len(synthetic_dataset),
                "n_human_reviewed": len(human_rows),
                "n_human_confirmed_bad": sum(1 for r in human_rows if r["label"] == 1),
                "n_human_false_positive": sum(1 for r in human_rows if r["label"] == 0),
                "n_train": len(train_rows),
                "n_test": len(test_rows),
            }
        )
        mlflow.log_metrics(
            {
                "synthetic_pr_auc": synthetic_pr_auc,
                "synthetic_precision": synthetic_precision,
                "synthetic_recall": synthetic_recall,
                "synthetic_f1": synthetic_f1,
            }
        )
        if registered_pr_auc is not None:
            mlflow.log_metric("registered_model_pr_auc", registered_pr_auc)
        if human_eval is not None:
            mlflow.log_metrics(
                {
                    "human_eval_n": human_eval["n"],
                    "human_eval_precision": human_eval["precision"],
                    "human_eval_recall": human_eval["recall"],
                }
            )

        registered_new_version = None
        if should_register:
            mlflow.lightgbm.log_model(clf, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME)
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
            registered_new_version = max(versions, key=lambda v: int(v.version)).version
        else:
            mlflow.lightgbm.log_model(clf, artifact_path="model")

        bundle = {
            "model": clf,
            "iso_forest": iso_forest,
            "feature_names": [*FEATURE_NAMES, "iso_forest_score"],
            "iso_forest_input_features": ISO_FOREST_INPUT_FEATURES,
            "threshold": THRESHOLD,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = Path(tmp_dir) / "quarantine_classifier.joblib"
            joblib.dump(bundle, bundle_path)
            mlflow.log_artifact(str(bundle_path), artifact_path="full_bundle")
            if should_register:
                # Only overwrite the committed local artifact if this candidate is actually
                # being shipped -- a rejected candidate's bundle still lives in the MLflow
                # run artifact above, but shouldn't silently replace what's deployed.
                joblib.dump(bundle, MODEL_PATH)

        print(f"\nLogged to MLflow run {run.info.run_id}" + (
            f", registered as '{REGISTERED_MODEL_NAME}' v{registered_new_version}"
            if should_register else " (not registered -- see tolerance check above)"
        ))

        return {
            "run_id": run.info.run_id,
            "synthetic_pr_auc": synthetic_pr_auc,
            "registered_pr_auc": registered_pr_auc,
            "should_register": should_register,
            "registered_new_version": registered_new_version,
            "human_eval": human_eval,
            "n_human_reviewed": len(human_rows),
            "n_human_confirmed_bad": sum(1 for r in human_rows if r["label"] == 1),
            "n_human_false_positive": sum(1 for r in human_rows if r["label"] == 0),
        }


if __name__ == "__main__":
    main()
