"""Data access for the LucidFlow dashboard.

Phase 5, Task 2's functions (pipeline summary, model results, drift status,
entity resolution) are read-only: they read outputs that already exist --
Postgres tables populated by run_pipeline.py, MLflow runs logged by each
model's train.py (Phase 4, Task 1), and the drift monitor's saved reference
profile / last per-batch report (Phase 4, Task 2). Nothing there recomputes
a metric or retrains anything.

"Most recent pipeline run" is derived without any schema change: each of
write_clean_records/write_quarantine_records runs inside one Postgres
transaction (see src/lucidflow/loading/*.py), and Postgres's now() is
stable within a transaction -- so every row written by one call shares one
exact loaded_at/quarantined_at timestamp. Grouping by MAX(loaded_at) (or
MAX(quarantined_at)) therefore isolates exactly the rows from the latest
write of each kind, with no run_id column needed.

Phase 5, Task 3's functions (bottom of the file) are the one place this
dashboard writes: submitting a human review decision to
quarantine.quarantine_reviews, and triggering
models/quarantine_classifier/retrain_with_reviews.py once enough decisions
exist.
"""

import json
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from lucidflow.loading.db import get_engine
from lucidflow.mlflow_config import configure_mlflow

DRIFT_DIR = Path(__file__).resolve().parents[1] / "src" / "lucidflow" / "drift"
LAST_CHECK_RESULTS_PATH = DRIFT_DIR / "last_check_results.json"
ENTITY_RESOLUTION_DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "entity_resolution_investigation.md"

COLUMN_TYPE_EXPERIMENT = "lucidflow-column-type-classifier"
IMPUTATION_EXPERIMENT = "lucidflow-imputation-selector"
QUARANTINE_EXPERIMENT = "lucidflow-quarantine-classifier"
QUARANTINE_REGISTERED_MODEL_NAME = "lucidflow-quarantine-classifier"


# ---------------------------------------------------------------------------
# Pipeline run summary (Postgres)
# ---------------------------------------------------------------------------


def get_latest_run_summary() -> dict | None:
    """Returns counts for the most recent clean write and the most recent quarantine
    write. These two writes happen a few seconds apart within one pipeline run (see
    module docstring), so they're reported as two timestamps, not force-merged into one.
    Returns None if no pipeline run has ever written to either table.
    """
    engine = get_engine()
    with engine.connect() as conn:
        clean_row = conn.execute(
            text(
                """
                SELECT loaded_at, count(*) AS n
                FROM clean.analytics_data
                WHERE loaded_at = (SELECT max(loaded_at) FROM clean.analytics_data)
                GROUP BY loaded_at
                """
            )
        ).fetchone()
        quarantine_row = conn.execute(
            text(
                """
                SELECT quarantined_at, count(*) AS n
                FROM quarantine.records
                WHERE quarantined_at = (SELECT max(quarantined_at) FROM quarantine.records)
                GROUP BY quarantined_at
                """
            )
        ).fetchone()

    if clean_row is None and quarantine_row is None:
        return None

    return {
        "clean_written_at": clean_row.loaded_at if clean_row else None,
        "clean_count": clean_row.n if clean_row else 0,
        "quarantine_written_at": quarantine_row.quarantined_at if quarantine_row else None,
        "quarantine_count": quarantine_row.n if quarantine_row else 0,
    }


def get_latest_quarantine_reasons(limit: int = 500) -> pd.DataFrame:
    """Reason-rule breakdown for the most recent quarantine write -- e.g. how many rows
    failed on 'url', how many were flagged by 'quarantine_classifier', etc."""
    engine = get_engine()
    query = text(
        """
        SELECT reasons
        FROM quarantine.records
        WHERE quarantined_at = (SELECT max(quarantined_at) FROM quarantine.records)
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()

    rule_counts = count_reasons_by_rule([row.reasons for row in rows])
    return pd.DataFrame(
        sorted(rule_counts.items(), key=lambda kv: -kv[1]), columns=["rule", "count"]
    )


def count_reasons_by_rule(reasons_per_row: list[list[dict]]) -> dict[str, int]:
    """Tallies how many rows carry each `rule` value across a list of per-row reasons
    lists (one row can carry multiple reasons, e.g. multiple contract violations).
    Pure function, split out from get_latest_quarantine_reasons for unit testing.
    """
    rule_counts: dict[str, int] = {}
    for row_reasons in reasons_per_row:
        for reason in row_reasons:
            rule_counts[reason["rule"]] = rule_counts.get(reason["rule"], 0) + 1
    return rule_counts


# ---------------------------------------------------------------------------
# Model results (MLflow)
# ---------------------------------------------------------------------------


def _latest_run(experiment_name: str) -> pd.Series | None:
    configure_mlflow(experiment_name)
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    if runs.empty:
        return None
    return runs.iloc[0]


def get_column_type_classifier_results() -> dict | None:
    run = _latest_run(COLUMN_TYPE_EXPERIMENT)
    if run is None:
        return None

    configure_mlflow(COLUMN_TYPE_EXPERIMENT)
    client = MlflowClient()
    confusion_matrix_text = None
    try:
        local_path = client.download_artifacts(run["run_id"], "confusion_matrix.txt")
        confusion_matrix_text = Path(local_path).read_text()
    except OSError:
        pass

    return {
        "run_id": run["run_id"],
        "start_time": run["start_time"],
        "macro_f1": run.get("metrics.macro_f1"),
        "n_total_columns": run.get("params.n_total_columns"),
        "n_test": run.get("params.n_test"),
        "n_misclassified": run.get("params.n_misclassified"),
        "confusion_matrix_text": confusion_matrix_text,
    }


def format_coverage_caveat(singleton_classes: float | None, evaluated_classes: float | None) -> str | None:
    """Renders the same class-coverage caveat run_missingness_engine already prints
    (see selector.py's `known_limitation`), from the two metrics logged to MLflow.
    Pure function, split out from get_imputation_benchmark_results for unit testing.
    """
    if not singleton_classes:
        return None
    singleton = int(singleton_classes)
    total = singleton + int(evaluated_classes)
    return (
        f"{singleton}/{total} classes had exactly 1 known example and went to "
        "training only -- structurally impossible to hold out for testing, so the "
        "F1 above reflects recovery only for the classes with enough examples to evaluate."
    )


def get_imputation_benchmark_results() -> dict | None:
    run = _latest_run(IMPUTATION_EXPERIMENT)
    if run is None:
        return None

    benchmarked = {}
    for column in ("company_size", "country"):
        scores = {
            method: run.get(f"metrics.{column}_{method}_f1")
            for method in ("median", "mode", "knn", "mice", "lightgbm")
            if run.get(f"metrics.{column}_{method}_f1") is not None
        }
        benchmarked[column] = {
            "scores": scores,
            "winner": run.get(f"params.{column}_winner"),
            "evaluated_classes": run.get(f"metrics.{column}_evaluated_classes"),
            "singleton_classes": run.get(f"metrics.{column}_singleton_classes"),
            "coverage_ratio": run.get(f"metrics.{column}_coverage_ratio"),
            "caveat": format_coverage_caveat(
                run.get(f"metrics.{column}_singleton_classes"),
                run.get(f"metrics.{column}_evaluated_classes"),
            ),
        }

    group_mode = {}
    for column in ("state", "city", "zip_code"):
        hit_rate = run.get(f"metrics.{column}_group_key_hit_rate")
        if hit_rate is not None:
            group_mode[column] = {
                "group_cols": run.get(f"params.{column}_group_cols"),
                "hit_rate": hit_rate,
            }

    skipped = [
        col for col in ("address", "description")
        if run.get(f"params.{col}_decision") == "skipped"
    ]

    return {
        "run_id": run["run_id"],
        "start_time": run["start_time"],
        "benchmarked": benchmarked,
        "group_mode": group_mode,
        "skipped": skipped,
    }


def get_quarantine_classifier_results() -> dict | None:
    run = _latest_run(QUARANTINE_EXPERIMENT)
    if run is None:
        return None

    corruption_types = (run.get("params.corruption_types") or "").split(",")
    corruption_types = [c for c in corruption_types if c]

    per_type = {}
    for corruption_type in corruption_types:
        per_type[corruption_type] = {
            "precision": run.get(f"metrics.{corruption_type}_precision"),
            "recall": run.get(f"metrics.{corruption_type}_recall"),
            "f1": run.get(f"metrics.{corruption_type}_f1"),
        }

    return {
        "run_id": run["run_id"],
        "start_time": run["start_time"],
        "pr_auc": run.get("metrics.pr_auc"),
        "precision": run.get("metrics.precision"),
        "recall": run.get("metrics.recall"),
        "f1": run.get("metrics.f1"),
        "clean_false_positive_rate": run.get("metrics.clean_false_positive_rate"),
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# Drift status (Phase 4, Task 2's saved results)
# ---------------------------------------------------------------------------


def get_drift_status() -> dict | None:
    if not LAST_CHECK_RESULTS_PATH.exists():
        return None
    return json.loads(LAST_CHECK_RESULTS_PATH.read_text())


# ---------------------------------------------------------------------------
# Entity resolution investigation (documented negative result)
# ---------------------------------------------------------------------------


def get_entity_resolution_doc() -> str | None:
    if not ENTITY_RESOLUTION_DOC_PATH.exists():
        return None
    return ENTITY_RESOLUTION_DOC_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Human-in-the-loop quarantine review (Phase 5, Task 3)
# ---------------------------------------------------------------------------

MIN_REVIEWS_FOR_RETRAIN = 20


def find_ml_flag_reason(reasons: list[dict]) -> dict | None:
    """Picks out the quarantine_classifier reason entry from a row's `reasons` list, if
    any -- ignores deterministic contract-validation failures (rule != quarantine_classifier)
    and older enriched-schema-less ML flags (no 'features' key, written before Phase 5, Task
    3 started persisting score/features at classification time). Pure function, split out
    from get_review_queue for unit testing.
    """
    return next((r for r in reasons if r.get("rule") == "quarantine_classifier" and "features" in r), None)


def row_to_queue_entry(record_id: int, raw_data: dict, reasons: list[dict], quarantined_at) -> dict | None:
    """Builds one review-queue entry from a quarantine.records row, or None if this row
    isn't a reviewable ML flag (see find_ml_flag_reason). Pure function, split out from
    get_review_queue for unit testing.
    """
    ml_reason = find_ml_flag_reason(reasons)
    if ml_reason is None:
        return None
    return {
        "record_id": record_id,
        "raw_data": raw_data,
        "score": ml_reason["score"],
        "message": ml_reason["message"],
        "model_version": ml_reason.get("model_version"),
        "features": ml_reason["features"],
        "quarantined_at": quarantined_at,
    }


def get_review_queue(limit: int = 200) -> list[dict]:
    """Real, unreviewed quarantine.records rows flagged by the ML classifier (not
    contract-validation failures -- those are deterministic, not model predictions, and
    Phase 3's own reasoning is that training on them would teach nothing new). Only rows
    with the enriched score/features fields are returned (see pipeline_flow.py's
    quarantine_classify_task); older rows written before that field existed are skipped.
    """
    engine = get_engine()
    query = text(
        """
        SELECT q.id, q.raw_data, q.reasons, q.quarantined_at
        FROM quarantine.records q
        LEFT JOIN quarantine.quarantine_reviews r ON r.record_id = q.id
        WHERE r.id IS NULL
        ORDER BY q.quarantined_at DESC, q.id
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).fetchall()

    queue = [row_to_queue_entry(row.id, row.raw_data, row.reasons, row.quarantined_at) for row in rows]
    return [entry for entry in queue if entry is not None]


def submit_review(record_id: int, decision: str) -> bool:
    """Records a reviewer's decision. Returns False (no-op) if the record was already
    reviewed -- ON CONFLICT DO NOTHING enforces one decision per record at the database
    level, not just in the dashboard's own query logic.
    """
    engine = get_engine()
    query = text(
        """
        INSERT INTO quarantine.quarantine_reviews (record_id, decision)
        VALUES (:record_id, :decision)
        ON CONFLICT (record_id) DO NOTHING
        """
    )
    with engine.begin() as conn:
        result = conn.execute(query, {"record_id": record_id, "decision": decision})
        return result.rowcount > 0


def get_review_progress() -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT decision, count(*) AS n FROM quarantine.quarantine_reviews GROUP BY decision")
        ).fetchall()
    counts = {row.decision: row.n for row in rows}
    return {
        "confirmed_bad": counts.get("confirmed_bad", 0),
        "false_positive": counts.get("false_positive", 0),
        "total": sum(counts.values()),
    }


def get_registered_quarantine_model_metrics() -> dict | None:
    """The currently-registered quarantine classifier's synthetic-test-set metrics, for
    display alongside a retrain candidate's -- not recomputed, read from the MLflow run
    that produced the currently-registered version.
    """
    configure_mlflow(QUARANTINE_EXPERIMENT)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{QUARANTINE_REGISTERED_MODEL_NAME}'")
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(latest.run_id)
    return {
        "version": latest.version,
        "run_id": latest.run_id,
        "pr_auc": run.data.metrics.get("pr_auc") or run.data.metrics.get("synthetic_pr_auc"),
    }


def trigger_retrain() -> dict:
    """Runs models/quarantine_classifier/retrain_with_reviews.py synchronously. The caller
    (the dashboard page) is responsible for gating this behind MIN_REVIEWS_FOR_RETRAIN --
    this function doesn't re-check the threshold itself, since retrain_with_reviews.main()
    already errors clearly if there are zero reviewed rows.
    """
    from lucidflow.models.quarantine_classifier.retrain_with_reviews import main as retrain_main

    return retrain_main()
