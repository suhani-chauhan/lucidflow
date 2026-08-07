"""Prefect 3 flow wrapping the LucidFlow pipeline: ingestion -> validation ->
cleaning -> imputation -> quarantine classification -> dual-route write, plus
a drift check that can trigger the retrain subflow (retrain_flow.py).

    python -m lucidflow.flows.pipeline_flow [path/to/file.csv]

This is what `run_pipeline.py` now calls -- see that file's docstring. Each
stage below is a Prefect task with retries and structured logging via
`get_run_logger()`, replacing the inline stage-by-stage script that used to
live directly in run_pipeline.py.

Quarantine classification is a real routing stage now, not just an
orchestration wrapper: rows that pass the Phase 1 Pydantic contract are
additionally scored by the trained quarantine classifier
(models/quarantine_classifier), and rows it flags get rerouted to
quarantine.records instead of clean.analytics_data, tagged as an ML flag
(not a contract violation) with the MLflow registry version that produced
the classifier -- so routing stays traceable to a model version, same as
the imputation selector (see mlflow_config.py, Phase 4 Task 1).

The drift check has no real incoming stream to check against --
companies.csv is a static snapshot (see drift/monitor.py) -- so it checks a
synthetic full-magnitude shifted copy of the just-ingested batch (reusing
drift/build_batches.py's shift mechanism) as a stand-in for "the next
incoming batch." This proves the trigger wiring works end-to-end; it does
not mean real-world drift was detected. Because the shift is always the
same full/"significant" magnitude, the check always flags and always
triggers retrain_flow -- so `run_drift_check` defaults to False and is
opt-in (`--check-drift`), rather than making every routine pipeline run
kick off an ~8-10 minute retrain. See retrain_flow.py for what a triggered
retrain actually does.
"""

import argparse
import logging
import random
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task

from lucidflow.drift.build_batches import FULL_SHIFT_FRACTION, SHIFT_SEEDS
from lucidflow.drift.monitor import check_drift
from lucidflow.drift.reference_profile import load_reference_profile
from lucidflow.drift.synthetic_shift import (
    shift_company_size,
    shift_description_length,
    shift_state_null_rate,
)
from lucidflow.flows.retrain_flow import retrain_flow
from lucidflow.ingestion.loader import load_file
from lucidflow.loading.db import get_engine
from lucidflow.loading.postgres_writer import write_clean_records
from lucidflow.loading.quarantine_writer import write_quarantine_records
from lucidflow.models.imputation_selector.selector import print_report, run_missingness_engine
from lucidflow.models.quarantine_classifier.features import (
    FEATURE_NAMES,
    ISO_FOREST_INPUT_FEATURES,
    extract_row_features,
)
from lucidflow.models.quarantine_classifier.zip_state_reference import build_zip3_state_reference
from lucidflow.pipeline_stages import clean_records, validate_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lucidflow.flows.pipeline_flow")

QUARANTINE_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "quarantine_classifier" / "quarantine_classifier.joblib"
)
QUARANTINE_REGISTERED_MODEL_NAME = "lucidflow-quarantine-classifier"


@task(retries=2, retry_delay_seconds=5)
def ingest_task(input_path: str) -> pl.DataFrame:
    run_logger = get_run_logger()
    df = load_file(input_path)
    run_logger.info("ingestion: loaded %d rows from %s", df.height, input_path)
    return df


@task(retries=1)
def validate_task(df: pl.DataFrame) -> tuple[list[dict], list[dict], list[list[dict]]]:
    """Validates each raw row against the Company contract.

    Returns (valid_records, invalid_raw_rows, invalid_reasons) where invalid_reasons[i]
    is the full list of every validation failure on invalid_raw_rows[i], not just the first.
    """
    run_logger = get_run_logger()
    valid_records, invalid_rows, invalid_reasons = validate_rows(df)
    run_logger.info("validation: %d passed, %d failed", len(valid_records), len(invalid_rows))
    return valid_records, invalid_rows, invalid_reasons


@task(retries=1)
def clean_task(valid_records: list[dict]) -> tuple[pl.DataFrame, int]:
    run_logger = get_run_logger()
    valid_df, removed_count = clean_records(valid_records)
    run_logger.info("cleaning: removed %d exact duplicates", removed_count)
    return valid_df, removed_count


@task(retries=1)
def impute_task(valid_df: pl.DataFrame) -> tuple[pl.DataFrame, list[dict]]:
    run_logger = get_run_logger()
    valid_df, missingness_report = run_missingness_engine(valid_df)
    if missingness_report:
        print_report(missingness_report)
    run_logger.info("imputation: processed %d columns with missing values", len(missingness_report))
    return valid_df, missingness_report


def _current_quarantine_model_version() -> str | None:
    try:
        versions = MlflowClient().search_model_versions(f"name='{QUARANTINE_REGISTERED_MODEL_NAME}'")
    except MlflowException:
        return None
    latest = max(versions, key=lambda v: int(v.version), default=None)
    return latest.version if latest else None


@task(retries=1)
def quarantine_classify_task(valid_df: pl.DataFrame) -> tuple[pl.DataFrame, list[dict], list[list[dict]]]:
    """Scores every row that passed the Pydantic gate with the trained quarantine classifier.
    Flagged rows are pulled out of `valid_df` and returned as (kept_df, flagged_raw_rows,
    flagged_reasons) so the caller can merge them into the same quarantine-write path as
    contract failures -- tagged as an ML flag, not a contract violation, with the registry
    version for traceability.
    """
    run_logger = get_run_logger()

    if valid_df.height == 0:
        return valid_df, [], []

    bundle = joblib.load(QUARANTINE_MODEL_PATH)
    model, iso_forest = bundle["model"], bundle["iso_forest"]
    threshold = bundle["threshold"]
    model_version = _current_quarantine_model_version()

    zip3_to_state = build_zip3_state_reference(valid_df)
    rows = valid_df.to_dicts()
    features = [extract_row_features(row, zip3_to_state) for row in rows]

    base_X = np.array([[f[name] for name in FEATURE_NAMES] for f in features])
    iso_X = np.array([[f[name] for name in ISO_FOREST_INPUT_FEATURES] for f in features])
    iso_scores = iso_forest.decision_function(iso_X).reshape(-1, 1)
    X = np.hstack([base_X, iso_scores])

    probabilities = model.predict_proba(X)[:, 1]
    flagged_mask = probabilities >= threshold

    kept_df = valid_df.filter(pl.Series(~flagged_mask))
    flagged_rows = [row for row, flagged in zip(rows, flagged_mask, strict=True) if flagged]
    flagged_reasons = [
        [
            {
                "rule": "quarantine_classifier",
                "message": (
                    f"ML quarantine classifier flagged this row (score={score:.4f}, "
                    f"threshold={threshold}); it passed the Pydantic contract."
                ),
                "severity": "warning",
                "model_version": model_version,
            }
        ]
        for row, score, flagged in zip(rows, probabilities, flagged_mask, strict=True)
        if flagged
    ]

    run_logger.info(
        "quarantine classification: %d/%d rows flagged by model version %s",
        len(flagged_rows), len(rows), model_version,
    )
    return kept_df, flagged_rows, flagged_reasons


@task(retries=2, retry_delay_seconds=5)
def write_task(
    clean_records: list[dict], invalid_rows: list[dict], invalid_reasons: list[list[dict]]
) -> tuple[int, int]:
    run_logger = get_run_logger()
    engine = get_engine()
    clean_written = write_clean_records(engine, clean_records)
    quarantine_written = write_quarantine_records(engine, invalid_rows, invalid_reasons)
    run_logger.info(
        "write: %d written to clean.analytics_data, %d written to quarantine.records",
        clean_written, quarantine_written,
    )
    return clean_written, quarantine_written


@task
def drift_check_task(ingested_df: pl.DataFrame) -> dict:
    """Checks a synthetic, full-magnitude shifted copy of the just-ingested batch against the
    persisted reference profile (drift/reference_profile.json), as a stand-in for a real
    incoming batch -- see this module's docstring for why there's no real one to check against.
    Reuses the exact shift mechanism and per-column full-magnitude fractions tuned in Phase 4
    Task 2 (drift/build_batches.py) so this is provably the same "significant" shift already
    demonstrated there, not a new untested magnitude.
    """
    run_logger = get_run_logger()
    reference_profile = load_reference_profile()

    batch = ingested_df.clone()
    batch = shift_company_size(batch, FULL_SHIFT_FRACTION["company_size"], random.Random(SHIFT_SEEDS["company_size"]))
    batch = shift_state_null_rate(batch, FULL_SHIFT_FRACTION["state"], random.Random(SHIFT_SEEDS["state"]))
    batch = shift_description_length(
        batch, FULL_SHIFT_FRACTION["description"], random.Random(SHIFT_SEEDS["description"])
    )

    report = check_drift(reference_profile, batch)
    run_logger.warning(
        "drift check (synthetic demo batch): any_flagged=%s -- company_size=%s state=%s description_len=%s",
        report["any_flagged"],
        report["company_size"]["severity"],
        report["state_null_rate"]["severity"],
        report["description_len"]["severity"],
    )
    return report


@flow(name="lucidflow-pipeline")
def pipeline_flow(input_path: str = "data/intake/companies.csv", run_drift_check: bool = False) -> dict:
    df = ingest_task(input_path)
    valid_records, invalid_rows, invalid_reasons = validate_task(df)

    if valid_records:
        valid_df, removed_count = clean_task(valid_records)
        valid_df, _missingness_report = impute_task(valid_df)
        valid_df, ml_flagged_rows, ml_flagged_reasons = quarantine_classify_task(valid_df)
        invalid_rows = invalid_rows + ml_flagged_rows
        invalid_reasons = invalid_reasons + ml_flagged_reasons
        clean_records = valid_df.to_dicts()
    else:
        clean_records, removed_count = [], 0

    clean_written, quarantine_written = write_task(clean_records, invalid_rows, invalid_reasons)

    report = {
        "ingested_rows": df.height,
        "passed_validation": len(valid_records),
        "failed_validation": len(invalid_rows),
        "duplicates_removed": removed_count,
        "written_clean": clean_written,
        "written_quarantine": quarantine_written,
    }
    print("=== LucidFlow Pipeline Report ===")
    print(f"Ingested rows:                        {report['ingested_rows']}")
    print(f"Passed validation:                    {report['passed_validation']}")
    print(f"Failed validation:                    {report['failed_validation']}")
    print(f"Exact duplicates removed (pre-write):  {report['duplicates_removed']}")
    print(f"Written to clean.analytics_data:      {report['written_clean']}")
    print(f"Written to quarantine.records:        {report['written_quarantine']}")

    if run_drift_check:
        drift_report = drift_check_task(df)
        report["drift_report"] = drift_report
        if drift_report["any_flagged"]:
            logger.warning("drift flagged -- triggering retrain_flow")
            report["retrain_triggered"] = True
            retrain_flow()
        else:
            report["retrain_triggered"] = False

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LucidFlow pipeline as a Prefect flow.")
    parser.add_argument("input_path", nargs="?", default="data/intake/companies.csv")
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help=(
            "Run the post-pipeline drift check. Off by default: the check always flags "
            "(it compares a deliberately full-magnitude synthetic shift, not real incoming "
            "data -- see drift_check_task's docstring), so it always triggers the ~8-10 "
            "minute retrain_flow. Opt in deliberately, not on every routine run."
        ),
    )
    args = parser.parse_args()
    pipeline_flow(args.input_path, run_drift_check=args.check_drift)


if __name__ == "__main__":
    main()
