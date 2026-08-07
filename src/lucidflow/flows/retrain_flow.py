"""Retrain subflow, triggered when the pipeline's drift check flags a shift
(see `drift_check_task` in pipeline_flow.py).

Retrains the two models whose training data the drift-tested columns
(company_size, state, description -- see drift/synthetic_shift.py) actually
feed: the imputation selector (company_size, state) and the quarantine
classifier (description-derived features, state). The column-type
classifier's training data is column statistical fingerprints across 9
separate files, untouched by this drift signal, so it's not included here.

Each retrain re-runs the model's existing training entry point unchanged --
this flow only orchestrates *when* they run, not *how* they train -- so
every retrain here logs to MLflow exactly like a manual `python -m
lucidflow.models...train` run would (see mlflow_config.py, Phase 4 Task 1).

    python -m lucidflow.flows.retrain_flow
"""

from prefect import flow, get_run_logger, task

from lucidflow.ingestion.loader import load_file
from lucidflow.models.imputation_selector.selector import run_missingness_engine
from lucidflow.models.quarantine_classifier import build_dataset as quarantine_build_dataset
from lucidflow.models.quarantine_classifier import train as quarantine_train
from lucidflow.pipeline_stages import validate_and_clean

IMPUTATION_SOURCE_PATH = "data/intake/companies.csv"


@task(retries=1, retry_delay_seconds=10)
def retrain_imputation_selector_task() -> list[dict]:
    """Re-benchmarks on a fresh load of the same real data the pipeline uses -- run through
    the same validate + clean steps the real pipeline applies before imputation, not the raw,
    all-string CSV rows (company_size etc. need to already be coerced to numeric for the
    benchmark strategies to run)."""
    run_logger = get_run_logger()
    df = load_file(IMPUTATION_SOURCE_PATH)
    valid_df = validate_and_clean(df)
    _, report = run_missingness_engine(valid_df)
    run_logger.info("retrain: imputation selector re-benchmarked %d columns", len(report))
    return report


@task(retries=1, retry_delay_seconds=10)
def retrain_quarantine_classifier_task() -> None:
    run_logger = get_run_logger()
    quarantine_build_dataset.main()
    quarantine_train.main()
    run_logger.info("retrain: quarantine classifier rebuilt its synthetic-corruption dataset and retrained")


@flow(name="lucidflow-retrain")
def retrain_flow() -> dict:
    run_logger = get_run_logger()
    run_logger.warning("retrain_flow triggered")

    imputation_report = retrain_imputation_selector_task()
    retrain_quarantine_classifier_task()

    return {"imputation_report": imputation_report, "quarantine_classifier_retrained": True}


if __name__ == "__main__":
    retrain_flow()
