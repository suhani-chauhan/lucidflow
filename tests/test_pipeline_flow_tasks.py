import logging
from unittest.mock import patch

import polars as pl

from lucidflow.flows.pipeline_flow import (
    clean_task,
    drift_check_task,
    ingest_task,
    quarantine_classify_task,
    validate_task,
    write_task,
)
from lucidflow.flows.retrain_flow import (
    retrain_imputation_selector_task,
    retrain_quarantine_classifier_task,
)

_TEST_LOGGER = logging.getLogger("test")


def _patched(module_path):
    return patch(module_path, return_value=_TEST_LOGGER)


def _raw_row(company_id, url="https://www.linkedin.com/company/acme", **overrides):
    row = {
        "company_id": company_id,
        "name": "Acme Corp",
        "description": "We build widgets for everyone, a real company description.",
        "company_size": "3",
        "state": "NY",
        "country": "US",
        "city": "New York",
        "zip_code": "10001",
        "address": "123 Main St",
        "url": url,
    }
    row.update(overrides)
    return row


def test_ingest_and_write_tasks_are_configured_with_retries():
    # ingestion and the DB write are the two stages most likely to hit a transient
    # failure (file lock, connection blip) -- both get real retries.
    assert ingest_task.retries == 2
    assert write_task.retries == 2


def test_transform_tasks_are_configured_with_retries():
    assert validate_task.retries == 1
    assert clean_task.retries == 1


def test_retrain_tasks_are_configured_with_retries():
    assert retrain_imputation_selector_task.retries == 1
    assert retrain_quarantine_classifier_task.retries == 1


def test_validate_task_splits_valid_and_invalid_rows_with_full_reasons():
    df = pl.DataFrame(
        [
            _raw_row("1009"),
            _raw_row("1010", url="not-a-url", company_size="99"),
        ]
    )

    with _patched("lucidflow.flows.pipeline_flow.get_run_logger"):
        valid, invalid, reasons = validate_task.fn(df)

    assert len(valid) == 1
    assert len(invalid) == 1
    rules = {r["rule"] for r in reasons[0]}
    assert rules == {"url", "company_size"}


def test_clean_task_dedups_and_coerces_types():
    valid_records = [_raw_row("1009"), _raw_row("1009")]  # exact duplicate

    with _patched("lucidflow.flows.pipeline_flow.get_run_logger"):
        cleaned_df, removed_count = clean_task.fn(valid_records)

    assert removed_count == 1
    assert cleaned_df.height == 1
    assert cleaned_df["company_id"].dtype == pl.Int64
    assert cleaned_df["company_size"].dtype == pl.Int64


def test_quarantine_classify_task_partitions_rows_and_tags_flagged_reasons():
    valid_df = pl.DataFrame(
        [
            {**_raw_row(str(cid)), "company_id": cid, "company_size": 3}
            for cid in range(1, 11)
        ]
    )

    with _patched("lucidflow.flows.pipeline_flow.get_run_logger"):
        kept_df, flagged_rows, flagged_reasons = quarantine_classify_task.fn(valid_df)

    assert kept_df.height + len(flagged_rows) == valid_df.height
    assert len(flagged_rows) == len(flagged_reasons)
    for row_reasons in flagged_reasons:
        assert len(row_reasons) == 1
        assert row_reasons[0]["rule"] == "quarantine_classifier"
        assert row_reasons[0]["severity"] == "warning"
        assert "model_version" in row_reasons[0]


def test_quarantine_classify_task_handles_an_empty_batch():
    empty_df = pl.DataFrame(
        schema={
            "company_id": pl.Int64, "name": pl.Utf8, "description": pl.Utf8, "company_size": pl.Int64,
            "state": pl.Utf8, "country": pl.Utf8, "city": pl.Utf8, "zip_code": pl.Utf8,
            "address": pl.Utf8, "url": pl.Utf8,
        }
    )

    with _patched("lucidflow.flows.pipeline_flow.get_run_logger"):
        kept_df, flagged_rows, flagged_reasons = quarantine_classify_task.fn(empty_df)

    assert kept_df.height == 0
    assert flagged_rows == []
    assert flagged_reasons == []


def test_drift_check_task_returns_the_expected_report_shape():
    df = pl.DataFrame([_raw_row(str(n)) for n in range(1, 21)])

    with _patched("lucidflow.flows.pipeline_flow.get_run_logger"):
        report = drift_check_task.fn(df)

    assert set(report) >= {"company_size", "state_null_rate", "description_len", "any_flagged"}
    assert report["company_size"]["metric"] == "psi"
    assert report["description_len"]["metric"] == "ks"
    assert isinstance(report["any_flagged"], bool)
