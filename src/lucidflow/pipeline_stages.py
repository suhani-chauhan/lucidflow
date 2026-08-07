"""Plain (non-Prefect) implementations of the pipeline's validation and cleaning
stages, shared between pipeline_flow.py (wraps these as @task) and
retrain_flow.py (calls them directly on a fresh load, so a retrain sees the
same validated/coerced/cleaned data the real pipeline would feed it -- not
raw, all-string CSV rows). Kept separate from flows/ specifically so
retrain_flow.py can import this without importing pipeline_flow.py (and
risking a circular import, since pipeline_flow.py imports retrain_flow.py
to trigger it on drift).
"""

import json

import polars as pl
from pydantic import ValidationError

from lucidflow.cleaning.dedup import remove_exact_duplicates
from lucidflow.cleaning.text_normalizer import normalize_text_columns
from lucidflow.cleaning.type_coercion import coerce_types
from lucidflow.validation.pydantic_models import Company

TEXT_COLUMNS = ["name", "description", "state", "country", "city", "address", "zip_code"]


def validate_rows(df: pl.DataFrame) -> tuple[list[dict], list[dict], list[list[dict]]]:
    """Validates each raw row against the Company contract.

    Returns (valid_records, invalid_raw_rows, invalid_reasons) where invalid_reasons[i]
    is the full list of every validation failure on invalid_raw_rows[i], not just the first.
    """
    valid_records: list[dict] = []
    invalid_rows: list[dict] = []
    invalid_reasons: list[list[dict]] = []

    for row in df.to_dicts():
        try:
            company = Company.model_validate(row)
            valid_records.append(json.loads(company.model_dump_json()))
        except ValidationError as exc:
            invalid_rows.append(row)
            invalid_reasons.append(
                [
                    {
                        "rule": ".".join(str(part) for part in error["loc"]) or "__root__",
                        "message": error["msg"],
                        "severity": "error",
                    }
                    for error in exc.errors()
                ]
            )

    return valid_records, invalid_rows, invalid_reasons


def clean_records(valid_records: list[dict]) -> tuple[pl.DataFrame, int]:
    valid_df = pl.DataFrame(valid_records)
    valid_df, removed_count = remove_exact_duplicates(valid_df)
    valid_df = coerce_types(valid_df, {"company_id": pl.Int64, "company_size": pl.Int64})
    valid_df = normalize_text_columns(valid_df, TEXT_COLUMNS)
    return valid_df, removed_count


def validate_and_clean(df: pl.DataFrame) -> pl.DataFrame:
    """Convenience wrapper for callers (e.g. retrain_flow) that just want validated,
    cleaned rows and don't need the invalid-rows/reasons detail."""
    valid_records, _invalid_rows, _invalid_reasons = validate_rows(df)
    if not valid_records:
        return pl.DataFrame(schema=df.schema)
    valid_df, _removed_count = clean_records(valid_records)
    return valid_df
