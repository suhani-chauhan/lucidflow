"""LucidFlow — Phase 1 pipeline entry point.

Chains: ingestion -> validation -> cleaning -> routing, on one input file.

    python run_pipeline.py [path/to/file.csv]

Defaults to data/intake/companies.csv if no path is given.
"""

import argparse
import json
import logging

import polars as pl
from pydantic import ValidationError

from lucidflow.cleaning.dedup import remove_exact_duplicates
from lucidflow.cleaning.text_normalizer import normalize_text_columns
from lucidflow.cleaning.type_coercion import coerce_types
from lucidflow.ingestion.loader import load_file
from lucidflow.loading.db import get_engine
from lucidflow.loading.postgres_writer import write_clean_records
from lucidflow.loading.quarantine_writer import write_quarantine_records
from lucidflow.validation.pydantic_models import Company

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("lucidflow.run_pipeline")

TEXT_COLUMNS = ["name", "description", "state", "country", "city", "address", "zip_code"]


def validate_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[list[dict]]]:
    """Validate each raw row against the Company contract.

    Returns (valid_records, invalid_raw_rows, invalid_reasons) where invalid_reasons[i]
    is the full list of every validation failure on invalid_raw_rows[i], not just the first.
    """
    valid_records: list[dict] = []
    invalid_rows: list[dict] = []
    invalid_reasons: list[list[dict]] = []

    for row in rows:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LucidFlow core-spine pipeline on one file.")
    parser.add_argument("input_path", nargs="?", default="data/intake/companies.csv")
    args = parser.parse_args()

    # 1. Ingestion
    df = load_file(args.input_path)
    logger.info("ingestion: loaded %d rows from %s", df.height, args.input_path)

    # 2. Validation (against the raw, as-ingested values)
    valid_records, invalid_rows, invalid_reasons = validate_rows(df.to_dicts())
    logger.info("validation: %d passed, %d failed", len(valid_records), len(invalid_rows))

    # 3. Cleaning (applied to the validated set before it's written to clean.analytics_data)
    if valid_records:
        valid_df = pl.DataFrame(valid_records)
        valid_df, removed_count = remove_exact_duplicates(valid_df)
        valid_df = coerce_types(valid_df, {"company_id": pl.Int64, "company_size": pl.Int64})
        valid_df = normalize_text_columns(valid_df, TEXT_COLUMNS)
        clean_records = valid_df.to_dicts()
    else:
        clean_records = []
        removed_count = 0

    # 4. Routing
    engine = get_engine()
    clean_written = write_clean_records(engine, clean_records)
    quarantine_written = write_quarantine_records(engine, invalid_rows, invalid_reasons)

    print("=== LucidFlow Pipeline Report ===")
    print(f"Ingested rows:                        {df.height}")
    print(f"Passed validation:                    {len(valid_records)}")
    print(f"Failed validation:                    {len(invalid_rows)}")
    print(f"Exact duplicates removed (pre-write):  {removed_count}")
    print(f"Written to clean.analytics_data:      {clean_written}")
    print(f"Written to quarantine.records:        {quarantine_written}")


if __name__ == "__main__":
    main()
