"""Builds the (column -> feature fingerprint) training table for the
column-type classifier, one row per column across the source CSVs.

    python -m lucidflow.models.column_type_classifier.build_dataset
"""

import json
from pathlib import Path

import polars as pl

from lucidflow.models.column_type_classifier.features import extract_column_features, sample_values

DATA_ROOT = Path("data/linkedin-job-postings")

# (relative path, row cap). Only postings.csv is large enough to need a cap.
SOURCE_FILES: list[tuple[str, int | None]] = [
    ("companies/companies.csv", None),
    ("companies/employee_counts.csv", None),
    ("companies/company_industries.csv", None),
    ("companies/company_specialities.csv", None),
    ("postings.csv", 200_000),
    ("jobs/benefits.csv", None),
    ("jobs/salaries.csv", None),
    ("mappings/industries.csv", None),
    ("mappings/skills.csv", None),
]

OUTPUT_PATH = Path("src/lucidflow/models/column_type_classifier/column_fingerprints.json")


def build_dataset() -> list[dict]:
    rows = []
    for rel_path, n_rows in SOURCE_FILES:
        path = DATA_ROOT / rel_path
        df = pl.read_csv(path, infer_schema=False, n_rows=n_rows)
        for column in df.columns:
            raw_values = df[column].to_list()
            features = extract_column_features(raw_values)
            rows.append(
                {
                    "source_file": rel_path,
                    "column": column,
                    "n_rows_sampled": len(raw_values),
                    "features": features,
                    "sample_values": sample_values(raw_values),
                }
            )
    return rows


def main() -> None:
    rows = build_dataset()
    OUTPUT_PATH.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} column fingerprints to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
