"""Builds and persists the drift reference profile from a baseline batch.

Aggregate-only, like `column_type_classifier/column_fingerprints.json` --
category counts and a derived numeric-length array, never the raw
name/description/address text itself -- so unlike
`quarantine_classifier/labeled_dataset.json` this is safe to commit even
though `companies.csv` (the source) is not.
"""

import json
from pathlib import Path

import polars as pl

REFERENCE_PROFILE_PATH = Path(__file__).parent / "reference_profile.json"

PSI_COLUMNS = ["company_size", "state"]


def build_reference_profile(baseline_df: pl.DataFrame) -> dict:
    company_size_counts = value_counts(baseline_df["company_size"])
    state_null_counts = null_bucket_counts(baseline_df["state"])

    descriptions = baseline_df["description"].to_list()
    description_len_values = [len(d) for d in descriptions if d is not None]

    return {
        "n_rows": baseline_df.height,
        "company_size_counts": company_size_counts,
        "state_null_counts": state_null_counts,
        "description_len_values": description_len_values,
    }


def value_counts(series: pl.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series.to_list():
        key = "null" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def null_bucket_counts(series: pl.Series) -> dict[str, int]:
    n_null = series.null_count()
    return {"null": n_null, "non_null": series.len() - n_null}


def save_reference_profile(profile: dict, path: Path = REFERENCE_PROFILE_PATH) -> None:
    path.write_text(json.dumps(profile, indent=2))


def load_reference_profile(path: Path = REFERENCE_PROFILE_PATH) -> dict:
    return json.loads(path.read_text())
