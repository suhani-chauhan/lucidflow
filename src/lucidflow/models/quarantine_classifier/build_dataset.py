"""Builds the labeled training set for the quarantine classifier.

Real positives are ~0 in `companies.csv` (see Task 0's grounding report:
0/24,473 rows fail the Pydantic contract, and Isolation Forest's ~20% flag
rate on raw data turned out to be legitimate unusual-but-valid companies,
not corruption). So positives are synthesized by corrupting real, valid
rows with one of four documented corruption types (e-h; see corruption.py
for why a-d were dropped) applied to disjoint row samples -- every row gets
at most one corruption type, and every corrupted row carries its
`corruption_type` label so synthetic rows are never blended in as if real.

    python -m lucidflow.models.quarantine_classifier.build_dataset
"""

import json
import random
from pathlib import Path

import polars as pl

from lucidflow.models.quarantine_classifier.corruption import (
    corrupt_encoding,
    corrupt_null_storm,
    corrupt_truncation,
    corrupt_zip_state_mismatch,
)
from lucidflow.models.quarantine_classifier.features import extract_row_features
from lucidflow.models.quarantine_classifier.zip_state_reference import (
    build_zip3_sample_pool,
    build_zip3_state_reference,
    eligible_rows,
)

DATA_PATH = Path("data/linkedin-job-postings/companies/companies.csv")
OUTPUT_PATH = Path("src/lucidflow/models/quarantine_classifier/labeled_dataset.json")

N_PER_TYPE = 800
RANDOM_STATE = 42


def build_dataset() -> list[dict]:
    df = pl.read_csv(DATA_PATH, infer_schema=False)
    rng = random.Random(RANDOM_STATE)

    zip3_to_state = build_zip3_state_reference(df)
    zip3_pool = build_zip3_sample_pool(df, zip3_to_state)
    eligible_ids = set(eligible_rows(df, zip3_to_state)["company_id"].to_list())

    all_ids = df["company_id"].to_list()
    rows_by_id = {row["company_id"]: row for row in df.to_dicts()}

    # type g can only be applied to rows with a known ground-truth state (the eligible pool);
    # sample it first so e/f/h draw from what's left, keeping every row single-corrupted.
    remaining_ids = set(all_ids)
    zip_state_sample = rng.sample(sorted(eligible_ids), N_PER_TYPE)
    remaining_ids -= set(zip_state_sample)

    remaining_sorted = sorted(remaining_ids)
    rng.shuffle(remaining_sorted)
    encoding_sample = remaining_sorted[:N_PER_TYPE]
    truncation_sample = remaining_sorted[N_PER_TYPE : 2 * N_PER_TYPE]
    null_storm_sample = remaining_sorted[2 * N_PER_TYPE : 3 * N_PER_TYPE]
    corrupted_ids = set(zip_state_sample) | set(encoding_sample) | set(truncation_sample) | set(null_storm_sample)
    clean_ids = [cid for cid in all_ids if cid not in corrupted_ids]

    dataset = []

    for cid in zip_state_sample:
        corrupted, corruption_type = corrupt_zip_state_mismatch(
            dict(rows_by_id[cid]), rng, zip3_to_state, zip3_pool
        )
        dataset.append(_labeled_row(corrupted, corruption_type, is_synthetic=True))

    for cid in encoding_sample:
        corrupted, corruption_type = corrupt_encoding(dict(rows_by_id[cid]), rng)
        dataset.append(_labeled_row(corrupted, corruption_type, is_synthetic=True))

    for cid in truncation_sample:
        corrupted, corruption_type = corrupt_truncation(dict(rows_by_id[cid]), rng)
        dataset.append(_labeled_row(corrupted, corruption_type, is_synthetic=True))

    for cid in null_storm_sample:
        corrupted, corruption_type = corrupt_null_storm(dict(rows_by_id[cid]), rng)
        dataset.append(_labeled_row(corrupted, corruption_type, is_synthetic=True))

    for cid in clean_ids:
        dataset.append(_labeled_row(rows_by_id[cid], "clean", is_synthetic=False))

    for row in dataset:
        row["features"] = extract_row_features(row["_row"], zip3_to_state)
        del row["_row"]

    return dataset


def _labeled_row(row: dict, corruption_type: str, is_synthetic: bool) -> dict:
    return {
        "company_id": row["company_id"],
        "is_synthetic": is_synthetic,
        "corruption_type": corruption_type,
        "_row": row,
    }


def main() -> None:
    dataset = build_dataset()
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2))
    n_synthetic = sum(1 for r in dataset if r["is_synthetic"])
    print(f"Wrote {len(dataset)} labeled rows ({n_synthetic} synthetic, {len(dataset) - n_synthetic} clean) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
