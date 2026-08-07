"""Builds the Task 2 grounding demonstration: a baseline reference profile plus
three "incoming" batches, all built from the *same* 6,000-row pool sampled from
`companies.csv` -- Batch A (pool, unmodified), Batch B (full synthetic shift),
Batch C (half synthetic shift) -- and reports PSI/KS drift for each against
the baseline. Holding the underlying rows fixed and varying only the shift
magnitude isolates the magnitude's effect instead of mixing it with sampling
noise from three independent draws.

`companies.csv` has no timestamp column, so the baseline and the incoming
pool are independent random draws from the same static snapshot rather than
a real non-overlapping sequential stream -- overlap between the baseline and
the pool is expected, not a bug (approved: these are independent hypothetical
batches for demonstration, not a real partition). See `synthetic_shift.py`
and `monitor.py` docstrings for what this is, and isn't, evidence of.

Per-column full-shift magnitudes (the fraction of the pool's rows mutated at
Batch B) were picked empirically so severity lands significant at full
magnitude and moderate at half -- see the module docstring in
`synthetic_shift.py` for the mechanism; the specific fractions below are not
representative of any real shift size, only tuned to demonstrate the
detector's three severity bands on this dataset.

    python -m lucidflow.drift.build_batches
"""

import random

import polars as pl

from lucidflow.drift.monitor import check_drift
from lucidflow.drift.reference_profile import build_reference_profile, save_reference_profile
from lucidflow.drift.synthetic_shift import (
    shift_company_size,
    shift_description_length,
    shift_state_null_rate,
)

DATA_PATH = "data/intake/companies.csv"

BASELINE_SEED = 42
BASELINE_SIZE = 12_000

POOL_SEED = 99
POOL_SIZE = 6_000

# Fraction of the pool's rows mutated at magnitude=1.0 ("Batch B" / full shift),
# tuned so PSI/KS severity lands significant at full and moderate at half.
FULL_SHIFT_FRACTION = {"company_size": 0.60, "state": 0.08, "description": 0.07}
SHIFT_SEEDS = {"company_size": 202, "state": 201, "description": 203}

BATCH_MAGNITUDES = {"A": 0.0, "B": 1.0, "C": 0.5}


def build_batch(pool: pl.DataFrame, magnitude_scale: float) -> pl.DataFrame:
    batch = pool.clone()
    batch = shift_company_size(
        batch, FULL_SHIFT_FRACTION["company_size"] * magnitude_scale,
        random.Random(SHIFT_SEEDS["company_size"]),
    )
    batch = shift_state_null_rate(
        batch, FULL_SHIFT_FRACTION["state"] * magnitude_scale,
        random.Random(SHIFT_SEEDS["state"]),
    )
    batch = shift_description_length(
        batch, FULL_SHIFT_FRACTION["description"] * magnitude_scale,
        random.Random(SHIFT_SEEDS["description"]),
    )
    return batch


def main() -> None:
    df = pl.read_csv(DATA_PATH, infer_schema=False)

    baseline_df = df.sample(n=BASELINE_SIZE, seed=BASELINE_SEED)
    reference_profile = build_reference_profile(baseline_df)
    save_reference_profile(reference_profile)
    print(f"Reference profile built from {baseline_df.height} baseline rows, saved.\n")

    pool = df.sample(n=POOL_SIZE, seed=POOL_SEED)

    for label, scale in BATCH_MAGNITUDES.items():
        batch_df = build_batch(pool, scale)
        report = check_drift(reference_profile, batch_df)

        print(f"=== Batch {label} (magnitude_scale={scale}) ===")
        for column in ("company_size", "state_null_rate", "description_len"):
            entry = report[column]
            if entry["metric"] == "psi":
                print(f"  {column:18s} PSI={entry['value']:.4f}  severity={entry['severity']}")
            else:
                print(
                    f"  {column:18s} KS_stat={entry['statistic']:.4f}  "
                    f"p_value={entry['p_value']:.6f}  severity={entry['severity']}"
                )
        print(f"  any_flagged: {report['any_flagged']}\n")


if __name__ == "__main__":
    main()
