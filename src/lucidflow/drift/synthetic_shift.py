"""Synthetic distribution shifts injected into demo "incoming" batches, for
Task 2's grounding demonstration only (see `build_batches.py`). `companies.csv`
has no timestamp column, so there is no real longitudinal drift to detect --
these shifts exist purely to prove the PSI/KS machinery fires when it should
and stays quiet when it shouldn't. Never blended silently with real data:
every shifted batch is built from a copy of real, valid rows and the shift
applied is documented here, not inferred from the data.

One mechanism, one magnitude knob: for each shift type, `magnitude` is the
fraction of the batch's rows that get the shift-specific mutation applied
(the rest of the batch is left as plain, unmodified real data). Halving
`magnitude` halves how many rows carry the shift -- a direct, literal
reading of "half the injected shift magnitude," applied the same way across
all three columns rather than three different bespoke formulas.

    magnitude=1.0 (full shift, "Batch B")   -> intended to land PSI/KS in the
                                                significant band
    magnitude=0.5 (half shift, "Batch C")   -> intended to land PSI/KS in the
                                                moderate band
    magnitude=0.0 (no shift, "Batch A")     -> not applied at all
"""

import random

import polars as pl

# Deliberately skewed toward large companies vs. the real baseline (~14% in
# codes 6-7) -- codes 6+7 get 55% of the weight here, a shift a domain expert
# would recognize immediately as "we started onboarding a different segment."
_SKEWED_COMPANY_SIZE_WEIGHTS = {1: 0.05, 2: 0.05, 3: 0.10, 4: 0.10, 5: 0.15, 6: 0.25, 7: 0.30}

_TRUNCATION_MIN_FRAC = 0.15
_TRUNCATION_MAX_FRAC = 0.5


def shift_company_size(df: pl.DataFrame, magnitude: float, rng: random.Random) -> pl.DataFrame:
    """Re-draws `company_size` on a `magnitude` fraction of rows (including originally-null
    ones) from `_SKEWED_COMPANY_SIZE_WEIGHTS`, leaving the rest of the batch untouched.
    """
    if magnitude <= 0:
        return df

    codes = list(_SKEWED_COMPANY_SIZE_WEIGHTS.keys())
    weights = list(_SKEWED_COMPANY_SIZE_WEIGHTS.values())

    values = df["company_size"].to_list()
    n_affected = int(len(values) * magnitude)
    affected_idx = set(rng.sample(range(len(values)), n_affected))

    new_values = [
        str(rng.choices(codes, weights=weights, k=1)[0]) if i in affected_idx else v
        for i, v in enumerate(values)
    ]
    return df.with_columns(pl.Series("company_size", new_values))


def shift_state_null_rate(df: pl.DataFrame, magnitude: float, rng: random.Random) -> pl.DataFrame:
    """Forces `state` to null on a `magnitude` fraction of rows (including rows that
    already happen to be null), inflating the batch's null rate toward `magnitude`.
    """
    if magnitude <= 0:
        return df

    n = df.height
    n_affected = int(n * magnitude)
    affected_idx = set(rng.sample(range(n), n_affected))

    values = df["state"].to_list()
    new_values = [None if i in affected_idx else v for i, v in enumerate(values)]
    return df.with_columns(pl.Series("state", new_values))


def shift_description_length(df: pl.DataFrame, magnitude: float, rng: random.Random) -> pl.DataFrame:
    """Truncates `description` to a random 15-50% of its original length on a `magnitude`
    fraction of rows with non-null description, shortening the batch's length distribution.
    No garbled suffix is appended (unlike the quarantine classifier's truncation corruption)
    -- this shift is only meant to move the length distribution, not simulate a corrupt record.
    """
    if magnitude <= 0:
        return df

    descriptions = df["description"].to_list()
    non_null_idx = [i for i, d in enumerate(descriptions) if d is not None]
    n_affected = int(len(non_null_idx) * magnitude)
    affected_idx = set(rng.sample(non_null_idx, n_affected))

    new_descriptions = list(descriptions)
    for i in affected_idx:
        text = new_descriptions[i]
        cut_frac = rng.uniform(_TRUNCATION_MIN_FRAC, _TRUNCATION_MAX_FRAC)
        cut_at = max(1, int(len(text) * cut_frac))
        new_descriptions[i] = text[:cut_at]

    return df.with_columns(pl.Series("description", new_descriptions))
