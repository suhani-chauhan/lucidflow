"""Compares an incoming batch against the persisted reference profile and
reports PSI (company_size, state-null-rate) and KS (description length)
drift per column, with the conventional none/moderate/significant banding
from `metrics.py`.

Grounding note (Task 0-style, see docs/entity_resolution_investigation.md
for the project's established pattern of stating this plainly): `companies.csv`
is one static snapshot with no timestamp column, so there is no real
longitudinal batch to monitor here. This module is exercised in
`build_batches.py` against batches with a *documented, synthetic* injected
shift (see `synthetic_shift.py`) to prove the PSI/KS wiring is correct and
sensitive to the kinds of shift it's designed to catch. It does NOT validate
real-world drift-detection performance or a "correct" threshold for this
pipeline in production -- there's no real drift in this dataset to calibrate
either of those against.
"""

import polars as pl

from lucidflow.drift.metrics import (
    ks_severity,
    ks_test,
    population_stability_index,
    psi_severity,
)
from lucidflow.drift.reference_profile import null_bucket_counts, value_counts


def check_drift(reference_profile: dict, batch_df: pl.DataFrame) -> dict:
    company_size_psi = population_stability_index(
        reference_profile["company_size_counts"], value_counts(batch_df["company_size"])
    )
    state_psi = population_stability_index(
        reference_profile["state_null_counts"], null_bucket_counts(batch_df["state"])
    )

    descriptions = batch_df["description"].to_list()
    batch_description_lens = [len(d) for d in descriptions if d is not None]
    ks_statistic, ks_pvalue = ks_test(
        reference_profile["description_len_values"], batch_description_lens
    )

    report = {
        "company_size": {
            "metric": "psi",
            "value": company_size_psi,
            "severity": psi_severity(company_size_psi),
        },
        "state_null_rate": {
            "metric": "psi",
            "value": state_psi,
            "severity": psi_severity(state_psi),
        },
        "description_len": {
            "metric": "ks",
            "statistic": ks_statistic,
            "p_value": ks_pvalue,
            "severity": ks_severity(ks_pvalue),
        },
    }
    report["any_flagged"] = any(
        entry["severity"] != "none" for entry in report.values() if isinstance(entry, dict)
    )
    return report
