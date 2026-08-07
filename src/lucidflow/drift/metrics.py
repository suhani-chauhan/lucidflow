"""PSI (categorical) and KS-test (continuous) drift metrics, plus the
conventional textbook thresholds used to bucket a score into
no-shift/moderate/significant. These thresholds are not calibrated against
this project's data -- there's no real drift here to calibrate against (see
`monitor.py` module docstring) -- they're the standard values used across
the industry for PSI and a comparable p-value banding applied to KS for
consistency.
"""

import math

from scipy.stats import ks_2samp

PSI_EPSILON = 1e-4

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25

KS_MODERATE_P = 0.05
KS_SIGNIFICANT_P = 0.01


def population_stability_index(baseline_counts: dict[str, int], batch_counts: dict[str, int]) -> float:
    """PSI over the union of categories seen in either distribution. Categories missing from
    one side are treated as zero-count (then floored by PSI_EPSILON to avoid log(0)/div-by-0).
    """
    categories = sorted(set(baseline_counts) | set(batch_counts))
    baseline_total = sum(baseline_counts.values())
    batch_total = sum(batch_counts.values())

    psi = 0.0
    for category in categories:
        baseline_pct = max(baseline_counts.get(category, 0) / baseline_total, PSI_EPSILON)
        batch_pct = max(batch_counts.get(category, 0) / batch_total, PSI_EPSILON)
        psi += (batch_pct - baseline_pct) * math.log(batch_pct / baseline_pct)
    return psi


def psi_severity(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate"
    return "none"


def ks_test(baseline_values: list[float], batch_values: list[float]) -> tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    result = ks_2samp(baseline_values, batch_values)
    return float(result.statistic), float(result.pvalue)


def ks_severity(p_value: float) -> str:
    if p_value < KS_SIGNIFICANT_P:
        return "significant"
    if p_value < KS_MODERATE_P:
        return "moderate"
    return "none"
