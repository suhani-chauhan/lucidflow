"""Practical MCAR/MAR missingness diagnostic for a single column.

Not a formal test (e.g. Little's MCAR test) — a lighter, defensible check:
for each other column, test whether the *fact* that `target_col` is missing
is statistically associated with that column's observed values. If any
association is significant, missingness is explainable by observed data
(MAR). If none are, we fall back to the practical assumption of MCAR — this
is an absence-of-evidence result, not proof, and is reported as such.
"""

import polars as pl
from scipy import stats

ALPHA = 0.05
MAX_PREDICTOR_CARDINALITY = 50  # above this, a chi2 contingency test is unreliable (sparse cells)


def diagnose_missingness(df: pl.DataFrame, target_col: str) -> dict:
    total = df.height
    is_null = df[target_col].is_null()
    missing_count = int(is_null.sum())

    associations = []
    for other_col in df.columns:
        if other_col == target_col:
            continue
        result = _test_association(df, target_col, other_col, is_null)
        if result is not None:
            associations.append(result)

    significant = [a for a in associations if a["significant"]]
    verdict = "MAR" if significant else "MCAR (practical - no association found)"

    return {
        "column": target_col,
        "missing_count": missing_count,
        "missing_ratio": missing_count / total if total else 0.0,
        "associations": associations,
        "verdict": verdict,
    }


def _test_association(df: pl.DataFrame, target_col: str, other_col: str, is_null: pl.Series) -> dict | None:
    other = df[other_col]
    valid_mask = other.is_not_null()
    if valid_mask.sum() < 2:
        return None

    is_null_np = is_null.filter(valid_mask).cast(pl.Int8).to_numpy()
    if len(set(is_null_np)) < 2:
        return None  # no variation in missingness within rows where other_col is observed

    if other.dtype.is_numeric():
        other_np = other.filter(valid_mask).cast(pl.Float64).to_numpy()
        statistic, p_value = stats.pointbiserialr(is_null_np, other_np)
        test = "point_biserial"
    else:
        n_unique = other.filter(valid_mask).n_unique()
        if n_unique < 2 or n_unique > MAX_PREDICTOR_CARDINALITY:
            return None  # too high-cardinality for a meaningful chi2 contingency test
        contingency = (
            pl.DataFrame({"is_null": is_null_np, "other": other.filter(valid_mask).to_list()})
            .pivot(index="is_null", on="other", values="other", aggregate_function="len")
            .fill_null(0)
        )
        table = contingency.drop("is_null").to_numpy()
        if table.shape[0] < 2 or table.shape[1] < 2:
            return None
        statistic, p_value, _, _ = stats.chi2_contingency(table)
        test = "chi2"

    return {
        "predictor": other_col,
        "test": test,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
    }
