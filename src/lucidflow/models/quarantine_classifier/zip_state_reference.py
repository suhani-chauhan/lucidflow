"""Empirically-derived ZIP3-prefix -> dominant-state mapping for US rows.

Built directly from real `companies.csv` rows, not a bundled external table —
no external geographic data source is used. `US_STATE_ABBREVS` below is a
fixed, standard list of USPS state codes used purely as a *validity filter*
on the messy `state` column (see Phase 2's imputation-selector investigation
into how noisy `state` actually is) — it carries no geographic correlation
information itself. The zip3 -> state correlation is learned from the data.

Used both to inject realistic zip/state cross-field corruption and as a
mismatch-detection feature at inference time.
"""

import polars as pl

US_STATE_ABBREVS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
    "VA", "WA", "WV", "WI", "WY", "DC",
}


def _clean_us_rows(df: pl.DataFrame) -> pl.DataFrame:
    """US rows with a validated USPS state abbreviation and a well-formed zip3 prefix."""
    us = (
        df.filter(pl.col("country") == "US")
        .filter(pl.col("state").is_not_null())
        .filter(pl.col("zip_code").is_not_null())
        .with_columns(pl.col("state").str.to_uppercase().str.strip_chars().alias("state_clean"))
        .filter(pl.col("state_clean").is_in(list(US_STATE_ABBREVS)))
        .with_columns(pl.col("zip_code").str.slice(0, 3).alias("zip3"))
        .filter(pl.col("zip3").str.contains(r"^\d{3}$"))
    )
    return us


def build_zip3_state_reference(df: pl.DataFrame, min_count: int = 2) -> dict[str, str]:
    """zip3 -> dominant state, for zip3 prefixes seen at least `min_count` times."""
    us = _clean_us_rows(df)
    dominant = (
        us.group_by("zip3")
        .agg(pl.col("state_clean").mode().first().alias("dominant_state"), pl.len().alias("n"))
        .filter(pl.col("n") >= min_count)
    )
    return dict(zip(dominant["zip3"].to_list(), dominant["dominant_state"].to_list(), strict=True))


def build_zip3_sample_pool(df: pl.DataFrame, zip3_to_state: dict[str, str]) -> dict[str, list[str]]:
    """zip3 -> list of real observed zip_code values for that prefix, for realistic swaps."""
    us = _clean_us_rows(df).filter(pl.col("zip3").is_in(list(zip3_to_state.keys())))
    pool: dict[str, list[str]] = {}
    for zip3, zip_code in zip(us["zip3"].to_list(), us["zip_code"].to_list(), strict=True):
        pool.setdefault(zip3, []).append(zip_code)
    return pool


def eligible_rows(df: pl.DataFrame, zip3_to_state: dict[str, str]) -> pl.DataFrame:
    """Rows with a known ground-truth state (via the reference), i.e. safe to corrupt for type g."""
    return _clean_us_rows(df).filter(pl.col("zip3").is_in(list(zip3_to_state.keys())))
