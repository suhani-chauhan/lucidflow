"""Whitespace and Unicode cleanup for free-text columns.

Deliberately does not alter casing: fields like company name and country code
already carry meaningful casing in the source data (e.g. "IBM", "US"), and a
generic casing rule would corrupt acronyms/proper nouns rather than clean them.
"""

import unicodedata

import polars as pl


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split())  # collapse/strip whitespace
    return value or None


def normalize_text_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    exprs = [
        pl.col(column).map_elements(_normalize, return_dtype=pl.Utf8).alias(column)
        for column in columns
        if column in df.columns
    ]
    return df.with_columns(exprs) if exprs else df
