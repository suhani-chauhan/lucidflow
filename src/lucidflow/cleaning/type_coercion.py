"""Basic column-level type casting."""

import polars as pl


def coerce_types(df: pl.DataFrame, schema: dict[str, pl.DataType | type[pl.DataType]]) -> pl.DataFrame:
    """Cast the given columns to target Polars dtypes. Values that don't fit become null."""
    exprs = [
        pl.col(column).cast(dtype, strict=False).alias(column)
        for column, dtype in schema.items()
        if column in df.columns
    ]
    return df.with_columns(exprs) if exprs else df
