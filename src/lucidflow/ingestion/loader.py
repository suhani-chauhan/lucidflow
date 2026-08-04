"""Reads a single CSV/JSON file from a local path into a Polars DataFrame."""

from pathlib import Path

import polars as pl


def load_file(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix == ".csv":
        # Read every column as a string: type inference from a sample of rows
        # is unreliable (e.g. a numeric-looking column like zip_code can be
        # all-digits in one file and alphanumeric in another), and it isn't
        # the ingestion layer's job to guess types anyway — the validation
        # layer (Pydantic contract) owns coercion into real types.
        return pl.read_csv(path, infer_schema=False)
    if path.suffix == ".json":
        return pl.read_json(path)

    raise ValueError(f"Unsupported file type: {path.suffix}")
