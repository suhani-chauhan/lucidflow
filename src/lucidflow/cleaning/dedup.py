"""Exact-duplicate row removal."""

import logging

import polars as pl

logger = logging.getLogger(__name__)


def remove_exact_duplicates(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Drop rows that are identical across every column, keeping the first occurrence."""
    before = df.height
    deduped = df.unique(keep="first", maintain_order=True)
    removed = before - deduped.height
    logger.info("dedup: removed %d exact duplicate rows (%d -> %d)", removed, before, deduped.height)
    return deduped, removed
