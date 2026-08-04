"""Writes records that fail validation to quarantine.records.

`reasons` is a JSONB array of {rule, message, severity} objects — one entry
per validation failure on that record, not just the first.
"""

import json
import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_INSERT_SQL = text(
    """
    INSERT INTO quarantine.records (raw_data, reasons)
    VALUES (CAST(:raw_data AS JSONB), CAST(:reasons AS JSONB))
    """
)


def write_quarantine_records(engine: Engine, raw_rows: list[dict], reasons: list[list[dict]]) -> int:
    """Insert one row per rejected record, pairing each raw row with its list of failures."""
    if not raw_rows:
        return 0
    written = 0
    with engine.begin() as conn:
        for raw_row, row_reasons in zip(raw_rows, reasons):
            result = conn.execute(
                _INSERT_SQL,
                {
                    "raw_data": json.dumps(raw_row, default=str),
                    "reasons": json.dumps(row_reasons),
                },
            )
            written += result.rowcount
    logger.info("quarantine writer: wrote %d/%d records to quarantine.records", written, len(raw_rows))
    return written
