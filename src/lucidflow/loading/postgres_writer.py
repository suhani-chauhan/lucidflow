"""Writes validated, cleaned records to clean.analytics_data."""

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_INSERT_SQL = text(
    """
    INSERT INTO clean.analytics_data
        (company_id, name, description, company_size, state, country, city, zip_code, address, url)
    VALUES
        (:company_id, :name, :description, :company_size, :state, :country, :city, :zip_code, :address, :url)
    ON CONFLICT (company_id) DO NOTHING
    """
)


def write_clean_records(engine: Engine, records: list[dict]) -> int:
    """Insert one row per record. Returns the number of rows actually written
    (re-running the pipeline on the same file will not duplicate rows)."""
    if not records:
        return 0
    written = 0
    with engine.begin() as conn:
        for record in records:
            result = conn.execute(_INSERT_SQL, record)
            written += result.rowcount
    logger.info("clean writer: wrote %d/%d records to clean.analytics_data", written, len(records))
    return written
