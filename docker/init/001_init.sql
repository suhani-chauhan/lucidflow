-- LucidFlow — Phase 1 schema bootstrap.
-- Runs automatically on first container start via docker-entrypoint-initdb.d.

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS quarantine;

-- raw is reserved for future ingestion snapshots (Phase 2+); no tables yet.

CREATE TABLE IF NOT EXISTS clean.analytics_data (
    company_id    BIGINT PRIMARY KEY,
    name          TEXT,
    description   TEXT,
    company_size  SMALLINT,
    state         TEXT,
    country       TEXT,
    city          TEXT,
    zip_code      TEXT,
    address       TEXT,
    url           TEXT,
    loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quarantine.records (
    id              BIGSERIAL PRIMARY KEY,
    raw_data        JSONB NOT NULL,
    reasons         JSONB NOT NULL,
    quarantined_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
