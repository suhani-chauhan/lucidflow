# LucidFlow

A data quality platform that replaces hand-written cleaning rules with trained ML models,
wrapped in a production ETL pipeline: a semantic column-type classifier, a learned imputation
selector, an LLM-distilled duplicate-pair classifier for entity resolution, and a quarantine
classifier for corrupt-record detection — orchestrated with Prefect, tracked with MLflow, and
served through a Streamlit review dashboard.

**This repository is currently at Phase 1: "Core Spine."** None of the ML models are built yet
(see the `README.md` stub in each unimplemented folder for what phase it belongs to). What's
here is the working, testable ETL skeleton the ML phases will plug into:
`ingestion -> validation -> cleaning -> routing`, backed by real Postgres schemas.

## Requirements

- Python 3.11+
- Docker + Docker Compose (for Postgres)

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\Activate.ps1 for PowerShell

# 2. Install the project
pip install -e ".[dev]"

# 3. Configure environment variables
cp .env.example .env
# edit .env if you need to change ports/credentials (e.g. if 5432 is already taken locally)

# 4. Start Postgres (creates the raw / clean / quarantine schemas via docker/init/001_init.sql)
docker compose up -d
```

## Running the pipeline

Export the variables in `.env` into your shell (so the Python process and `docker compose` agree
on connection details), then run the entry point against a CSV in `data/intake/`:

```bash
set -a; source .env; set +a
python run_pipeline.py data/intake/companies.csv
```

If no path is given, it defaults to `data/intake/companies.csv`. A small synthetic file with
deliberately bad rows is included at `data/intake/demo_dirty_sample.csv` to exercise the
quarantine path:

```bash
python run_pipeline.py data/intake/demo_dirty_sample.csv
```

Each run prints a report:

```
=== LucidFlow Pipeline Report ===
Ingested rows:                        5
Passed validation:                    2
Failed validation:                    3
Exact duplicates removed (pre-write):  1
Written to clean.analytics_data:      1
Written to quarantine.records:        3
```

Rows that pass the `Company` data contract (`src/lucidflow/validation/pydantic_models.py`) land
in `clean.analytics_data`. Rows that fail land in `quarantine.records`, with a `reasons` JSONB
column holding every validation failure for that row (not just the first), each as
`{"rule": ..., "message": ..., "severity": ...}`.

## Reference dataset

The Phase 1 contract (`Company`) is hand-written against the `companies.csv` table from the
[LinkedIn Job Postings dataset on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings).
Download it yourself and drop `companies.csv` into `data/intake/` — the dataset isn't committed
to this repo (see `.gitignore`).

## Tests

```bash
pytest
ruff check .
```

## Project layout

```
src/lucidflow/
├── ingestion/       # CSV/JSON loading (Polars)          — Phase 1
├── profiling/                                             — Phase 2
├── validation/      # Pydantic data contracts             — Phase 1
├── cleaning/        # dedup, type coercion, text normalization — Phase 1
├── models/          # column-type / imputation / duplicate / quarantine classifiers — Phase 2-3
├── resolution/       # entity resolution                  — Phase 3
├── drift/            # PSI/KS drift monitoring             — Phase 4
├── loading/          # dual-route Postgres writers         — Phase 1
├── flows/            # Prefect orchestration               — Phase 4
└── observability/                                          — Phase 4
dashboard/            # Streamlit review UI                 — Phase 4
```
