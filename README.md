# LucidFlow

A data quality platform that replaces hand-written cleaning rules with trained ML models, wrapped
in a production ETL pipeline: a semantic column-type classifier, a learned imputation selector,
and a quarantine classifier for corrupt-record detection — orchestrated with Prefect, tracked
with MLflow, and served through a Streamlit review dashboard.

**This repository has completed Phases 1-3.** Phase 1 built the ETL skeleton
(`ingestion -> validation -> cleaning -> routing`) backed by real Postgres schemas. Phase 2 added
the first two of LucidFlow's trained models — a semantic column-type classifier and a learned
imputation selector — both trained on the real
[LinkedIn Job Postings dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings).
Phase 3 added the third: a quarantine classifier trained on synthetic corruption injected into
real, valid rows (see below for why, and what it does and doesn't catch). Of the three, only the
imputation selector is actually wired into `run_pipeline.py`'s inference-time execution. The
column-type classifier is trained and tested standalone as the foundation for planned automatic
contract generation — Phase 1's hand-written `Company` contract
(`src/lucidflow/validation/pydantic_models.py`) is still what runs at inference time. The
quarantine classifier is likewise trained, tested, and benchmarked standalone, not yet wired into
the pipeline's write path. Wiring both in, along with drift monitoring, Prefect orchestration, and
MLflow tracking, is Phase 4 (MLOps). The Streamlit review dashboard is Phase 5. See the
`README.md` stub in each unimplemented folder for what phase it belongs to.

A fourth model — an LLM-distilled duplicate-pair classifier for entity resolution — was scoped
for Phase 3, investigated against the real dataset, and deliberately dropped: the data doesn't
support it (see [`docs/entity_resolution_investigation.md`](docs/entity_resolution_investigation.md)
for the full investigation with real examples and counts).

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

As of Phase 2, cleaning also runs a missingness-imputation stage (see below) between structural
cleaning and the write — its per-column strategy and benchmark scores print as part of every run.

## Phase 2 models

### Semantic column-type classifier

Infers a column's semantic type (`identifier`, `categorical`, `free_text`, `numeric_continuous`,
`geographic`, `date`, `url`, `boolean`) from its statistical fingerprint — character-class mix,
token-length stats, cardinality ratio, regex hit rates, numeric-parse rate — rather than its
header name. Trained on 64 columns pulled from 9 files across the LinkedIn Job Postings dataset
(`src/lucidflow/models/column_type_classifier/build_dataset.py`), with labels proposed from the
computed fingerprints and confirmed by hand
(`src/lucidflow/models/column_type_classifier/confirmed_labels.csv`) before training.

```bash
python -m lucidflow.models.column_type_classifier.build_dataset  # rebuild the fingerprint dataset
python -m lucidflow.models.column_type_classifier.train           # train + report
```

Held-out results (17/64 columns, split guarantees every class with ≥2 members gets ≥1 test
example — see `split.py`):

- **Macro-F1: 0.7446**
- 3/17 misclassified: a `location` free-text/geographic mix-up, a low-cardinality numeric column
  mistaken for an identifier, and a `date` column with only one other training example of its
  kind mistaken for `geographic`.
  These are small-n, explainable statistical confusions rather than something to fix:
  cardinality-based features can't fully separate "large count" from "ID," and epoch-timestamp
  dates share a fixed-length-digit signature with zip codes.
- **Known limitation**: the only `boolean` examples in this dataset are numeric-coded (`"0"`/`"1"`)
  — the model has never seen a text-coded (`"true"`/`"false"`) boolean and will currently
  mislabel one as `categorical`. Documented and regression-tested
  (`tests/test_column_type_classifier_model.py`), not silently wrong.

### Learned imputation selector

For each column with missing values in `companies.csv`, benchmarks median/mode, `KNNImputer`,
`IterativeImputer` (MICE), and a `LightGBM` classifier by masking known values and scoring
recovery (macro-F1), then auto-selects the winner and persists the fitted imputer. Not every
column fits that benchmark equally well, so treatment is decided per column based on its actual
distribution, not guessed uniformly:

| column | treatment | why |
|---|---|---|
| `company_size` | full benchmark, hard-coded ordinal | clean 7-code ordinal; ordinal-ness is structural knowledge the column-type classifier can't infer from stats alone, so it's hard-coded independent of that model's label |
| `country` | full benchmark, categorical | 80 clean ISO-2 codes (verified no near-duplicate encodings like `state` had) |
| `state` | mode-within-country fallback | real distribution is hundreds of singleton/dirty free-text values (non-ASCII province names, malformed entries) — too noisy to benchmark meaningfully despite looking like a reasonable class count on paper |
| `city`, `zip_code` | mode-within-(state, country) fallback | near-identifier cardinality, but genuinely recoverable via geographic correlation |
| `address`, `description` | left null, no fallback | ~unique per row / open-ended free text — no strategy in the benchmark suite meaningfully applies |

Benchmark results on real `companies.csv` (macro-F1):

| column | median/mode | knn | mice | lightgbm | winner | class coverage |
|---|---|---|---|---|---|---|
| `company_size` | 0.0358 | 0.1254 | 0.0493 | **0.1660** | lightgbm | 7/7 classes (100%) |
| `country` | 0.0180 | **0.0356** | 0.0013 | 0.0229 | knn | 53/80 classes (66.2%) |

**Known limitation on `country`**: 27 of 80 country codes have exactly one known example each —
structurally impossible to both train on and hold out for testing, so the macro-F1 above reflects
recovery only for the 53 classes with enough examples to evaluate. Performance on the other 27 is
unverified, not verified-good. This is why the benchmark uses a guaranteed-min-1-per-class split
(`stratified_min1_split`, shared with the column-type classifier) instead of sklearn's strict
stratified split, which rejects singleton classes outright. Every pipeline run prints this
coverage figure and the limitation note alongside the benchmark scores — see
`src/lucidflow/models/imputation_selector/selector.py`.

Fitted imputers are regenerated on every pipeline run (from `companies.csv`, which isn't
committed) and gitignored — unlike the column-type classifier's model, they're not durable
trained artifacts meant for reuse across runs.

## Phase 3 model: quarantine classifier

Flags corrupt/low-quality records that the Phase 1 Pydantic contract structurally can't catch —
e.g. plausible-looking text with encoding corruption, rather than a missing field or an
out-of-range value. Task 0 grounding found real `companies.csv` has **0/24,473 rows** failing
the contract, so real positive examples for a classifier don't exist. The classifier is trained
on synthetic corruption injected into real, valid rows instead — never blended silently with
real data, always carrying an `is_synthetic` flag and a `corruption_type` label.

Only four corruption types are used, chosen specifically because the existing Pydantic gate
**cannot** catch any of them (an earlier 8-type proposal included four more — missing field,
malformed URL, out-of-range value, type corruption — dropped because the gate already catches
those with perfect precision, so training on them would teach the classifier nothing new):

- **encoding** — mojibake-style character substitution (UTF-8 misread as cp1252), e.g. `'` → `â€™`
- **truncation** — text cut short mid-word plus a garbled trailing suffix
- **zip_state_mismatch** — real zip code swapped for one from a different state, `state` field
  left untouched (US rows only; the zip↔state reference is built empirically from
  `companies.csv` itself — no external geographic table — and verified 98.5% internally
  consistent before use)
- **null_storm** — 5-7 of 7 optional fields blanked, `company_id`/`name`/`url` left intact

```bash
python -m lucidflow.models.quarantine_classifier.build_dataset  # inject corruption, build labeled set
python -m lucidflow.models.quarantine_classifier.train           # train + report
```

An Isolation Forest is fit on train-only shape features and its anomaly score is folded in as one
input feature to a `LightGBM` classifier — features are deliberately generic quality signals
(missingness, text-shape, a regex-based mojibake-pattern detector built from the real cp1252
misdecode byte table) rather than one bespoke detector per corruption type, except
`zip_state_mismatch`, which is inherently cross-field.

Held-out results (4,895/24,473 rows, stratified by corruption type):

- **Aggregate PR-AUC: 0.9852** | precision/recall/F1 @ threshold=0.5: **0.9406 / 0.9656 / 0.9530**
- Clean-row false-positive rate: 39/4,255 (0.92%) — shared across every per-type view below, so
  per-type precision differences mostly reflect each type's own recall, not a different
  false-positive rate.

| corruption type | n | precision | recall | F1 |
|---|---|---|---|---|
| `zip_state_mismatch` | 160 | 0.804 | 1.000 | 0.891 |
| `null_storm` | 160 | 0.804 | 1.000 | 0.891 |
| `encoding` | 160 | 0.803 | 0.994 | 0.888 |
| `truncation` | 160 | 0.781 | 0.869 | 0.823 |

All four types are recovered well above chance on failure modes the Pydantic gate structurally
cannot check — real evidence the model adds value beyond the existing gate. (`encoding` recall
was initially the weak point at 0.306 with a plain non-ASCII-rate feature — accented names and
emoji in legitimate company text swamp that signal — and was fixed by replacing it with a regex
mojibake-pattern detector built from the actual cp1252 continuation-byte table, not just the
literal substitution strings the injector itself uses.)

The trained model (`quarantine_classifier.joblib`) is committed as a durable artifact, same as the
column-type classifier's. `labeled_dataset.json` is gitignored — it embeds verbatim rows from
`companies.csv`, itself not committed; regenerate via `build_dataset.py`.

**Not yet wired into `run_pipeline.py`** — the pipeline's quarantine routing today is driven
entirely by the Pydantic contract (Phase 1), not this model. Wiring it into the write path is
Phase 4 scope.

## Reference dataset

The Phase 1 contract (`Company`) is hand-written against the `companies.csv` table from the
[LinkedIn Job Postings dataset on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings).
Download it yourself and drop `companies.csv` into `data/intake/` — the dataset isn't committed
to this repo (see `.gitignore`).

The Phase 2 column-type classifier trains across more of the same dataset — drop the full
Kaggle download into `data/linkedin-job-postings/` (also gitignored) with its native layout:
`companies/companies.csv`, `companies/employee_counts.csv`, `companies/company_industries.csv`,
`companies/company_specialities.csv`, `postings.csv`, `jobs/benefits.csv`, `jobs/salaries.csv`,
`mappings/industries.csv`, `mappings/skills.csv` — see `SOURCE_FILES` in
`src/lucidflow/models/column_type_classifier/build_dataset.py` for the exact list.

## Tests

```bash
pytest
pytest --cov=src/lucidflow --cov-report=term-missing
ruff check .
```

## Project layout

```
src/lucidflow/
├── ingestion/                    # CSV/JSON loading (Polars)                    — Phase 1
├── profiling/                                                                   — Phase 2 (unused so far)
├── validation/                   # Pydantic data contracts                      — Phase 1
├── cleaning/                     # dedup, type coercion, text normalization     — Phase 1
├── models/
│   ├── column_type_classifier/   # semantic column-type classifier (RandomForest) — Phase 2, done
│   ├── imputation_selector/      # learned imputation benchmark + selector       — Phase 2, done
│   └── quarantine_classifier/    # gradient-boosted quarantine classifier — Phase 3, done (not pipeline-wired yet)
├── resolution/                   # entity resolution — investigated, not built (see docs/)
├── drift/                        # PSI/KS drift monitoring                      — Phase 4
├── loading/                      # dual-route Postgres writers                  — Phase 1
├── flows/                        # Prefect orchestration                        — Phase 4
└── observability/                                                               — Phase 4
dashboard/                        # Streamlit review UI                          — Phase 5
docs/
└── entity_resolution_investigation.md   # why the duplicate-pair classifier was dropped
```
