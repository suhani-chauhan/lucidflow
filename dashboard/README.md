# dashboard/

Phase 5, Task 2 done: read-only observability dashboard. Task 3 (human-in-the-loop quarantine
review) is next.

```bash
set -a; source .env; set +a
streamlit run dashboard/app.py
```

- `app.py` — Pipeline Summary (landing page): most recent run's clean/quarantine counts and
  quarantine-reason breakdown, read from Postgres.
- `pages/1_Model_Results.py` — latest MLflow run per model: column-type classifier confusion
  matrix, imputation benchmark table (with class-coverage caveats), quarantine classifier
  per-corruption-type precision/recall.
- `pages/2_Drift_Status.py` — the three-batch (none/moderate/significant) PSI/KS
  characterization from Phase 4, Task 2, read from `src/lucidflow/drift/last_check_results.json`.
- `pages/3_Entity_Resolution.py` — renders `docs/entity_resolution_investigation.md` directly,
  so the dashboard can't drift out of sync with the actual investigation writeup.
- `data_access.py` — all read-only data access (Postgres, MLflow, drift results, the doc).
  "Most recent pipeline run" is derived from Postgres write timestamps without any schema
  change or run_id column -- see the module docstring for why that works.

No new backend logic for this part, with one small exception: `last_check_results.json` didn't
previously exist (Phase 4's drift check only printed results) -- `build_batches.py` now also
persists what it already computes, same "save the aggregate, not raw text" pattern as
`reference_profile.json`.
