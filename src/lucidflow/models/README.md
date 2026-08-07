# models/

Each subfolder is one of LucidFlow's trained models:

- `column_type_classifier/` — Phase 2, done: semantic column-type classifier, trained as the
  foundation for planned automatic contract generation. Not wired up yet — Phase 1's hand-written
  `Company` contract (`src/lucidflow/validation/pydantic_models.py`) is still what runs at
  inference time.
- `imputation_selector/` — Phase 2, done: learned imputation-method selector (median/KNN/MICE/LightGBM benchmarking).
- `quarantine_classifier/` — Phase 3, done: gradient-boosted quarantine classifier trained on synthetic corruption
  (real positives are ~0 in `companies.csv`). Trained and benchmarked standalone; wiring it into `run_pipeline.py`'s
  write path and retraining it from dashboard review are Phase 4/5 scope. See root `README.md` for results.

A fourth model (entity resolution / duplicate-pair classifier) was investigated and deliberately
dropped — see [`docs/entity_resolution_investigation.md`](../../../docs/entity_resolution_investigation.md).
