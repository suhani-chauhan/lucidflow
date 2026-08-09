# models/

Each subfolder is one of LucidFlow's trained models:

- `column_type_classifier/` — Phase 2, done: semantic column-type classifier, trained as the
  foundation for planned automatic contract generation. Not wired up yet — Phase 1's hand-written
  `Company` contract (`src/lucidflow/validation/pydantic_models.py`) is still what runs at
  inference time.
- `imputation_selector/` — Phase 2, done: learned imputation-method selector (median/KNN/MICE/LightGBM benchmarking).
- `quarantine_classifier/` — Phase 3, done: gradient-boosted quarantine classifier trained on synthetic corruption
  (real positives are ~0 in `companies.csv`). Wired into the pipeline's write path as of Phase 4, Task 3 — see
  `src/lucidflow/flows/pipeline_flow.py`. `retrain_with_reviews.py` (Phase 5, Task 3) retrains it by combining
  the synthetic dataset with human-reviewed real quarantine.records decisions from the dashboard's review
  queue — see `dashboard/README.md` for the full loop and a real end-to-end result.

A fourth model (entity resolution / duplicate-pair classifier) was investigated and deliberately
dropped — see [`docs/entity_resolution_investigation.md`](../../../docs/entity_resolution_investigation.md).
