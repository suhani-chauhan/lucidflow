# models/

Not implemented yet. Each subfolder is a Phase 2/3 ML model:

- `column_type_classifier/` — Phase 2: semantic column-type classifier (feeds the auto-generated data contract).
- `imputation_selector/` — Phase 2: learned imputation-method selector (median/KNN/MICE/LightGBM benchmarking).
- `duplicate_classifier/` — Phase 3: LLM-distilled duplicate-pair classifier for entity resolution.
- `quarantine_classifier/` — Phase 3: gradient-boosted quarantine classifier, retrained from dashboard review.
