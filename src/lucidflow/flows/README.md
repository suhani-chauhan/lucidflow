# flows/

Phase 4, Task 3 done: Prefect 3 orchestration.

- `pipeline_flow.py` — ingestion, validation, cleaning, imputation, quarantine classification,
  and dual-route write, each a Prefect task with retries and structured logging
  (`get_run_logger()`). `run_pipeline.py` at the repo root now just calls this flow. Quarantine
  classification is real routing here, not orchestration-only: rows that pass the Pydantic
  contract are also scored by the trained quarantine classifier, and flagged rows get rerouted
  to `quarantine.records` (tagged as an ML flag, with the MLflow model version, not a contract
  violation).
- `retrain_flow.py` — retrains the imputation selector and quarantine classifier (the two models
  whose training data the drift-tested columns actually feed).
- The pipeline flow's optional drift check (`--check-drift`, off by default) compares a
  deliberately full-magnitude synthetic shift of the just-ingested batch against the Task 2
  reference profile — see `pipeline_flow.py`'s docstring for why it's always-flagging by design
  (there's no real incoming stream to check against) and therefore opt-in rather than run on
  every routine pipeline execution. When it flags, it triggers `retrain_flow`.
