"""LucidFlow — pipeline entry point.

Runs the LucidFlow pipeline (ingestion -> validation -> cleaning ->
imputation -> quarantine classification -> dual-route write, plus a
drift check that can trigger a retrain) as a Prefect flow.

    python run_pipeline.py [path/to/file.csv]

Defaults to data/intake/companies.csv if no path is given. The actual stage
implementations, task boundaries, retries, and the drift-triggered retrain
now live in src/lucidflow/flows/pipeline_flow.py (Phase 4, Task 3) -- this
file is kept as the top-level entry point named in the README.
"""

from lucidflow.flows.pipeline_flow import main

if __name__ == "__main__":
    main()
