"""Shared MLflow tracking configuration for all three models' training runs.

Local-only, no external service. A SQLite-backed tracking store is used
instead of MLflow's plain file store because the Model Registry (used by all
three training scripts) requires a database-backed store -- the file store
doesn't support registry operations. `mlflow.db` and the artifact directory
are both gitignored, same as the imputation selector's per-run artifacts.
"""

from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (REPO_ROOT / "mlruns").as_uri()


def configure_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_name)
