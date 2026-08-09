"""Shared MLflow tracking configuration for all three models' training runs.

Defaults to local-only, no external service: a SQLite-backed tracking store
(the Model Registry needs a database backend -- the plain file store doesn't
support registry operations) with a local `mlruns/` artifact directory, both
gitignored, same as the imputation selector's per-run artifacts.

Both are overridable via environment variables so the Docker Compose stack
(Phase 4, Task 4) can point them at container-network locations instead --
a separate SQLite file under a named volume (so the container's tracking
store never collides with a host `mlflow.db` from local runs) and a MinIO
bucket as the artifact store (`s3://...`, read by MLflow's boto3-based S3
artifact repository via MLFLOW_S3_ENDPOINT_URL). Local, non-Docker runs are
unaffected since neither variable is set outside the compose stack.
"""

import os
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
ARTIFACT_ROOT = os.environ.get("MLFLOW_ARTIFACT_ROOT", (REPO_ROOT / "mlruns").as_uri())


def configure_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_name)
