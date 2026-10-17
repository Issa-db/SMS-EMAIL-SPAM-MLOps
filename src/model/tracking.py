"""Optional MLflow tracking for offline model-building runs."""

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from loguru import logger

from src.config.settings import APP_ROOT, Settings


class TrainingRun:
    """Small tracking interface used by the model builder."""

    def __init__(self, run_id: str, client: Any = None) -> None:
        """Create a run wrapper with an optional MLflow client module."""
        self.run_id = run_id
        self._client = client

    def log_params(self, parameters: dict[str, object]) -> None:
        """Record parameters when remote or local MLflow tracking is enabled."""
        if self._client is None:
            return
        normalized = {
            name: json.dumps(value, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else str(value)
            for name, value in parameters.items()
        }
        self._client.log_params(normalized)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Record numeric evaluation metrics in the active MLflow run."""
        if self._client is not None:
            self._client.log_metrics(metrics)

    def log_artifact(self, artifact: Path) -> None:
        """Copy a generated artifact into the active MLflow run."""
        if self._client is not None:
            self._client.log_artifact(str(artifact))


@contextmanager
def start_training_run(settings: Settings) -> Iterator[TrainingRun]:
    """Create a tracked run or a local run identifier when tracking is disabled."""
    if not settings.mlflow_tracking_enabled:
        yield TrainingRun(uuid4().hex)
        return

    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking is enabled, but MLflow is not installed. "
            "Install the tracking dependency group.",
        ) from error

    tracking_uri = settings.mlflow_tracking_uri
    if tracking_uri is None:
        tracking_database = (APP_ROOT / "mlflow.db").resolve()
        tracking_uri = f"sqlite:///{tracking_database}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run() as active_run:
        logger.info(
            "Started MLflow run {} in experiment '{}'",
            active_run.info.run_id,
            settings.mlflow_experiment_name,
        )
        yield TrainingRun(active_run.info.run_id, mlflow)
