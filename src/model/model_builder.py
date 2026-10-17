"""Offline training service, separate from online inference."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from loguru import logger

from src.config.settings import APP_ROOT, Settings
from src.db.collection import load_data
from src.model.pipeline.model import TrainingResult, train_and_export
from src.model.tracking import start_training_run


class ModelBuilderService:
    """Load labeled rows and publish a new model artifact pair."""

    def __init__(self, settings: Settings) -> None:
        """Keep builder configuration explicit and injectable."""
        self.settings = settings

    def train_model(self) -> float:
        """Train from the configured table and return held-out accuracy."""
        logger.info(
            "Starting model training from table '{}'",
            self.settings.table_name,
        )
        engine = None
        try:
            engine = create_engine(self.settings.db_url)
            dataset = load_data(engine, self.settings.table_name)
            logger.debug("Loaded {} rows for model training", len(dataset))
            code_revision = self._get_code_revision()
            with start_training_run(self.settings) as tracking_run:
                result = train_and_export(
                    dataset,
                    self.settings.model_path,
                    self.settings.vectorizer_path,
                    random_state=self.settings.random_state,
                    test_size=self.settings.test_size,
                    cross_validation_folds=(
                        self.settings.cross_validation_folds
                    ),
                    minimum_accuracy=self.settings.minimum_accuracy,
                )
                bundle_directory, manifest = self._write_manifest(
                    tracking_run.run_id,
                    code_revision,
                    result,
                )
                tracking_run.log_params(
                    {
                        "dataset.table": self.settings.table_name,
                        "dataset.version": self.settings.dataset_version,
                        "dataset.rows": result.row_count,
                        "dataset.class_counts": result.class_counts,
                        "build.code_revision": code_revision,
                        "training.random_state": self.settings.random_state,
                        "training.test_size": self.settings.test_size,
                        "training.cv_folds": (
                            self.settings.cross_validation_folds
                        ),
                        "training.minimum_accuracy": (
                            self.settings.minimum_accuracy
                        ),
                        "model.best_parameters": result.best_parameters,
                        "model.vectorizer_parameters": (
                            result.vectorizer_parameters
                        ),
                        "model.classifier_search_space": (
                            result.classifier_search_space
                        ),
                    },
                )
                tracking_run.log_metrics(
                    {
                        "accuracy": result.accuracy,
                        "precision": result.precision,
                        "recall": result.recall,
                        "f1_score": result.f1_score,
                    },
                )
                for artifact in (
                    bundle_directory / self.settings.model_path.name,
                    bundle_directory / self.settings.vectorizer_path.name,
                    manifest,
                ):
                    tracking_run.log_artifact(artifact)
            accuracy = result.accuracy
            logger.info(
                "Training run completed with held-out accuracy={:.4f}",
                accuracy,
            )
            return accuracy
        except Exception:
            logger.opt(exception=True).critical(
                "Model-building run failed during training or artifact tracking",
            )
            raise
        finally:
            if engine is not None:
                engine.dispose()

    def _write_manifest(
        self,
        run_id: str,
        code_revision: str | None,
        result: TrainingResult,
    ) -> tuple[Path, Path]:
        """Publish an immutable local bundle and update its latest manifest."""
        if Path(run_id).name != run_id:
            raise ValueError("Training run ID cannot contain path components")
        latest_manifest_path = self.settings.manifest_path
        latest_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings.artifact_store_path.mkdir(parents=True, exist_ok=True)
        bundle_directory = self.settings.artifact_store_path / run_id
        try:
            relative_bundle_directory = bundle_directory.resolve().relative_to(
                latest_manifest_path.parent.resolve(),
            )
        except ValueError as error:
            raise ValueError(
                "Manifest directory must contain the artifact bundle directory",
            ) from error
        if bundle_directory.exists():
            raise FileExistsError(f"Model bundle already exists: {bundle_directory}")
        staging_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{run_id}.",
                suffix=".tmp",
                dir=self.settings.artifact_store_path,
            ),
        )
        try:
            model_bundle_path = staging_directory / self.settings.model_path.name
            vectorizer_bundle_path = (
                staging_directory / self.settings.vectorizer_path.name
            )
            shutil.copy2(self.settings.model_path, model_bundle_path)
            shutil.copy2(self.settings.vectorizer_path, vectorizer_bundle_path)
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "code_revision": code_revision,
                "bundle_directory": str(relative_bundle_directory),
                "dataset": {
                    "table": self.settings.table_name,
                    "version": self.settings.dataset_version,
                    "row_count": result.row_count,
                    "class_counts": result.class_counts,
                },
                "training": {
                    "random_state": self.settings.random_state,
                    "test_size": self.settings.test_size,
                    "cross_validation_folds": (
                        self.settings.cross_validation_folds
                    ),
                    "minimum_accuracy": self.settings.minimum_accuracy,
                    "train_row_count": result.train_row_count,
                    "test_row_count": result.test_row_count,
                    "feature_count": result.feature_count,
                    "best_parameters": result.best_parameters,
                    "vectorizer_parameters": result.vectorizer_parameters,
                    "classifier_search_space": (
                        result.classifier_search_space
                    ),
                },
                "metrics": {
                    "accuracy": result.accuracy,
                    "precision": result.precision,
                    "recall": result.recall,
                    "f1_score": result.f1_score,
                },
                "artifacts": {
                    "model": {
                        "file": model_bundle_path.name,
                        "sha256": self._sha256(model_bundle_path),
                    },
                    "vectorizer": {
                        "file": vectorizer_bundle_path.name,
                        "sha256": self._sha256(vectorizer_bundle_path),
                    },
                },
            }
            self._write_json(staging_directory / "manifest.json", manifest)
            staging_directory.replace(bundle_directory)
        except Exception:
            shutil.rmtree(staging_directory, ignore_errors=True)
            raise
        self._write_json(latest_manifest_path, manifest)
        logger.info("Published local model bundle {}", bundle_directory)
        return bundle_directory, bundle_directory / "manifest.json"

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        """Atomically serialize one JSON metadata file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @staticmethod
    def _sha256(path: Path) -> str:
        """Return a streaming SHA-256 digest for one model artifact."""
        digest = hashlib.sha256()
        with path.open("rb") as artifact_file:
            for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _get_code_revision() -> str | None:
        """Read the CI-provided or current Git revision when available."""
        configured_revision = os.getenv("CODE_REVISION") or os.getenv(
            "GITHUB_SHA",
        )
        if configured_revision:
            return configured_revision
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=APP_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=APP_ROOT,
                capture_output=True,
                check=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            logger.debug("Git revision is unavailable for this training run")
            return None
        revision = result.stdout.strip()
        if revision and status.stdout.strip():
            revision = f"{revision}-dirty"
        return revision or None
