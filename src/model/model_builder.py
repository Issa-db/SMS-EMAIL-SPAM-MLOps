"""Offline training service, separate from online inference."""

from sqlalchemy import create_engine
from loguru import logger

from src.config.settings import Settings
from src.db.collection import load_data
from src.model.pipeline.model import train_and_export


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
        engine = create_engine(self.settings.db_url)
        try:
            dataset = load_data(engine, self.settings.table_name)
            logger.debug("Loaded {} rows for model training", len(dataset))
            accuracy = train_and_export(
                dataset,
                self.settings.model_path,
                self.settings.vectorizer_path,
            )
            logger.info(
                "Training run completed with held-out accuracy={:.4f}",
                accuracy,
            )
            return accuracy
        except Exception:
            logger.opt(exception=True).critical(
                "Model training failed; no successful artifact bundle was produced",
            )
            raise
        finally:
            engine.dispose()
