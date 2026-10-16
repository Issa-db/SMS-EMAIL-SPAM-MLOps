"""Command-line entry point for offline model training."""

from loguru import logger

from src.config.logger import configure_logger
from src.config.settings import APP_ROOT, Settings
from src.model.model_builder import ModelBuilderService


def main() -> None:
    """Train and export the configured classifier and vectorizer pair."""
    settings = Settings()
    configure_logger(settings.log_level, APP_ROOT / "src/logs")
    accuracy = ModelBuilderService(settings).train_model()
    logger.info("Training complete; held-out accuracy={:.4f}", accuracy)


if __name__ == "__main__":
    main()
