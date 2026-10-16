"""Command-line entry point for the Flask inference API."""

from loguru import logger

from src.api import create_app
from src.config.logger import configure_logger
from src.config.settings import APP_ROOT, Settings


def main() -> None:
    """Start the API using configured host and port values."""
    settings = Settings()
    configure_logger(settings.log_level, APP_ROOT / "src/logs")
    logger.info("Starting inference API on {}:{}", settings.host, settings.port)
    create_app(settings).run(host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
