"""Configure Loguru sinks at application startup."""

import sys
from pathlib import Path

from loguru import logger


def configure_logger(log_level: str, log_directory: Path) -> None:
    """Set console and daily file sinks for one application process."""
    log_directory.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=log_level, colorize=True)
    logger.add(log_directory / "app_{time:YYYY-MM-DD}.log", level=log_level)
    logger.info("Logging configured at {} level", log_level.upper())
