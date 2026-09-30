"""
Logger Configuration Module.

Loads and validates logger settings from the .env file via
pydantic_settings, and configures the global loguru logger.
"""

import sys
from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_DIR = Path(__file__).resolve().parent


class LoggerSettings(BaseSettings):
    """
    Logger settings loaded from environment variables and .env file.

    Defines and validates the log level required by logging
    configuration. Values are read once at import time and shared
    across the app via the module-level `logger_settings` instance.

    Args:
        BaseSettings (pydantic_settings.BaseSettings): Base class
        providing automatic environment variable loading and
        validation.
    """
    model_config = SettingsConfigDict(
        env_file=f"{CONFIG_DIR}/.env",
        env_file_encoding="utf-8",
        extra="ignore",)

    log_level: str


def configure_logger(log_level: str) -> None:
    """
    Configure the global loguru logger with the specified log level.

    Args:
        log_level (str): The desired log level
        (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
    """
    logger.remove()  # Remove default handler
    logger.add(sys.stderr,
               level=log_level,
               colorize=True)
    logger.add("logs/app_{time:YYYY-MM-DD}.log",
               level=log_level,
               mode="a")


try:
    logger_settings = LoggerSettings()
    Path("logs").mkdir(exist_ok=True)
    configure_logger(logger_settings.log_level)
    logger.info("settings loaded successfully from .env")
except Exception as e:
    print(f"CRITICAL: failed to load settings from .env: {e}")
    raise
