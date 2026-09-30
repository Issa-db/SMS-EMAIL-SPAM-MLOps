"""
Configuration Module.

Loads and validates aaplication settings from the .env file via
pydantic_settings, configures the global loguru logger, and initializes
the SQLAlchemy database engine for the sqlite connection.
Exposes a module-level`settings` and `engine` instances for use across the app.
"""

import sys
from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath, FilePath
from sqlalchemy import create_engine


CONFIG_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment varibels and .env file.

    Defines and validates all paths, filenames, and runtime parameters
    required by the data pipeline, model training, and inference services.
    Values are read once at import time and shared across the app via
    the module-level `settings` instance.

    Args:
        BaseSettings (pydantic_settings.BaseSettings): Base class providing
        automatic environment variable loading and validation.
    """
    model_config = SettingsConfigDict(
        env_file=f"{CONFIG_DIR}/.env",
        env_file_encoding="utf-8")

    data_file_name: FilePath
    model_path: DirectoryPath
    vectorizer_path: DirectoryPath
    db_file_name: FilePath
    db_url: str
    table_name: str
    model_name: str
    vectorizer_name: str
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
    settings = Settings()
    Path("logs").mkdir(exist_ok=True)
    configure_logger(settings.log_level)
    logger.info("settings loaded successfully from .env")
except Exception as e:
    print(f"CRITICAL: failed to load settings from .env: {e}")
    raise

try:
    engine = create_engine(settings.db_url)
    logger.info(f"Database engine created for {settings.db_url}")
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}")
    raise
