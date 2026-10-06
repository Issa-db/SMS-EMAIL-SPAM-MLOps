"""
Database Configuration Module.

Loads and validates database settings from the .env file via
pydantic_settings, and initializes the SQLAlchemy database engine
for the sqlite connection. Exposes module-level `db_settings` and
`engine` instances for use across the app.
"""

from pathlib import Path

from loguru import logger
from pydantic import FilePath
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine

CONFIG_DIR = Path(__file__).resolve().parent


class DbSettings(BaseSettings):
    """
    Database settings loaded from environment variables and .env file.

    Defines and validates the paths and connection parameters
    required by the data pipeline. Values are read once at import
    time and shared across the app via the module-level
    `db_settings` instance.

    Args:
        BaseSettings (pydantic_settings.BaseSettings): Base class
        providing automatic environment variable loading and
        validation.
    """
    model_config = SettingsConfigDict(
        env_file=f"{CONFIG_DIR}/.env",
        env_file_encoding="utf-8",
        extra="ignore")

    data_file_name: FilePath | None = None
    db_file_name: FilePath
    db_url: str
    table_name: str


try:
    db_settings = DbSettings()
    logger.info("settings loaded successfully from .env")
except Exception as e:
    logger.critical(f'CRITICAL: failed to load db settings from .env: {e}')
    raise

try:
    engine = create_engine(db_settings.db_url)
    logger.info(f'Database engine created for {db_settings.db_url}')
except Exception as e:
    logger.critical(f'Failed to create database engine: {e}')
    raise
