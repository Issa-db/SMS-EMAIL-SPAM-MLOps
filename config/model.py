"""
Model Configuration Module.

Loads and validates ML pipeline settings from the .env file via
pydantic_settings. Exposes a module-level `model_settings` instance
for use across the app.
"""

from pathlib import Path

from loguru import logger
from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


CONFIG_DIR = Path(__file__).resolve().parent


class ModelSettings(BaseSettings):
    """
    Model settings loaded from environment variables and .env file.

    Defines and validates all paths and filenames required by model
    training and inference. Values are read once at import time and
    shared across the app via the module-level `model_settings`
    instance.

    Args:
        BaseSettings (pydantic_settings.BaseSettings): Base class
        providing automatic environment variable loading and
        validation.
    """
    model_config = SettingsConfigDict(
        env_file=f"{CONFIG_DIR}/.env",
        env_file_encoding="utf-8",
        extra="ignore")

    model_path: DirectoryPath
    vectorizer_path: DirectoryPath
    model_name: str
    vectorizer_name: str


try:
    model_settings = ModelSettings()
    logger.info("settings loaded successfully from .env")
except Exception as e:
    logger.critical(f"CRITICAL: failed to load settings from .env: {e}")
    raise
