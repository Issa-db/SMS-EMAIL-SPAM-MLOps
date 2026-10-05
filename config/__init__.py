"""
Config Package.

Aggregates logger, database, and model configuration submodules.
Import order matters: logger must be configured first so that any
settings-loading messages from db/model modules use the configured
sinks rather than loguru's default handler.
"""

from .db import DbSettings, db_settings, engine
from .logger import configure_logger, logger_settings
from .model import ModelSettings, model_settings

__all__ = [
    "DbSettings",
    "ModelSettings",
    "configure_logger",
    "db_settings",
    "engine",
    "logger_settings",
    "model_settings",
]
