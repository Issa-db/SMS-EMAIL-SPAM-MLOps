"""
Config Package.

Aggregates logger, database, and model configuration submodules.
Import order matters: logger must be configured first so that any
settings-loading messages from db/model modules use the configured
sinks rather than loguru's default handler.
"""

from .logger import logger_settings, configure_logger
from .db import DbSettings, db_settings, engine
from .model import ModelSettings, model_settings
