import sys
from pathlib import Path
from sqlalchemy import create_engine
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath, FilePath
from loguru import logger


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    data_file_name: FilePath
    model_path: DirectoryPath
    vectorizer_path: DirectoryPath
    db_file_name: FilePath
    table_name: str
    model_name: str
    vectorizer_name: str
    log_level: str


try:
    settings = Settings()
    logger.info("settings loaded successfuly from .env")
except Exception as e:
    # logger not configured yet, so print as a last resort here
    print(f"CRITICAL: failed to load settings from .env: {e}")
    raise

Path("logs").mkdir(exist_ok=True)

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add("logs/app_{time:YYYY-MM-DD}.log", level=settings.log_level, rotation="10 MB", retention="7 days")

logger.info("Settings loaded successfully from .env")

try:
    engine = create_engine(f"sqlite:///{settings.db_file_name}")
    logger.info(f"Database engine created for {settings.db_file_name}")
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}")
    raise
        