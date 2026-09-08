import sys
from pathlib import Path
from sqlalchemy import create_engine
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import DirectoryPath, FilePath
from loguru import logger


CONFIG_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
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
    engine = create_engine(settings.db_url)
    logger.info(f"Database engine created for {settings.db_url}")
except Exception as e:
    logger.critical(f"Failed to create database engine: {e}")
    raise
        