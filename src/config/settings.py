"""Configuration loaded from the environment or src/config/.env."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Runtime settings for data access, artifacts, logging, and API server."""

    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_url: str = f"sqlite:///{APP_ROOT / 'src/db/emails.sqlite'}"
    table_name: str = "ham"
    model_path: Path = APP_ROOT / "src/model/models/NB_db_v1.onnx"
    vectorizer_path: Path = APP_ROOT / "src/model/models/vectorizer.joblib"
    log_level: str = Field(default="INFO", min_length=1)
    host: str = "127.0.0.1"
    port: int = Field(default=5000, ge=1, le=65535)

    @field_validator("model_path", "vectorizer_path", mode="before")
    @classmethod
    def resolve_artifact_paths(cls, value: str | Path) -> Path:
        """Resolve relative artifact paths against the application root."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (APP_ROOT / path).resolve()

    @field_validator("db_url")
    @classmethod
    def resolve_relative_sqlite_url(cls, value: str) -> str:
        """Resolve a relative SQLite filename against the application root."""
        prefix = "sqlite:///"
        if value.startswith(prefix) and not value.startswith("sqlite:////"):
            database_path = value.removeprefix(prefix)
            if database_path != ":memory:":
                path = Path(database_path).expanduser()
                if not path.is_absolute():
                    return f"{prefix}{(APP_ROOT / path).resolve()}"
        return value
