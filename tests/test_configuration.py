"""Tests for validated settings and logger setup."""

from pathlib import Path
from unittest.mock import Mock

from src.config import logger as app_logger
from src.config.settings import Settings


def test_settings_resolve_relative_paths_from_app_root() -> None:
    """Resolve configured database and artifacts independent of CWD."""
    settings = Settings(
        db_url="sqlite:///src/db/emails.sqlite",
        model_path=Path("src/model/models/NB_db_v1.onnx"),
        vectorizer_path=Path("src/model/models/vectorizer.joblib"),
    )

    assert settings.db_url.startswith("sqlite:////")
    assert settings.model_path.is_absolute()
    assert settings.vectorizer_path.is_absolute()


def test_settings_validate_environment_values(monkeypatch, tmp_path: Path) -> None:
    """Allow deployment environment variables to override local settings."""
    model_path = tmp_path / "custom-model.onnx"
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("TABLE_NAME", "training_rows")
    monkeypatch.setenv("PORT", "8080")
    settings = Settings()

    assert settings.model_path == model_path
    assert settings.table_name == "training_rows"
    assert settings.port == 8080


def test_configure_logger_creates_sinks(monkeypatch, tmp_path: Path) -> None:
    """Create the log directory and install console plus file handlers."""
    remove = Mock()
    add = Mock()
    monkeypatch.setattr(app_logger.logger, "remove", remove)
    monkeypatch.setattr(app_logger.logger, "add", add)

    app_logger.configure_logger("DEBUG", tmp_path / "logs")

    assert (tmp_path / "logs").is_dir()
    remove.assert_called_once_with()
    assert add.call_count == 2
