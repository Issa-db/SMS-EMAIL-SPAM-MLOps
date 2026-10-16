"""Tests for SQLAlchemy training-data reads."""

import pytest
from sqlalchemy import create_engine, text

from src.db.collection import load_data


def test_load_data_preserves_repeated_labels() -> None:
    """Keep every row when categories are repeated across training examples."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE examples (Category TEXT, Message TEXT)"))
        connection.execute(
            text("INSERT INTO examples VALUES ('ham', 'one'), ('ham', 'two')"),
        )

    result = load_data(engine, "examples")

    assert result.to_dict("records") == [
        {"Category": "ham", "Message": "one"},
        {"Category": "ham", "Message": "two"},
    ]
    engine.dispose()


def test_load_data_rejects_missing_columns() -> None:
    """Fail before training if the configured table has the wrong schema."""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE invalid (body TEXT)"))

    with pytest.raises(ValueError, match="missing columns"):
        load_data(engine, "invalid")
    engine.dispose()
