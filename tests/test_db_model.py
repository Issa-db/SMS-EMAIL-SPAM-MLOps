"""Tests for the shared SQLAlchemy training table definition."""

from sqlalchemy import create_engine, inspect

from src.db.db_model import SpamMessage, metadata


def test_training_table_matches_existing_sqlite_schema() -> None:
    """Create the table without incorrectly making category a unique key."""
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    inspector = inspect(engine)

    assert SpamMessage.name == "ham"
    assert {column["name"] for column in inspector.get_columns("ham")} == {
        "Category",
        "Message",
    }
    assert inspector.get_pk_constraint("ham")["constrained_columns"] == []
    engine.dispose()
