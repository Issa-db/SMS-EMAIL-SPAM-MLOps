"""Tests for CSV and SQLite data collection."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.db_model import Base, SpamMessage
from model.pipeline import collection


def test_csv_loader_reads_expected_columns(tmp_path: Path) -> None:
    """Load the two supported fields and ignore additional CSV columns."""
    csv_path = tmp_path / "emails.csv"
    csv_path.write_text(
        "Category,Message,Extra\nham,hello,1\n",
        encoding="latin-1",
    )

    dataframe = collection.load_data(csv_path)

    assert dataframe.to_dict("records") == [
        {"Category": "ham", "Message": "hello"},
    ]


def test_db_loader_returns_saved_messages(monkeypatch) -> None:
    """Read ORM message records from the configured SQLAlchemy engine."""
    db_engine = create_engine("sqlite://")
    Base.metadata.create_all(db_engine)
    with Session(db_engine) as session:
        session.add_all(
            [
                SpamMessage(Category="ham", Message="hello"),
                SpamMessage(Category="spam", Message="claim prize"),
            ],
        )
        session.commit()
    monkeypatch.setattr(collection, "engine", db_engine)

    try:
        dataframe = collection.load_data_from_db()
    finally:
        db_engine.dispose()

    assert sorted(dataframe["Category"].tolist()) == ["ham", "spam"]
    assert set(dataframe.columns) == {"Category", "Message"}
