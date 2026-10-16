"""SQLAlchemy table definition for labeled training messages."""

from sqlalchemy import Column, MetaData, String, Table

metadata = MetaData()
SpamMessage = Table(
    "ham",
    metadata,
    Column("Category", String(50), nullable=False),
    Column("Message", String(512), nullable=False),
)
