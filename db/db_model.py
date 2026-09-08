from sqlalchemy import VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from config.config import settings

class Base(DeclarativeBase):
    """Base class for all ORM models in this project."""
    pass

class ham(Base):
    """
     A single labeled message record.

    Args:
        Category (str): Label for the message — expected values: "ham" or "spam".
        Message (str): The raw, unprocessed SMS/email text.
    """
    __tablename__ = settings.table_name
    
    Category: Mapped[str]= mapped_column(VARCHAR(), primary_key=True)
    Message: Mapped[str]= mapped_column(VARCHAR())
    
    