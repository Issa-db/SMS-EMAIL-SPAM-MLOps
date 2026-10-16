"""Read training rows from a configured SQLAlchemy database table."""

import pandas as pd
from loguru import logger
from sqlalchemy import Engine, MetaData, Table, select


def load_data(engine: Engine, table_name: str = "ham") -> pd.DataFrame:
    """Load every row after validating the required training columns."""
    logger.debug("Inspecting training table '{}'", table_name)
    table = Table(table_name, MetaData(), autoload_with=engine)
    required = {"Category", "Message"}
    missing = required.difference(table.c.keys())
    if missing:
        logger.error(
            "Training table '{}' is missing required columns: {}",
            table_name,
            sorted(missing),
        )
        raise ValueError(f"Training table is missing columns: {sorted(missing)}")
    query = select(table.c.Category, table.c.Message)
    with engine.connect() as connection:
        dataset = pd.read_sql(query, connection)
    if dataset.empty:
        logger.warning("Training table '{}' contains no rows", table_name)
    else:
        logger.info(
            "Loaded {} training rows from table '{}'",
            len(dataset),
            table_name,
        )
    return dataset
