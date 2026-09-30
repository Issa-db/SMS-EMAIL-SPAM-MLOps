"""
Data Collection Module.

Providing functions to load the raw email dataset either
from a local csv file or from the sqlite database connection engine.
returning a pandas dataframe with "Category" and "Message" columns
cleaning and model training.
"""


import pandas as pd
from pathlib import Path
from loguru import logger
from sqlalchemy import select

from config import db_settings, engine
from db.db_model import ham


def load_data(path: str | Path = db_settings.data_file_name) -> pd.DataFrame:
    """
    Load the dataset from the CSV file
    Returns:
        DataFrame: The Loaded dataset as a pandas DataFrame
    """
    logger.info(f"loading csv file at path {path}")
    return pd.read_csv(path,
                       encoding='latin-1',
                       header=0,
                       usecols=[0, 1],
                       names=["Category", "Message"]
                       )


def load_data_from_db() -> pd.DataFrame:
    """
    Load the dataset from the sqlite db connection
    Returns:
            DataFrame: The Loaded dataset as a pandas DataFrame
    """
    logger.info("Extracting the data from database")
    query = select(ham)
    return pd.read_sql(query, engine)


# # test data collection script
# print ("Data Loaded Successfully")
# print ("Data head: ", load_data())
