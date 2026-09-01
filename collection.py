import pandas as pd 
#from pathlib import Path
from config import settings
from config import engine
from db_model import spam
from sqlalchemy import select 
from loguru import logger

def load_data(path= settings.data_file_name):
    """
    Load the dataset from the CSV file
    Returns: 
        DataFrame: The Loaded dataset as a pandas DataFrame 
    """
    logger.info(f"loading csv file at path {path}")
    return pd.read_csv(path, encoding='latin-1', header=0, usecols=[0, 1], names=["Category", "Message"])


def load_data_from_db():
    """
    Load the dataset from the sqlite db connection 
    Returns: 
            DataFrame: The Loaded dataset as a pandas DataFrame 
    """
    logger.info("Extracting the data from database")
    query= select(spam)
    return pd.read_sql(query, engine)
    

# # test data collection script
# print ("Data Loaded Successfully")
# print ("Data head: ", load_data())