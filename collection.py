import pandas as pd 
#from pathlib import Path
from config import settings

from loguru import logger
def load_data(path= settings.data_file_name):
    """
    Load the dataset from the CSV file
    Returns: 
        DataFrame: The Loaded dataset as a pandas DataFrame 
    """
    logger.info(f"loading csv file at path {path}")
    return pd.read_csv(path, encoding='latin-1', header=0, usecols=[0, 1], names=["Category", "Message"])

# # test data collection script
# print ("Data Loaded Successfully")
# print ("Data head: ", load_data())