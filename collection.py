import pandas as pd 
from pathlib import Path


def load_data(path = Path(__file__).parent.parent / "data" / "spam.csv"):
    """
    Load the dataset from the CSV file
    Returns: 
        DataFrame: The Loaded dataset as a pandas DataFrame 
    """
    return pd.read_csv(path, encoding='latin-1', header=0, usecols=[0, 1], names=["Category", "Message"])


# test data collection script
# print ("Data Loaded Successfully")
# print ("Data head: ", load_data())