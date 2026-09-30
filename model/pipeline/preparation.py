"""
Preparation Module.

Provides functions to clean and normalize the raw email dataset,
returning a pandas dataframe with "Category" and "Message" columns
for downstream cleaning and model training.

"""
import re

import pandas as pd
from loguru import logger

from model.pipeline.collection import load_data_from_db


@logger.catch(message="Failed to clean a message")
def _clean_text(text: str) -> str:
    """
    Clean and normalize raw SMS/email text.

    Args:
        text (str): The raw text message to be cleaned.
    """

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split())


# test _clean_text function
# df = load_data()
# df["Message"] = df["Message"].apply(_clean_text)
# print("Cleaned Data head: ", df.head())


def _normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize labels, validate categories, and prepare message text.

    Args:
        data (pd.DataFrame): The raw dataset
        containing "Category" and "Message" columns.
    """
    data = data.copy()
    before = len(data)

    data["Category"] = data["Category"].astype(str).str.strip().str.lower()
    data["target"] = data["Category"].map({"ham": 0, "spam": 1}).astype("int8")

    unmapped = data["target"].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows had unrecognized category values")
    data["Message"] = data["Message"].fillna("").astype(str).str.strip()
    data = data[data["Message"].str.len() > 0].reset_index(drop=True)

    dropped = before - len(data)
    if dropped > 0:
        logger.warning(f"dropped {dropped} rows with empty messages")

    data["Message"] = data["Message"].apply(_clean_text)
    return data


def prepare_data() -> tuple[pd.Series, pd.Series]:
    """
    Prepare the dataset for training and testing.

    Returns:
        tuple[pd.Series, pd.Series]: cleaned text and binary labels.
    """
    logger.info("Starting data Preparation pipeline")
    data = load_data_from_db()
    logger.debug(f"Loaded {len(data)} raw rows")
    # clean the data before normalizing it
    data["Message"] = data["Message"].apply(_clean_text)
    data = _normalize_data(data)

    logger.info(f"Data preparation complete: {len(data)} rows ready")
    return data["Message"], data["target"]


if __name__ == "__main__":
    X, y = prepare_data()
    print(X.head())
    print(y.head())
