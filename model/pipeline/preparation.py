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

MESSAGE_COLUMN = "Message"


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


def _normalize_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize labels, validate categories, and prepare message text.

    Args:
        dataset (pd.DataFrame): The raw dataset containing "Category"
        and "Message" columns.
    """
    dataset = dataset.copy()
    before = len(dataset)
    messages = dataset[MESSAGE_COLUMN].copy()

    dataset["Category"] = (
        dataset["Category"].astype(str).str.strip().str.lower()
    )
    dataset["target"] = (
        dataset["Category"].map({"ham": 0, "spam": 1}).astype("int8")
    )

    unmapped = dataset["target"].isna().sum()
    if unmapped > 0:
        logger.warning(f"{unmapped} rows had unrecognized category values")

    messages = messages.fillna("").astype(str).str.strip()
    dataset = dataset[messages.str.len() > 0].reset_index(drop=True)
    messages = messages[messages.str.len() > 0].reset_index(drop=True)

    dropped = before - len(dataset)
    if dropped > 0:
        logger.warning(f"dropped {dropped} rows with empty messages")

    messages = messages.apply(_clean_text)
    dataset[MESSAGE_COLUMN] = messages

    return dataset


def prepare_data() -> tuple[pd.Series, pd.Series]:
    """
    Prepare the dataset for training and testing.

    Returns:
        tuple[pd.Series, pd.Series]: cleaned text and binary labels.
    """
    logger.info("Starting data Preparation pipeline")
    query_data = load_data_from_db()
    logger.debug(f"Loaded {len(query_data)} raw rows")
    # clean the data before normalizing it
    query_data[MESSAGE_COLUMN] = query_data[MESSAGE_COLUMN].apply(_clean_text)
    query_data = _normalize_data(query_data)

    logger.info(f'Data preparation complete: {len(query_data)} rows ready')
    return query_data[MESSAGE_COLUMN], query_data["target"]


if __name__ == "__main__":
    X_sample, y_sample = prepare_data()
    logger.info(f"Sample text:\n{X_sample.head()}")
    logger.info(f"Sample labels:\n{y_sample.head()}")
