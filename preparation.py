import re

import pandas as pd

from collection import load_data


def _clean_text(text):
    """Clean and normalize raw SMS/email text."""

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


def _normalize_data(data):
    """Normalize labels, validate categories, and prepare message text."""
    data = data.copy()

    data["Category"] = data["Category"].astype(str).str.strip().str.lower()
    data["target"] = data["Category"].map({"ham": 0, "spam": 1}).astype("int8")
    data["Message"] = data["Message"].fillna("").astype(str).str.strip()
    data = data[data["Message"].str.len() > 0].reset_index(drop=True)
    data["Message"] = data["Message"].apply(_clean_text)
    return data


def prepare_data():
    """
    Prepare the dataset for training and testing.

    Returns:
        tuple[pd.Series, pd.Series]: cleaned text and binary labels.
    """
    data = load_data()
    # clean the data before normalizing it
    data["Message"] =  data["Message"].apply(_clean_text)
    data = _normalize_data(data)
    return data["Message"], data["target"]


if __name__ == "__main__":
    X, y = prepare_data()
    print(X.head())
    print(y.head())