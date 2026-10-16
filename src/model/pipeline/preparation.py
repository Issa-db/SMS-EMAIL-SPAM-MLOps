"""Clean message text and map spam/ham labels for model training."""

import re

import pandas as pd
from loguru import logger

MESSAGE_COLUMN = "Message"
TARGET_COLUMN = "target"


def clean_text(text: object) -> str:
    """Normalize text to lowercase alphabetic tokens."""
    normalized = str(text).lower()
    normalized = re.sub(r"http\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"\d+", " ", normalized)
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    return " ".join(normalized.split())


def normalize_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Map known labels and remove rows without valid label or message."""
    required = {"Category", MESSAGE_COLUMN}
    missing = required.difference(dataset.columns)
    if missing:
        logger.error("Training data is missing required columns: {}", sorted(missing))
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")

    normalized = dataset.copy()
    normalized["Category"] = (
        normalized["Category"].astype(str).str.strip().str.lower()
    )
    targets = normalized["Category"].map({"ham": 0, "spam": 1})
    messages = normalized[MESSAGE_COLUMN].fillna("").astype(str).str.strip()
    valid = targets.notna() & messages.ne("")
    dropped_rows = int((~valid).sum())
    logger.debug("Preparing {} training rows", len(dataset))
    if dropped_rows:
        logger.warning(
            "Dropped {} training rows with unknown labels or empty messages",
            dropped_rows,
        )
    normalized = normalized.loc[valid].reset_index(drop=True)
    normalized[MESSAGE_COLUMN] = (
        messages.loc[valid].map(clean_text).reset_index(drop=True)
    )
    normalized[TARGET_COLUMN] = (
        targets.loc[valid].astype("int8").reset_index(drop=True)
    )
    return normalized


def prepare_data(dataset: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return normalized text and labels, failing when no rows are usable."""
    normalized = normalize_data(dataset)
    if normalized.empty:
        logger.error("Training data contains no valid labeled messages")
        raise ValueError("No valid labeled messages are available for training")
    logger.info("Prepared {} valid labeled messages", len(normalized))
    return normalized[MESSAGE_COLUMN], normalized[TARGET_COLUMN]
