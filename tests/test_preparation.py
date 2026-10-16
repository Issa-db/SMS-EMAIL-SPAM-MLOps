"""Tests for message cleanup and target-label preparation."""

import pandas as pd
import pytest

from src.model.pipeline.preparation import clean_text, normalize_data, prepare_data


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (" FREE!! www.example.com 123 ", "free"),
        (None, "none"),
        ("hello\n  world", "hello world"),
    ],
)
def test_clean_text_normalizes_messages(source: object, expected: str) -> None:
    """Remove links, digits, punctuation, and repeated whitespace."""
    assert clean_text(source) == expected


def test_normalize_data_maps_labels_and_drops_invalid_rows() -> None:
    """Keep non-empty messages with known labels and generate binary targets."""
    raw = pd.DataFrame(
        {
            "Category": [" HAM ", "Spam", "other", "ham"],
            "Message": ["Hi!", "BUY", "ignored", "  "],
        },
    )

    prepared = normalize_data(raw)

    assert prepared["target"].tolist() == [0, 1]
    assert prepared["Message"].tolist() == ["hi", "buy"]
    assert "target" not in raw


def test_prepare_data_requires_valid_training_rows() -> None:
    """Raise clear errors for wrong schemas and datasets without usable rows."""
    with pytest.raises(ValueError, match="missing columns"):
        prepare_data(pd.DataFrame({"Message": ["message"]}))
    with pytest.raises(ValueError, match="No valid"):
        prepare_data(
            pd.DataFrame({"Category": ["unknown"], "Message": ["message"]}),
        )
