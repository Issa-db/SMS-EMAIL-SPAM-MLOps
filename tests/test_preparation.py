"""Tests for message cleaning and training-data preparation."""

import pandas as pd
import pytest

from model.pipeline import preparation


@pytest.mark.parametrize(
    ("raw_text", "expected_text"),
    [
        (
            "  FREE!!! visit https://spam.test and win 123  ",
            "free visit and win",
        ),
        (None, "none"),
        ("Hello\n\tWorld", "hello world"),
        ("www.example.com", ""),
    ],
)
def test_clean_text_normalizes_message_content(
    raw_text: str | None,
    expected_text: str,
) -> None:
    """Lowercase text and remove URLs, digits, punctuation, and extra spaces."""
    cleaned_text = preparation._clean_text(raw_text)
    assert cleaned_text == expected_text


def test_normalize_data_maps_and_filters_rows() -> None:
    """Map known labels and discard rows with invalid labels or empty text."""
    raw_data = pd.DataFrame(
        {
            "Category": [" HAM ", "Spam", "other", "ham"],
            "Message": ["  Hello  ", "WIN 20", "ignored", "   "],
        },
    )

    normalized_data = preparation._normalize_data(raw_data)

    assert normalized_data["Category"].tolist() == ["ham", "spam"]
    assert normalized_data["target"].tolist() == [0, 1]
    assert normalized_data["Message"].tolist() == ["hello", "win"]
    assert "target" not in raw_data.columns


def test_prepare_data_cleans_db_messages(monkeypatch) -> None:
    """Return cleaned messages and binary labels from the database loader."""
    raw_data = pd.DataFrame(
        {"Category": ["ham", "spam"], "Message": ["Hello 22", "WIN! now"]},
    )
    monkeypatch.setattr(
        preparation,
        "load_data_from_db",
        lambda: raw_data.copy(),
    )

    messages, targets = preparation.prepare_data()

    assert messages.tolist() == ["hello", "win now"]
    assert targets.tolist() == [0, 1]
