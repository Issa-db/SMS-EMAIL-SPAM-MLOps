"""Tests for builder orchestration and artifact export."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.config.settings import Settings
from src.model import model_builder
from src.model.pipeline import model as training_pipeline


def test_builder_loads_configured_data_and_disposes_engine(monkeypatch) -> None:
    """Send the selected table and artifact paths into the training pipeline."""
    engine = create_engine("sqlite://")
    dispose = Mock()
    monkeypatch.setattr(engine, "dispose", dispose)
    dataset = pd.DataFrame({"Category": ["ham"], "Message": ["example"]})
    read_data = Mock(return_value=dataset)
    train = Mock(return_value=0.95)
    monkeypatch.setattr(model_builder, "create_engine", Mock(return_value=engine))
    monkeypatch.setattr(model_builder, "load_data", read_data)
    monkeypatch.setattr(model_builder, "train_and_export", train)
    settings = Settings(table_name="examples", db_url="sqlite://")

    result = model_builder.ModelBuilderService(settings).train_model()

    assert result == 0.95
    read_data.assert_called_once_with(engine, "examples")
    train.assert_called_once_with(
        dataset,
        settings.model_path,
        settings.vectorizer_path,
    )
    dispose.assert_called_once_with()


def test_train_and_export_writes_both_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Train on both classes and save ONNX plus the matching vectorizer."""
    rows = [
        {
            "Category": "spam" if index % 2 else "ham",
            "Message": (
                f"{('spam' if index % 2 else 'ham')} commonword "
                f"offer{index % 4} token{index % 6}"
            ),
        }
        for index in range(60)
    ]
    model_path = tmp_path / "artifacts/model.onnx"
    vectorizer_path = tmp_path / "artifacts/vectorizer.joblib"
    converter = Mock(
        return_value=SimpleNamespace(
            SerializeToString=lambda: b"onnx model bytes",
        ),
    )
    monkeypatch.setattr(training_pipeline, "convert_classifier", converter)

    accuracy = training_pipeline.train_and_export(
        pd.DataFrame(rows),
        model_path,
        vectorizer_path,
    )

    assert 0 <= accuracy <= 1
    assert model_path.read_bytes() == b"onnx model bytes"
    assert vectorizer_path.stat().st_size > 0
    converter.assert_called_once()


def test_train_and_export_rejects_invalid_training_sets(tmp_path: Path) -> None:
    """Refuse small and single-class datasets before fitting a classifier."""
    model_path = tmp_path / "model.onnx"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    too_small = pd.DataFrame(
        {"Category": ["ham", "spam"], "Message": ["hello", "offer"]},
    )
    with pytest.raises(ValueError, match="at least 10 rows"):
        training_pipeline.train_and_export(too_small, model_path, vectorizer_path)

    one_class = pd.DataFrame(
        {
            "Category": ["ham"] * 12,
            "Message": [f"common text token{index}" for index in range(12)],
        },
    )
    with pytest.raises(ValueError, match="both classes"):
        training_pipeline.train_and_export(one_class, model_path, vectorizer_path)
