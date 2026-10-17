"""Tests for builder orchestration and artifact export."""

import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.config.settings import Settings
from src.model import model_builder
from src.model.pipeline import model as training_pipeline
from src.model.pipeline.model import TrainingResult


def test_builder_loads_configured_data_and_disposes_engine(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Send the selected table and artifact paths into the training pipeline."""
    engine = create_engine("sqlite://")
    dispose = Mock()
    monkeypatch.setattr(engine, "dispose", dispose)
    dataset = pd.DataFrame({"Category": ["ham"], "Message": ["example"]})
    read_data = Mock(return_value=dataset)
    result = TrainingResult(
        accuracy=0.95,
        precision=0.94,
        recall=0.93,
        f1_score=0.935,
        row_count=60,
        train_row_count=48,
        test_row_count=12,
        feature_count=18,
        class_counts={"0": 30, "1": 30},
        best_parameters={"alpha": 0.5, "fit_prior": True},
        vectorizer_parameters={"min_df": 5},
        classifier_search_space={"alpha": [0.1, 0.5, 1.0]},
    )
    model_path = tmp_path / "models/model.onnx"
    vectorizer_path = tmp_path / "models/vectorizer.joblib"
    manifest_path = tmp_path / "latest_bundle.json"

    def train(*args, **kwargs):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"model")
        vectorizer_path.write_bytes(b"vectorizer")
        return result

    monkeypatch.setattr(model_builder, "create_engine", Mock(return_value=engine))
    monkeypatch.setattr(model_builder, "load_data", read_data)
    monkeypatch.setattr(model_builder, "train_and_export", Mock(side_effect=train))
    settings = Settings(
        table_name="examples",
        db_url="sqlite://",
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        artifact_store_path=tmp_path / "bundle_store",
        manifest_path=manifest_path,
    )

    result = model_builder.ModelBuilderService(settings).train_model()

    assert result == 0.95
    read_data.assert_called_once_with(engine, "examples")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["metrics"]["f1_score"] == 0.935
    assert manifest["artifacts"]["model"]["sha256"]
    versioned_bundle = settings.artifact_store_path / manifest["run_id"]
    assert (versioned_bundle / "manifest.json").is_file()
    assert (versioned_bundle / "model.onnx").read_bytes() == b"model"
    dispose.assert_called_once_with()


def test_code_revision_uses_explicit_environment_value(monkeypatch) -> None:
    """Prefer a CI-provided revision over inspecting the local Git checkout."""
    monkeypatch.setenv("CODE_REVISION", "release-commit-sha")

    revision = model_builder.ModelBuilderService._get_code_revision()

    assert revision == "release-commit-sha"


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
    result = training_pipeline.train_and_export(
        pd.DataFrame(rows),
        model_path,
        vectorizer_path,
    )

    assert 0 <= result.accuracy <= 1
    assert result.train_row_count + result.test_row_count == result.row_count
    assert result.feature_count > 0
    assert set(result.class_counts) == {"0", "1"}
    assert model_path.read_bytes()
    assert vectorizer_path.stat().st_size > 0


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


def test_train_and_export_enforces_configured_accuracy_floor(
    tmp_path: Path,
) -> None:
    """Do not publish a model that misses its explicitly configured gate."""
    dataset = pd.DataFrame(
        {
            "Category": ["ham", "spam"] * 30,
            "Message": ["the same training text"] * 60,
        },
    )

    with pytest.raises(ValueError, match="accuracy threshold"):
        training_pipeline.train_and_export(
            dataset,
            tmp_path / "model.onnx",
            tmp_path / "vectorizer.joblib",
            minimum_accuracy=0.99,
        )

    assert not (tmp_path / "model.onnx").exists()
    assert not (tmp_path / "vectorizer.joblib").exists()
