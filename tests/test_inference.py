"""Tests for artifact loading and prediction behavior."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.model import model_inference


class FakeVectorizer:
    """Return deterministic feature rows and retain normalized inputs."""

    def transform(self, messages: list[str]) -> SimpleNamespace:
        """Build a two-column dense matrix for each message."""
        self.messages = messages
        return SimpleNamespace(toarray=lambda: np.ones((len(messages), 2)))


class FakeSession:
    """Provide the ONNX Runtime methods consumed by the inference service."""

    def get_inputs(self) -> list[SimpleNamespace]:
        """Return the expected ONNX input metadata."""
        return [SimpleNamespace(name="features")]

    def get_outputs(self) -> list[SimpleNamespace]:
        """Return the expected ONNX output metadata."""
        return [SimpleNamespace(name="labels")]

    def run(self, output_names, inputs) -> list[np.ndarray]:
        """Return deterministic class IDs and capture model inputs."""
        self.inputs = inputs
        return [np.array([[0], [1]])]


def test_load_model_fails_if_artifacts_are_missing(tmp_path: Path) -> None:
    """Require both files before the API can become ready."""
    service = model_inference.ModelInferenceService(
        tmp_path / "missing.onnx",
        tmp_path / "missing.joblib",
    )

    with pytest.raises(FileNotFoundError, match="bundle files not found"):
        service.load_model()


def test_load_model_reads_vectorizer_and_onnx_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Load both runtime dependencies from their configured artifact paths."""
    model_path = tmp_path / "model.onnx"
    vectorizer_path = tmp_path / "vectorizer.joblib"
    model_path.touch()
    vectorizer_path.touch()
    vectorizer = FakeVectorizer()
    session = FakeSession()
    monkeypatch.setattr(model_inference.joblib, "load", Mock(return_value=vectorizer))
    monkeypatch.setattr(
        model_inference.ort,
        "InferenceSession",
        Mock(return_value=session),
    )
    service = model_inference.ModelInferenceService(model_path, vectorizer_path)

    service.load_model()

    assert service.vectorizer is vectorizer
    assert service.session is session


def test_predict_cleans_text_and_returns_class_ids() -> None:
    """Apply the training-time text normalization before inference."""
    service = model_inference.ModelInferenceService(
        Path("unused.onnx"),
        Path("unused.joblib"),
    )
    vectorizer = FakeVectorizer()
    session = FakeSession()
    service.vectorizer = vectorizer
    service.session = session

    assert service.predict(["Hello 123!", "WIN NOW"]) == [0, 1]
    assert vectorizer.messages == ["hello", "win now"]
    assert session.inputs["features"].dtype == np.float32


def test_predict_rejects_unloaded_service_and_blank_input() -> None:
    """Reject invalid lifecycle and messages before invoking ONNX Runtime."""
    service = model_inference.ModelInferenceService(
        Path("unused.onnx"),
        Path("unused.joblib"),
    )
    with pytest.raises(RuntimeError, match="load_model"):
        service.predict("hello")

    service.vectorizer = FakeVectorizer()
    service.session = FakeSession()
    with pytest.raises(ValueError, match="non-empty"):
        service.predict(["  "])
