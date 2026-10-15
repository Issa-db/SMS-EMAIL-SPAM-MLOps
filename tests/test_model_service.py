"""Unit tests for model loading and prediction behavior."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from model import model_service

FEATURES_PER_MESSAGE = 2


class FakeVectorizer:
    """Return a dense feature row for every input message."""

    def transform(self, messages: list[str]) -> SimpleNamespace:
        """Transform messages into a two-column fake feature matrix."""
        return SimpleNamespace(
            toarray=lambda: np.ones((len(messages), FEATURES_PER_MESSAGE)),
        )


class FakeInferenceSession:
    """Provide the ONNX Runtime methods used by ModelService."""

    def get_inputs(self) -> list[SimpleNamespace]:
        """Expose the fake session input name."""
        return [SimpleNamespace(name="features")]

    def get_outputs(self) -> list[SimpleNamespace]:
        """Expose the fake session output name."""
        return [SimpleNamespace(name="labels")]

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Return fixed labels after checking the ONNX request shape."""
        assert output_names == ["labels"]
        assert inputs["features"].dtype == np.float32
        return [np.array([[0], [1]])]


def fake_session_factory(*args, **kwargs) -> FakeInferenceSession:
    """Create a fake ONNX session for patched inference tests."""
    return FakeInferenceSession()


def test_predict_requires_a_loaded_model(monkeypatch) -> None:
    """Reject prediction requests until both model assets are loaded."""
    service = model_service.ModelService()
    log_error = Mock()
    monkeypatch.setattr(model_service.logger, "error", log_error)

    with pytest.raises(ValueError, match="not loaded"):
        service.predict("hello")

    log_error.assert_called_once()


def test_load_model_loads_both_assets(monkeypatch, tmp_path: Path) -> None:
    """Load the vectorizer and create the ONNX Runtime session."""
    model_file = tmp_path / "model.onnx"
    vectorizer_file = tmp_path / "vectorizer.joblib"
    model_file.touch()
    vectorizer_file.touch()
    vectorizer = FakeVectorizer()
    monkeypatch.setattr(model_service.joblib, "load", Mock(return_value=vectorizer))
    monkeypatch.setattr(
        model_service.ort,
        "InferenceSession",
        fake_session_factory,
    )

    service = model_service.ModelService()
    service.load_model(model_file, vectorizer_file)

    assert service.vectorizer is vectorizer
    assert isinstance(service.session, FakeInferenceSession)


def test_predict_handles_single_and_multiple_inputs() -> None:
    """Return flattened binary labels for either supported input form."""
    service = model_service.ModelService()
    service.vectorizer = FakeVectorizer()
    service.session = FakeInferenceSession()

    assert service.predict(["hello", "offer"]) == [0, 1]
    assert service.predict("hello") == [0, 1]


def test_load_model_builds_missing_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Trigger the build pipeline when either configured asset is absent."""
    build_model = Mock()
    vectorizer = FakeVectorizer()
    monkeypatch.setattr(model_service, "build_model", build_model)
    monkeypatch.setattr(model_service.joblib, "load", Mock(return_value=vectorizer))
    monkeypatch.setattr(
        model_service.ort,
        "InferenceSession",
        fake_session_factory,
    )

    service = model_service.ModelService()
    service.load_model(
        tmp_path / "missing.onnx",
        tmp_path / "missing.joblib",
    )

    build_model.assert_called_once_with()
    assert service.vectorizer is vectorizer
    assert isinstance(service.session, FakeInferenceSession)


@pytest.mark.parametrize(
    "load_error",
    [FileNotFoundError, OSError, ValueError],
)
def test_load_model_raises_on_bad_vectorizer(
    monkeypatch,
    tmp_path: Path,
    load_error: type[Exception],
) -> None:
    """Raise a clear error when joblib cannot load the vectorizer."""
    model_file = tmp_path / "model.onnx"
    vectorizer_file = tmp_path / "vectorizer.joblib"
    model_file.touch()
    vectorizer_file.touch()
    load_vectorizer = Mock(side_effect=load_error("invalid vectorizer"))
    monkeypatch.setattr(model_service.joblib, "load", load_vectorizer)
    monkeypatch.setattr(
        model_service.ort,
        "InferenceSession",
        fake_session_factory,
    )
    log_critical = Mock()
    monkeypatch.setattr(model_service.logger, "critical", log_critical)

    with pytest.raises(ValueError, match="could not be loaded"):
        model_service.ModelService().load_model(model_file, vectorizer_file)

    assert log_critical.call_count == 2
