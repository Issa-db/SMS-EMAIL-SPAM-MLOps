"""Tests for model training, evaluation, and artifact export."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from model.pipeline import model as pipeline

SAMPLE_COUNT = 20
EXPECTED_TEST_COUNT = 4
EXPECTED_CV_FOLDS = 5


class GridSearchStub:
    """Capture grid-search options and fit the supplied estimator."""

    def __init__(self, estimator, param_grid, **options) -> None:
        """Save the estimator and search settings for assertions."""
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = options["cv"]
        self.best_estimator_ = estimator
        self.best_params_ = {"alpha": 0.5}
        self.best_score_ = 1.0

    def fit(self, features, labels) -> None:
        """Fit the configured estimator on the provided training data."""
        self.estimator.fit(features, labels)


def test_split_train_test_is_repeatable() -> None:
    """Use the fixed seed to create stable train and test partitions."""
    features = pd.Series(
        [f"message {index}" for index in range(SAMPLE_COUNT)],
    )
    labels = pd.Series([index % 2 for index in range(SAMPLE_COUNT)])

    first_split = pipeline.split_train_test(features, labels)
    second_split = pipeline.split_train_test(features, labels)

    assert all(
        first_part.equals(second_part)
        for first_part, second_part in zip(first_split, second_split)
    )
    assert len(first_split[1]) == EXPECTED_TEST_COUNT


def test_vectorizer_uses_training_messages() -> None:
    """Build the vocabulary from training text and ignore unseen test words."""
    training_messages = pd.Series(
        ["apple pear", "apple fruit", "apple pie"] * 5,
    )
    test_messages = pd.Series(["apple pear", "unseen token"])

    training_vectors, test_vectors, vectorizer = pipeline.vectorize_data(
        training_messages,
        test_messages,
    )

    assert training_vectors.shape[0] == 15
    assert test_vectors.shape[0] == 2
    assert "unseen" not in vectorizer.vocabulary_


def test_train_model_uses_grid_search(monkeypatch) -> None:
    """Search the expected alpha values and return the fitted NB estimator."""
    captured_searches = []

    def grid_search_factory(estimator, param_grid, **options):
        search = GridSearchStub(estimator, param_grid, **options)
        captured_searches.append(search)
        return search

    monkeypatch.setattr(pipeline, "GridSearchCV", grid_search_factory)
    training_vectors = csr_matrix([[2, 0], [0, 2]])
    training_labels = pd.Series([0, 1])

    trained_model = pipeline.train_model(training_vectors, training_labels)

    search = captured_searches[0]
    assert isinstance(search.estimator, MultinomialNB)
    assert search.param_grid["alpha"] == [0.1, 0.5, 1.0]
    assert search.cv == EXPECTED_CV_FOLDS
    assert isinstance(trained_model, MultinomialNB)
    assert trained_model.predict(csr_matrix([[1, 0]])).tolist() == [0]


def test_evaluation_warns_on_low_accuracy(monkeypatch) -> None:
    """Log a warning when held-out accuracy misses the configured threshold."""
    warning = Mock()
    monkeypatch.setattr(pipeline.logger, "warning", warning)
    model = Mock()
    model.score.return_value = 0.5

    pipeline.evaluate_model(model, csr_matrix([[1]]), pd.Series([1]))

    warning.assert_called_once()
    assert "below expected threshold" in warning.call_args.args[0]


def test_save_model_onnx_writes_both_artifacts(tmp_path: Path) -> None:
    """Export a fitted classifier to ONNX and persist its vectorizer."""
    vectorizer = CountVectorizer().fit(["apple", "pear"])
    training_vectors = vectorizer.transform(["apple", "pear"])
    classifier = MultinomialNB().fit(training_vectors, [0, 1])
    model_path = tmp_path / "classifier.onnx"
    vectorizer_path = tmp_path / "vectorizer.joblib"

    pipeline.save_model_onnx(
        classifier,
        vectorizer,
        model_path,
        vectorizer_path,
    )

    assert model_path.stat().st_size > 0
    assert vectorizer_path.stat().st_size > 0


def test_build_model_runs_each_pipeline_stage(monkeypatch) -> None:
    """Prepare data, train, evaluate, then export the resulting model."""
    training_data = (pd.Series(["sample"]), pd.Series([0]))
    train_test_data = (
        training_data[0],
        training_data[0],
        training_data[1],
        training_data[1],
    )
    vectorized_data = (
        csr_matrix([[1]]),
        csr_matrix([[1]]),
        "vectorizer",
    )
    prepare_data = Mock(return_value=training_data)
    split_data = Mock(return_value=train_test_data)
    vectorize_data = Mock(return_value=vectorized_data)
    train_model = Mock(return_value="trained model")
    evaluate_model = Mock()
    save_model = Mock()
    monkeypatch.setattr(pipeline, "prepare_data", prepare_data)
    monkeypatch.setattr(pipeline, "split_train_test", split_data)
    monkeypatch.setattr(pipeline, "vectorize_data", vectorize_data)
    monkeypatch.setattr(pipeline, "train_model", train_model)
    monkeypatch.setattr(pipeline, "evaluate_model", evaluate_model)
    monkeypatch.setattr(pipeline, "save_model_onnx", save_model)

    pipeline.build_model()

    prepare_data.assert_called_once_with()
    split_data.assert_called_once_with(*training_data)
    vectorize_data.assert_called_once_with(*train_test_data[:2])
    train_model.assert_called_once_with(vectorized_data[0], train_test_data[2])
    evaluate_model.assert_called_once_with(
        "trained model",
        vectorized_data[1],
        train_test_data[3],
    )
    save_model.assert_called_once()


def test_evaluation_accepts_high_accuracy(monkeypatch) -> None:
    """Avoid the low-accuracy warning when evaluation meets its threshold."""
    warning = Mock()
    monkeypatch.setattr(pipeline.logger, "warning", warning)
    model = Mock()
    model.score.return_value = 1.0

    pipeline.evaluate_model(model, np.ones((1, 1)), pd.Series([1]))

    warning.assert_not_called()
