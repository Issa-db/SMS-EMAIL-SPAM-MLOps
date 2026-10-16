"""Train the classifier and export it with its matching vectorizer."""

from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.model.pipeline.preparation import prepare_data

TEST_SIZE = 0.2
RANDOM_STATE = 42


def convert_classifier(model: MultinomialNB, feature_count: int):
    """Convert the fitted classifier to an ONNX model."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, feature_count]))]
    return convert_sklearn(model, initial_types=initial_type)


def train_and_export(
    dataset: pd.DataFrame,
    model_path: Path,
    vectorizer_path: Path,
) -> float:
    """Train, evaluate, and export the paired classifier/vectorizer artifacts."""
    messages, targets = prepare_data(dataset)
    if len(messages) < 10 or targets.nunique() < 2:
        logger.error(
            "Training requires at least 10 rows and both classes; "
            "received {} rows and {} classes",
            len(messages),
            targets.nunique(),
        )
        raise ValueError("Training requires at least 10 rows from both classes")

    x_train, x_test, y_train, y_test = train_test_split(
        messages,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
    )
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)
    logger.debug(
        "Vectorized training data: train_rows={}, test_rows={}, features={}",
        x_train_vectors.shape[0],
        x_test_vectors.shape[0],
        x_train_vectors.shape[1],
    )
    search = GridSearchCV(
        MultinomialNB(),
        {"alpha": [0.1, 0.5, 1.0], "fit_prior": [True, False]},
        cv=3,
        scoring="accuracy",
    )
    search.fit(x_train_vectors, y_train)
    classifier = search.best_estimator_
    accuracy = float(classifier.score(x_test_vectors, y_test))
    logger.info("Model evaluation completed; held-out accuracy={:.4f}", accuracy)
    logger.debug("Selected classifier parameters: {}", search.best_params_)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_model = convert_classifier(classifier, len(vectorizer.vocabulary_))
    temporary_model = model_path.with_suffix(model_path.suffix + ".tmp")
    temporary_vectorizer = vectorizer_path.with_suffix(
        vectorizer_path.suffix + ".tmp",
    )
    logger.info("Exporting model bundle to configured artifact paths")
    try:
        temporary_model.write_bytes(onnx_model.SerializeToString())
        joblib.dump(vectorizer, temporary_vectorizer)
        temporary_model.replace(model_path)
        temporary_vectorizer.replace(vectorizer_path)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_vectorizer.unlink(missing_ok=True)

    logger.info("Model bundle exported; held-out accuracy={:.4f}", accuracy)
    return accuracy
