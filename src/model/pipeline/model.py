"""Train the classifier and export it with its matching vectorizer."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.model.pipeline.preparation import prepare_data

TEST_SIZE = 0.2
RANDOM_STATE = 42
VECTORIZER_PARAMETERS = {
    "lowercase": True,
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 5,
}
CLASSIFIER_SEARCH_SPACE = {
    "alpha": [0.1, 0.5, 1.0],
    "fit_prior": [True, False],
}


@dataclass(frozen=True)
class TrainingResult:
    """Evaluation and provenance details from one fitted model bundle."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    row_count: int
    train_row_count: int
    test_row_count: int
    feature_count: int
    class_counts: dict[str, int]
    best_parameters: dict[str, object]
    vectorizer_parameters: dict[str, object]
    classifier_search_space: dict[str, list[object]]


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
    *,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
    cross_validation_folds: int = 3,
    minimum_accuracy: float | None = None,
) -> TrainingResult:
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
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )
    vectorizer = CountVectorizer(**VECTORIZER_PARAMETERS)
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
        CLASSIFIER_SEARCH_SPACE,
        cv=cross_validation_folds,
        scoring="accuracy",
    )
    search.fit(x_train_vectors, y_train)
    classifier = search.best_estimator_
    predictions = classifier.predict(x_test_vectors)
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))
    logger.info("Model evaluation completed; held-out accuracy={:.4f}", accuracy)
    logger.debug("Selected classifier parameters: {}", search.best_params_)
    if minimum_accuracy is not None and accuracy < minimum_accuracy:
        logger.error(
            "Held-out accuracy {:.4f} is below configured minimum {:.4f}",
            accuracy,
            minimum_accuracy,
        )
        raise ValueError("Model did not meet the configured accuracy threshold")

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
        validate_exported_bundle(
            temporary_model,
            temporary_vectorizer,
            x_test,
            x_test_vectors,
            predictions,
        )
        temporary_model.replace(model_path)
        temporary_vectorizer.replace(vectorizer_path)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_vectorizer.unlink(missing_ok=True)

    logger.info("Model bundle exported; held-out accuracy={:.4f}", accuracy)
    return TrainingResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        row_count=len(messages),
        train_row_count=len(x_train),
        test_row_count=len(x_test),
        feature_count=x_train_vectors.shape[1],
        class_counts={
            str(label): int(count)
            for label, count in targets.value_counts().sort_index().items()
        },
        best_parameters=dict(search.best_params_),
        vectorizer_parameters=dict(VECTORIZER_PARAMETERS),
        classifier_search_space=dict(CLASSIFIER_SEARCH_SPACE),
    )


def validate_exported_bundle(
    model_path: Path,
    vectorizer_path: Path,
    test_messages: pd.Series,
    test_features,
    expected_predictions: np.ndarray,
) -> None:
    """Verify serialized preprocessing/model artifacts preserve predictions."""
    import onnxruntime as ort

    loaded_vectorizer = joblib.load(vectorizer_path)
    if not hasattr(loaded_vectorizer, "transform"):
        raise ValueError("Exported vectorizer does not provide transform()")
    serialized_features = loaded_vectorizer.transform(test_messages)
    if not np.array_equal(
        serialized_features.toarray(),
        test_features.toarray(),
    ):
        raise ValueError(
            "Exported vectorizer features differ from training features",
        )
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    actual_predictions = session.run(
        [output_name],
        {input_name: test_features.toarray().astype(np.float32)},
    )[0].astype(int).flatten()
    if not np.array_equal(actual_predictions, expected_predictions):
        raise ValueError("Exported ONNX predictions differ from the trained model")
    logger.info("Exported ONNX and vectorizer artifacts passed validation")
