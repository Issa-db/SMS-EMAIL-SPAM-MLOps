"""
Model Training Module.

Loads prepared data, splits it into train/test sets, vectorizers the
text with CountVectorizer, trains a MultinomialNB classifier using
GridSearchCV hyperparameter tuning, evaluates the moddel, and exports the
trained model and vectorizer to disk (ONNX and joblib formats) for use by
the inference service.
"""

from pathlib import Path

import joblib
import pandas as pd
import scipy
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model() -> None:
    """
    Build and train the model using the prepared data.
    Returns:
        model: The trained model.
    """
    logger.info("Starting model building pipeline")
    X, y = prepare_data()
    logger.info(f"Prepare data: {len(X)} samples")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    logger.info(f"Split data: {len(X_train)} train / {len(X_test)} test")
    # Vectorize the text data
    X_train_vectorized, X_test_vectorized, vectorizer = vectorize_data(
        X_train,
        X_test)
    logger.debug(f"Vectorizer features: {X_train_vectorized.shape[1]} columns")
    # Train the model
    NB = train_model(X_train_vectorized, y_train)
    # Evaluate the model
    evaluate_model(NB, X_test_vectorized, y_test)

    save_model_onnx(
        NB,
        vectorizer,
        model_path=model_settings.model_path / model_settings.model_name,
        vectorizer_path=(
            model_settings.vectorizer_path / model_settings.vectorizer_name
            ),
    )
    logger.info("Model build pipeline complete")


def split_train_test(X: pd.Series, y: pd.Series) -> tuple[pd.Series,
                                                          pd.Series,
                                                          pd.Series,
                                                          pd.Series]:
    """
       Split arrays into random train and test subsets.

    Args:
        X (str): The input features (text messages).
        y (int): The target labels (spam or ham).
    """
    # Split the data into training and testing sets
    logger.debug(
        f"Splitting data: test_size=0.2, random_state=42, "
        f"total_samples={len(X)}"
        )
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42
                                                        )
    return X_train, X_test, y_train, y_test


def vectorize_data(X_train: pd.Series, X_test: pd.Series) -> tuple[
    scipy.sparse.spmatrix,
    scipy.sparse.spmatrix,
    CountVectorizer
]:
    """
    Vectorize the text data using CountVectorizer.

    Args:
        X_train (str): The input features for training (text messages).
        X_test (str): The input features for testing (text messages).
    """
    # Initialize CountVectorizer
    logger.debug(
        "Vectorizing with CountVectorizer "
        "(ngram_range=(1,2), min_df=5, stop_words='english')"
    )

    vectorizer = CountVectorizer(lowercase=True,
                                 stop_words="english",
                                 ngram_range=(1, 2),
                                 min_df=5)

    # Fit and transform the training data
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Transform the test data
    X_test_vectorized = vectorizer.transform(X_test)
    logger.debug(f"vocabulary size: {len(vectorizer.get_feature_names_out())}")
    return X_train_vectorized, X_test_vectorized, vectorizer


def train_model(X_train_vectorized: scipy.sparse.spmatrix,
                y_train: pd.Series) -> MultinomialNB:
    """
    Train the model using MultinomialNB.

    Args:
        X_train_vectorized (str): The input features for vectorized training.
        y_train (int): The target labels for training (spam or ham).
    """
    logger.info("starting GridSearchCV hyperparameter search")
    # build model and train it
    grid_space = {
        "alpha": [0.1, 0.5, 1.0],
        "fit_prior": [True, False],
    }
    grid_search = GridSearchCV(MultinomialNB(),
                               grid_space,
                               cv=5,
                               scoring="accuracy")
    # train the model
    grid_search.fit(X_train_vectorized, y_train)
    logger.info(f"best parms: {grid_search.best_params_}"
                f" best CV score: {grid_search.best_score_:.4f}")
    # return model
    return grid_search.best_estimator_


def evaluate_model(model: MultinomialNB,
                   X_test_vectorized: scipy.sparse.spmatrix,
                   y_test: pd.Series) -> None:
    """
    Evaluate the model's performance on the test set.

    Args:
        model: The trained model.
        X_test_vectorized (scipy.sparse._matrix):
        The input features for vectorized testing.
        y_test (int): The target labels for testing (spam or ham).
    """
    # Evaluate the model
    accuracy = model.score(X_test_vectorized, y_test)
    logger.info(f"model accuracy: {accuracy:.4f}")
    if accuracy < 0.99:
        logger.warning(f"Accuracy ({accuracy:.4f})"
                       f" is below expected threshold (0.99)")


def save_model_onnx(model: MultinomialNB,
                    vectorizer: CountVectorizer,
                    model_path: str | Path | None = None,
                    vectorizer_path: str | Path | None = None) -> None:
    """
    Save the trained model and vectorizer to disk in ONNX format.

    Args:
        model: The trained model.
        vectorizer: The CountVectorizer used for text vectorization.
        model_path (str): The path to save the ONNX model.
        vectorizer_path (str): The path to save the vectorizer.
    """
    try:
        # Save the vectorizer using joblib
        joblib.dump(vectorizer, vectorizer_path)
        logger.debug(f"vectorizer saved to {vectorizer_path}")
    except Exception as e:
        logger.error(f"failed to save vectorizer: {e}")
        raise

    try:
        n_features = len(vectorizer.get_feature_names_out())
        # Convert the model to ONNX format
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
    except Exception as e:
        logger.critical(f"ONNX conversion failed, model not saved: {e}")
    try:
        # Save the ONNX model to disk
        with open(model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
    except OSError as e:
        logger.critical(f"cannot write model file to disk: {e}")
        raise

    logger.info(f"Model saved to {model_path}"
                f" and vectorizer saved to {vectorizer_path}")


# run the build_model function to train and evaluate the model

model = build_model()
