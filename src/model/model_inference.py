"""Load the published artifact pair and predict spam/ham labels."""

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from loguru import logger

from src.model.pipeline.preparation import clean_text


class ModelInferenceService:
    """Make predictions only; missing artifacts never start model training."""

    def __init__(self, model_path: Path, vectorizer_path: Path) -> None:
        """Capture exact paths for the model and its preprocessing artifact."""
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        self.session = None

    def load_model(self) -> None:
        """Load both artifacts or fail with their missing paths."""
        logger.debug("Loading inference model bundle from {}", self.model_path.parent)
        missing = [
            path for path in (self.model_path, self.vectorizer_path)
            if not path.is_file()
        ]
        if missing:
            logger.critical("Required model bundle artifacts are missing: {}", missing)
            raise FileNotFoundError(
                "Required model bundle files not found: "
                + ", ".join(str(path) for path in missing),
            )
        try:
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"],
            )
        except Exception:
            logger.opt(exception=True).critical(
                "Failed to load the inference model bundle from {}",
                self.model_path.parent,
            )
            raise
        logger.info("Inference model bundle loaded from {}", self.model_path.parent)

    def predict(self, messages: str | list[str]) -> list[int]:
        """Return one binary class ID for each non-empty message."""
        if self.vectorizer is None or self.session is None:
            logger.error("Prediction requested before model bundle was loaded")
            raise RuntimeError("Call load_model() before predict()")
        inputs = [messages] if isinstance(messages, str) else messages
        if not inputs or any(
            not isinstance(message, str) or not message.strip()
            for message in inputs
        ):
            logger.warning("Prediction service received an empty or invalid batch")
            raise ValueError("Provide one or more non-empty message strings")
        logger.debug("Running inference for batch of {} messages", len(inputs))
        cleaned_inputs = [clean_text(message) for message in inputs]
        features = self.vectorizer.transform(cleaned_inputs)
        feature_array = features.toarray().astype(np.float32)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run(
            [output_name],
            {input_name: feature_array},
        )[0]
        return predictions.astype(int).flatten().tolist()
