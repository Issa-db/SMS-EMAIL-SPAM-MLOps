from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from model import build_model
from config import settings

from loguru import logger 
class ModelService:
    """Load and use the saved spam model and vectorizer"""
    def __init__(self):
        self.model = None 
        self.vectorizer = None 
        self.session = None 
    
    def load_model(
        self,
        model_path: str | Path | None = None,
        vectorizer_path: str | Path | None = None,
        model_name: str = "NB",
        vectorizer_name: str = "vectorizer"
        ):
        """
        Load the trained model and vectorizer from local disk.
        
        Args:
            model_path (Path): The path to the ONNX model file.
            vectorizer_path (Path): The path to the vectorizer file.
        """
        logger.info("loading model and vectorizer")
        if model_path is None:
            model_path = settings.model_path/ settings.model_name
        if vectorizer_path is None:
            vectorizer_path = settings.vectorizer_path / settings.vectorizer_name
        model_path = Path(model_path)
        vectorizer_path = Path(vectorizer_path)

        logger.debug(f"resolve model_path={model_path}, vectorizer_path={vectorizer_path}")
        if not model_path.exists() or not vectorizer_path.exists():
            logger.warning("Model or vectorizer file not found. Building new model...")
            build_model()

        try:
            self.vectorizer = joblib.load(vectorizer_path)
            logger.debug("vectorizer loaded successfully")
        except Exception as e:
            logger.critical(f"failed to load vectorizer from {vectorizer_path}: {e}")
            
        try:    
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            logger.debug("ONNX inference session created succesfully")
        except Exception as e:
            logger.critical(f"failed to create ONNX inference session from {model_path}: {e}")
        
        if self.session is None or self.vectorizer is None:
            logger.critical("Model or vectorizer could not be loaded despite no exception raised")
            raise ValueError("Model or vectorizer could not be loaded. Please check the paths.")

        logger.info(f"Model loaded from {model_path} and vectorizer loaded from {vectorizer_path}")
    
    def predict(self, message):
        """Predict spam/ham for a string or list of strings"""
        if self.vectorizer is None or self.session is None:
            logger.error("predict() called before load_model()")
            raise ValueError("Model is not loaded. Call load_model() first.")
        
        if isinstance(message, str):
            message = [message]
            
        logger.debug(f"predicting on {len(message)} message(s)")
        # transform the text message 
        x_vec = self.vectorizer.transform(message)
        x_array = x_vec.toarray().astype(np.float32) 
        
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run([output_name], {input_name: x_array})[0]
        
        result = predictions.astype(int).flatten().tolist()
        logger.debug(f"Predication: {result} ")
        return result
        

if __name__ == "__main__":
    service = ModelService()
    service.load_model()

    texts = [
        "Hi, can we meet tomorrow for lunch?",
        "You have won a free prize! Call now to claim your reward.",
        "Please check the attached invoice.",
    ]

    preds = service.predict(texts)

    for msg, pred in zip(texts, preds):
        label = "spam" if pred == 1 else "ham"
        print(f"Message: {msg}")
        print("Predication:", label)
        print("-" * 50)

            