from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from model import build_model
from config import settings


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
        
        if model_path is None:
            model_path = settings.model_path/ settings.model_name
        if vectorizer_path is None:
            vectorizer_path = settings.vectorizer_path / settings.vectorizer_name
        model_path = Path(model_path)
        vectorizer_path = Path(vectorizer_path)

        if not model_path.exists() or not vectorizer_path.exists():
            print("Model or vectorizer file not found. Building new model...")
            build_model()

        self.vectorizer = joblib.load(vectorizer_path)
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        if self.session is None or self.vectorizer is None:
            raise ValueError("Model or vectorizer could not be loaded. Please check the paths.")

        print(f"Model loaded from {model_path} and vectorizer loaded from {vectorizer_path}")
    
    def predict(self, message):
        """Predict spam/ham for a string or list of strings"""
        if self.vectorizer is None or self.session is None:
            raise ValueError("Model is not loaded. Call load_model() first.")
        if isinstance(message, str):
            message = [message]
        # transform the text message 
        x_vec = self.vectorizer.transform(message)
        x_array = x_vec.toarray().astype(np.float32) 
        
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        predictions = self.session.run([output_name], {input_name: x_array})[0]
    
        return predictions.astype(int).flatten().tolist()

        

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

            