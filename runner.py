import sys

from model_service import ModelService
from loguru import logger



def main():
    """Run spam detection inference on sample messages using the trained ONNX model"""
    logger.info("starting spam detection runner")
    service = ModelService()
    service.load_model()

    texts = [
        "Hi, can we meet tomorrow for lunch?",
        "You have won a free prize! Call now to claim your reward.",
        "Please check the attached invoice.",
    ]

    logger.debug(f"sample inputs: {texts}")
    preds = service.predict(texts)

    for msg, pred in zip(texts, preds):
        label = "spam" if pred == 1 else "ham"
        print(f"Message: {msg}")
        print("Predication:", label)
        print("-" * 50)
    
    logger.info("Runner completed successfully")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Runner failed with unhandled exception: {e}")
        raise