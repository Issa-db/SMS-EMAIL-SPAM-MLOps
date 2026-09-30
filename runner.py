"""
Runner Module.

Application entry point. Loads the trained spam classifier via
ModelService and runs inference on a set of sample email messages.
"""

from loguru import logger

from model.model_service import ModelService


def main() -> None:
    """Run detection inference on sample messages using the trained ONNX"""
    logger.info("starting spam detection runner")
    service = ModelService()
    service.load_model()

    texts = [
        "Hi, can we meet tomorrow for lunch?",
        "You have won a free prize! Call now to claim your reward.",
        "Please check the attached invoice.",
        (
            "I'm reaching out to follow up on my previous message and "
            "would appreciate any updates when you have a moment."
        ),
        (
            "Congratulations! You've been selected to win a $1000 gift "
            "card. Click here now to claim before it expires!"
        ),
    ]

    logger.debug(f"sample inputs: {texts}")
    preds = service.predict(texts)

    for msg, pred in zip(texts, preds):
        label = "spam" if pred == 1 else "ham"
        print(f"Message: {msg}")
        print(f"Prediction: {label}")
        print("-" * 50)

    logger.info("Runner completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Runner failed with unhandled exception: {e}")
        raise
