from model_service import ModelService



def main():
    """Run spam detection inference on sample messages using the trained ONNX model"""
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
        
if __name__ == "__main__":
    main()