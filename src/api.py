"""Flask HTTP boundary for the model inference service."""

from flask import Flask, jsonify, request
from loguru import logger
from pydantic import BaseModel, StrictStr, ValidationError

from src.config.settings import Settings
from src.model.artifacts import resolve_bundle_manifest
from src.model.model_inference import ModelInferenceService


class PredictionRequest(BaseModel):
    """Validated batch of message text accepted by the prediction endpoint."""

    messages: list[StrictStr]


def create_app(
    settings: Settings | None = None,
    inference_service: ModelInferenceService | None = None,
) -> Flask:
    """Create the API and load inference dependencies at application startup."""
    application_settings = settings or Settings()
    model_path = application_settings.model_path
    vectorizer_path = application_settings.vectorizer_path
    if application_settings.inference_bundle_manifest is not None:
        try:
            model_path, vectorizer_path = resolve_bundle_manifest(
                application_settings.inference_bundle_manifest,
            )
        except Exception:
            logger.opt(exception=True).critical(
                "Inference startup failed while resolving the selected bundle",
            )
            raise
    service = inference_service or ModelInferenceService(
        model_path,
        vectorizer_path,
    )
    if service.session is None or service.vectorizer is None:
        try:
            service.load_model()
        except Exception:
            logger.opt(exception=True).critical(
                "Inference service startup failed while loading its model bundle",
            )
            raise

    app = Flask(__name__)

    @app.get("/")
    def index():
        """Describe available endpoints for browser and operator requests."""
        return jsonify(
            {
                "service": "spam-inference-api",
                "status": "ready",
                "endpoints": {
                    "health": "GET /health",
                    "predict": "POST /predict",
                },
            },
        )

    @app.get("/favicon.ico")
    def favicon():
        """Avoid a noisy 404 for browsers that request a default favicon."""
        return "", 204

    @app.get("/health")
    def health():
        """Report readiness after both model artifacts have loaded."""
        return jsonify({"status": "ok", "model_loaded": True})

    @app.post("/predict")
    def predict():
        """Validate a batch request and return binary predictions."""
        try:
            payload = PredictionRequest.model_validate(
                request.get_json(silent=True),
            )
        except (ValidationError, TypeError, ValueError) as error:
            logger.warning(
                "Rejected malformed prediction request ({})",
                type(error).__name__,
            )
            return jsonify({"error": str(error)}), 400
        if not payload.messages or any(
            not message.strip() for message in payload.messages
        ):
            logger.warning("Rejected prediction request with empty messages")
            return jsonify({"error": "messages must contain non-empty strings"}), 400

        try:
            labels = service.predict(payload.messages)
        except Exception:
            logger.exception("Prediction request failed during inference")
            return jsonify({"error": "Prediction failed"}), 500
        logger.debug("Prediction completed for batch of {} messages", len(labels))
        predictions = [
            {"label": "spam" if label else "ham", "class_id": label}
            for label in labels
        ]
        return jsonify({"predictions": predictions})

    return app
