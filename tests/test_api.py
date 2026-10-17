"""HTTP contract tests for the inference API."""

from unittest.mock import Mock

from src import api
from src.api import create_app
from src.config.settings import Settings


def make_service() -> Mock:
    """Return an already-loaded inference fake with deterministic outputs."""
    service = Mock()
    service.session = object()
    service.vectorizer = object()
    service.predict.return_value = [0, 1]
    return service


def test_index_describes_api_routes() -> None:
    """Return endpoint links for a browser visiting the service root."""
    client = create_app(Settings(), make_service()).test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert response.json["endpoints"]["predict"] == "POST /predict"


def test_favicon_request_is_silent() -> None:
    """Return no content instead of a 404 for the browser favicon request."""
    client = create_app(Settings(), make_service()).test_client()

    response = client.get("/favicon.ico")

    assert response.status_code == 204


def test_health_reports_loaded_model() -> None:
    """Report readiness after app construction has loaded inference assets."""
    client = create_app(Settings(), make_service()).test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok", "model_loaded": True}


def test_predict_returns_human_readable_labels() -> None:
    """Map class IDs into stable API labels for each input message."""
    service = make_service()
    client = create_app(Settings(), service).test_client()

    response = client.post(
        "/predict",
        json={"messages": ["ordinary note", "claim a prize"]},
    )

    assert response.status_code == 200
    assert response.json["predictions"] == [
        {"label": "ham", "class_id": 0},
        {"label": "spam", "class_id": 1},
    ]
    service.predict.assert_called_once_with(["ordinary note", "claim a prize"])


def test_predict_works_with_published_model_bundle() -> None:
    """Exercise the checked-in vectorizer and ONNX model through HTTP."""
    client = create_app().test_client()

    response = client.post(
        "/predict",
        json={
            "messages": [
                "Hi, can we meet tomorrow for lunch?",
                "You have won a free prize! Call now to claim your reward.",
            ],
        },
    )

    assert response.status_code == 200
    assert [item["label"] for item in response.json["predictions"]] == [
        "ham",
        "spam",
    ]


def test_predict_rejects_invalid_requests() -> None:
    """Return HTTP 400 for malformed, empty, or blank-message payloads."""
    client = create_app(Settings(), make_service()).test_client()

    invalid_payloads = (
        None,
        {},
        {"messages": "not a list"},
        {"messages": [1]},
        {"messages": []},
        {"messages": ["  "]},
    )
    for payload in invalid_payloads:
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
        assert "error" in response.json


def test_create_app_loads_cold_inference_service() -> None:
    """Load the model once when the injected service has no artifacts yet."""
    service = Mock()
    service.session = None
    service.vectorizer = None

    create_app(Settings(), service)

    service.load_model.assert_called_once_with()


def test_create_app_resolves_configured_bundle_manifest(monkeypatch, tmp_path) -> None:
    """Build inference from the exact model pair selected by a manifest."""
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text("{}", encoding="utf-8")
    model_path = tmp_path / "versioned-model.onnx"
    vectorizer_path = tmp_path / "versioned-vectorizer.joblib"
    resolve_manifest = Mock(return_value=(model_path, vectorizer_path))
    constructor = Mock(return_value=make_service())
    monkeypatch.setattr(api, "resolve_bundle_manifest", resolve_manifest)
    monkeypatch.setattr(api, "ModelInferenceService", constructor)

    create_app(Settings(inference_bundle_manifest=manifest_path))

    resolve_manifest.assert_called_once_with(manifest_path)
    constructor.assert_called_once_with(model_path, vectorizer_path)
