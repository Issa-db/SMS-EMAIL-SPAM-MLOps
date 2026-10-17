"""Tests for optional MLflow model-building run tracking."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from src.config.settings import APP_ROOT, Settings
from src.model.tracking import start_training_run


def test_tracking_disabled_yields_local_run_identifier() -> None:
    """Keep local model training usable without installing MLflow."""
    settings = Settings(mlflow_tracking_enabled=False)

    with start_training_run(settings) as run:
        run.log_params({"seed": 42})
        run.log_metrics({"accuracy": 0.9})

    assert len(run.run_id) == 32


def test_tracking_enabled_configures_mlflow_and_logs_values(
    monkeypatch,
    tmp_path,
) -> None:
    """Set the experiment and forward parameters, metrics, and artifacts."""
    calls = {
        "set_tracking_uri": Mock(),
        "set_experiment": Mock(),
        "log_params": Mock(),
        "log_metrics": Mock(),
        "log_artifact": Mock(),
    }
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=calls["set_tracking_uri"],
        set_experiment=calls["set_experiment"],
        start_run=lambda: nullcontext(active_run),
        log_params=calls["log_params"],
        log_metrics=calls["log_metrics"],
        log_artifact=calls["log_artifact"],
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    settings = Settings(
        mlflow_tracking_enabled=True,
        mlflow_tracking_uri="http://tracking.example",
        mlflow_experiment_name="unit-test",
    )
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"artifact")

    with start_training_run(settings) as run:
        run.log_params({"seed": 42, "best": {"alpha": 0.5}})
        run.log_metrics({"accuracy": 0.9})
        run.log_artifact(artifact)

    assert run.run_id == "run-123"
    calls["set_tracking_uri"].assert_called_once_with(
        "http://tracking.example",
    )
    calls["set_experiment"].assert_called_once_with("unit-test")
    calls["log_params"].assert_called_once_with(
        {"seed": "42", "best": '{"alpha": 0.5}'},
    )
    calls["log_metrics"].assert_called_once_with({"accuracy": 0.9})
    calls["log_artifact"].assert_called_once_with(str(artifact))


def test_tracking_defaults_to_a_local_sqlite_store(monkeypatch) -> None:
    """Use a local SQL backend by default for current MLflow releases."""
    set_tracking_uri = Mock()
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-default"))
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=set_tracking_uri,
        set_experiment=Mock(),
        start_run=lambda: nullcontext(active_run),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    settings = Settings(mlflow_tracking_enabled=True)

    with start_training_run(settings) as run:
        assert run.run_id == "run-default"

    expected_uri = f"sqlite:///{(APP_ROOT / 'mlflow.db').resolve()}"
    set_tracking_uri.assert_called_once_with(expected_uri)
