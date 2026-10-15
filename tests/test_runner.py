"""Tests for the inference runner entry point."""

from unittest.mock import Mock

import runner

SAMPLE_MESSAGE_COUNT = 5
SAMPLE_LABELS = (0, 1, 0, 0, 1)


def test_main_loads_and_predicts(monkeypatch) -> None:
    """Load the model once and submit all sample messages for prediction."""
    service = Mock()
    service.predict.return_value = SAMPLE_LABELS
    service_factory = Mock(return_value=service)
    monkeypatch.setattr(runner, "ModelService", service_factory)

    runner.main()

    service_factory.assert_called_once_with()
    service.load_model.assert_called_once_with()
    assert len(service.predict.call_args.args[0]) == SAMPLE_MESSAGE_COUNT
