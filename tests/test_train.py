from datetime import timedelta

import pytest

from subroformer.training.train import resolve_distributed_timeout, validate_training_data_sources
from tests.helpers import tiny_config


def test_resolve_distributed_timeout():
    assert resolve_distributed_timeout(None) is None
    assert resolve_distributed_timeout("null") is None
    assert resolve_distributed_timeout(7200) == timedelta(hours=2)


@pytest.mark.parametrize("value", [0, -1])
def test_resolve_distributed_timeout_rejects_non_positive_values(value):
    with pytest.raises(ValueError, match="distributed_timeout_seconds"):
        resolve_distributed_timeout(value)


def test_training_requires_public_folder_data():
    config = tiny_config()
    assert validate_training_data_sources(config, ["data/train"]) == "mss"
    with pytest.raises(ValueError, match="--data_path"):
        validate_training_data_sources(config, [])
    config.data = {"kind": "unsupported_adapter"}
    with pytest.raises(ValueError, match="expected 'mss'"):
        validate_training_data_sources(config, ["data/train"])
