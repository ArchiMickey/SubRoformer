from pathlib import Path

import pytest
import torch

from checkpoint_utils import export_safetensors, load_checkpoint, load_model_checkpoint


@pytest.mark.parametrize("wrapper", [None, "model", "state_dict", "model_state_dict", "state"])
def test_safetensors_roundtrip(tmp_path: Path, wrapper):
    model = torch.nn.Linear(3, 2)
    state = model.state_dict()
    checkpoint = state if wrapper is None else {wrapper: state, "epoch": 9, "global_update": 1400000}
    source = tmp_path / "source.pt"
    destination = tmp_path / "model.safetensors"
    torch.save(checkpoint, source)
    export_safetensors(source, destination)
    restored = torch.nn.Linear(3, 2)
    load_model_checkpoint(restored, destination)
    for name, value in state.items():
        assert torch.equal(value, restored.state_dict()[name])
    if wrapper:
        assert load_checkpoint(destination)["global_update"] == 1400000
        assert load_checkpoint(destination)["epoch"] == 9
    with pytest.raises(FileExistsError):
        export_safetensors(source, destination)
    with pytest.raises(RuntimeError):
        load_model_checkpoint(torch.nn.Linear(4, 2), destination)


def test_export_handles_shared_noncontiguous_tensors(tmp_path: Path):
    weight = torch.arange(6).reshape(2, 3).T
    source = tmp_path / "source.pt"
    destination = tmp_path / "model.safetensors"
    torch.save({"weight": weight, "alias": weight}, source)
    export_safetensors(source, destination)
    state = load_checkpoint(destination)["model"]
    assert torch.equal(state["weight"], weight)
    assert torch.equal(state["alias"], weight)
