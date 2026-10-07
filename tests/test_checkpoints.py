from pathlib import Path

import pytest
import torch

from subroformer.checkpoint_utils import export_safetensors, load_checkpoint, load_model_checkpoint


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


def test_fp16_export_preserves_metadata_and_nonfloating_tensors(tmp_path):
    from safetensors import safe_open

    source = tmp_path / "source.pt"
    state = {"weight": torch.randn(2, 3), "count": torch.tensor(4), "mask": torch.tensor([True, False])}
    torch.save({"model": state, "global_update": 1400000}, source)
    original = export_safetensors(source, tmp_path / "original.safetensors", metadata={"weights": "ema"})
    destination = export_safetensors(original, tmp_path / "fp16.safetensors", fp16=True)
    restored = load_checkpoint(destination)["model"]
    for name, value in state.items():
        expected = value.half() if value.is_floating_point() else value
        assert restored[name].dtype == expected.dtype
        assert torch.equal(restored[name], expected)
    with safe_open(destination, framework="pt") as checkpoint:
        assert checkpoint.metadata() == {"weights": "ema", "global_update": "1400000", "precision": "fp16"}
    assert load_checkpoint(original)["model"]["weight"].dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_load_preserves_checkpoint_dtype_when_requested(tmp_path, dtype):
    source = torch.nn.Linear(3, 2).to(dtype)
    path = tmp_path / "model.pt"
    torch.save(source.state_dict(), path)
    restored = torch.nn.Linear(3, 2)
    load_model_checkpoint(restored, path, preserve_dtype=True)
    assert next(restored.parameters()).dtype == dtype
    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)
    default = torch.nn.Linear(3, 2)
    load_model_checkpoint(default, path)
    assert next(default.parameters()).dtype == torch.float32
