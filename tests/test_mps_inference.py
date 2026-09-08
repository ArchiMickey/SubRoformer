from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from inference import load_separator, resolve_amp_dtype, resolve_device, separate_audio, separate_file
from model_utils import build_model
from tests.helpers import tiny_config


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"),
])
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device("auto").type == expected
    assert resolve_device("cpu").type == "cpu"


def test_unavailable_mps_fails_before_download(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="MPS is unavailable"):
        load_separator(device="mps")


@pytest.mark.parametrize("supported", [False, True])
def test_mps_auto_precision(monkeypatch, supported):
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *args: supported)
    assert resolve_amp_dtype(torch.device("mps")) == torch.float16
    assert resolve_amp_dtype(torch.device("mps"), "fp16") == torch.float16
    assert resolve_amp_dtype(torch.device("mps"), "fp32") is None
    if not supported:
        with pytest.raises(ValueError, match="BF16 is unavailable"):
            resolve_amp_dtype(torch.device("mps"), "bf16")
    else:
        assert resolve_amp_dtype(torch.device("mps"), "bf16") == torch.bfloat16


def test_cpu_auto_precision():
    assert resolve_amp_dtype(torch.device("cpu")) is None
    with pytest.raises(ValueError, match="fp32 precision on CPU"):
        resolve_amp_dtype(torch.device("cpu"), "fp16")


@pytest.mark.parametrize("supported", [False, True])
def test_cuda_auto_precision_uses_selected_device(monkeypatch, supported):
    from contextlib import contextmanager

    selected = []

    @contextmanager
    def device_context(device):
        selected.append(device)
        yield

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    assert resolve_amp_dtype(torch.device("cuda:1")) == (torch.bfloat16 if supported else torch.float16)
    assert selected == [torch.device("cuda:1")]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires a Metal GPU")
@pytest.mark.parametrize("joint", [False, True])
def test_mps_inference_matches_cpu(tmp_path: Path, joint: bool):
    torch.manual_seed(42)
    config = tiny_config(target_instrument=None if joint else "vocals")
    config.model.num_stems = 4 if joint else 1
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(build_model(config).state_dict(), checkpoint_path)
    audio = np.random.default_rng(42).standard_normal((1, 277)).astype(np.float32) * 0.01
    cpu_model, config = load_separator(config_path, checkpoint_path, device="cpu")
    expected = separate_audio(cpu_model, audio, config, batch_size=2)
    mps_model, config = load_separator(config_path, checkpoint_path, device="mps")
    assert next(mps_model.parameters()).device.type == "mps"
    actual = separate_audio(mps_model, audio, config, batch_size=2)
    for stem in expected:
        assert actual[stem].shape == (2, 277)
        assert np.isfinite(actual[stem]).all()
        np.testing.assert_allclose(actual[stem], expected[stem], rtol=1e-3, atol=1e-5)
    input_path = tmp_path / "input.wav"
    sf.write(input_path, audio.T, config.audio.sample_rate, subtype="FLOAT")
    paths = separate_file(config_path, checkpoint_path, str(input_path), str(tmp_path / "output"), device="mps", precision="fp32", batch_size=2)
    assert [path.stem for path in paths] == list(expected)
    for path in paths:
        output, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        assert sample_rate == config.audio.sample_rate
        np.testing.assert_allclose(output.T, actual[path.stem], rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires a Metal GPU")
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_mps_mixed_precision_forward(precision):
    if precision == "bf16" and not torch.backends.mps.is_macos_or_newer(15, 0):
        pytest.skip("Requires MPS BF16 support")
    config = tiny_config()
    config.model.num_stems = 4
    model = build_model(config).eval().to("mps")
    observed = []
    handle = model.down.register_forward_hook(lambda module, inputs, output: observed.append(output.dtype))
    audio = np.random.default_rng(42).standard_normal((2, 277)).astype(np.float32) * 0.01
    dtype = resolve_amp_dtype(torch.device("mps"), precision)
    outputs = separate_audio(model, audio, config, batch_size=2, amp_dtype=dtype)
    handle.remove()
    assert observed and all(value == dtype for value in observed)
    assert len(outputs) == 4
    assert all(value.shape == (2, 277) and np.isfinite(value).all() for value in outputs.values())
