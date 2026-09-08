import numpy as np
import pytest
import torch

from app.streaming import separate_audio, separate_overlap_add
from tests.legacy_overlap_add import separate_overlap_add as reference_overlap_add
from model_utils import build_model
from inference import load_separator
from checkpoint_utils import export_safetensors
from omegaconf import OmegaConf
from tests.helpers import tiny_config


class Separator(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))
        self.joint = joint

    def forward(self, audio):
        output = audio * self.scale + audio.mean(dim=-1, keepdim=True)
        return torch.stack((output, -output), dim=1) if self.joint else output


@pytest.mark.parametrize("samples", [1, 20, 128, 277, 800])
@pytest.mark.parametrize("batch", [1, 3, 100])
@pytest.mark.parametrize("hop", [1, 31, 32, 128])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("collect_output", [False, True])
def test_stream_matches_unchanged_engine(samples, batch, hop, joint, collect_output):
    model = Separator(joint)
    audio = np.random.default_rng(5).normal(size=(2, samples)).astype(np.float32)
    chunks = []
    result = separate_overlap_add(
        model, audio, 128, hop, batch, on_chunk=lambda start, chunk: chunks.append((start, chunk)),
        collect_output=collect_output,
    )
    expected = reference_overlap_add(model, audio, 128, hop, batch)
    if collect_output:
        np.testing.assert_array_equal(result, expected)
    else:
        assert result is None
    position = 0
    for start, chunk in chunks:
        assert start == position
        np.testing.assert_array_equal(chunk, expected[..., start:start + chunk.shape[-1]])
        position += chunk.shape[-1]
    assert position == samples


@pytest.mark.parametrize("joint", [False, True])
def test_real_model_named_stream(joint):
    config = tiny_config(target_instrument=None if joint else "vocals")
    config.model.num_stems = 4 if joint else 1
    model = build_model(config)
    audio = np.random.default_rng(9).normal(0, .01, (1, 277)).astype(np.float32)
    chunks = []
    result = separate_audio(model, audio, config, batch_size=2, on_chunk=lambda start, values: chunks.append(values))
    for stem in result:
        np.testing.assert_array_equal(np.concatenate([chunk[stem] for chunk in chunks], axis=-1), result[stem])


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_fp16_checkpoint_stream_keeps_weights_and_activations_half(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Requires a Metal GPU")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a CUDA GPU")
    config = tiny_config()
    config.model.num_stems = 4
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    source = tmp_path / "model.pt"
    torch.save(build_model(config).state_dict(), source)
    checkpoint = export_safetensors(source, tmp_path / "fp16.safetensors", fp16=True)
    model, config = load_separator(config_path, checkpoint, device, preserve_dtype=True)
    observed = []
    handle = model.down.register_forward_hook(lambda module, inputs, output: observed.append(output.dtype))
    audio = np.random.default_rng(9).normal(0, .01, (2, 277)).astype(np.float32)
    try:
        result = separate_audio(model, audio, config, batch_size=2, amp_dtype=torch.float16)
    finally:
        handle.remove()
    assert all(parameter.dtype == torch.float16 for parameter in model.parameters())
    assert observed and all(dtype == torch.float16 for dtype in observed)
    assert len(result) == 4
    assert all(value.shape == (2, 277) and np.isfinite(value).all() for value in result.values())


def test_cancel_after_first_callback():
    chunks = []
    with pytest.raises(InterruptedError):
        separate_overlap_add(
            Separator(False), np.zeros((2, 400), dtype=np.float32), 128, 32, 1,
            on_chunk=lambda start, value: chunks.append(value), should_cancel=lambda: bool(chunks),
        )
    assert len(chunks) == 1


def test_nonfinite_output_rejected_before_callback():
    model = Separator(False)
    model.scale.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        separate_overlap_add(model, np.zeros((2, 100)), 128, 32, 1)
