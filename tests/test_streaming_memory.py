import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

import inference
from model_utils import build_model
from tests.helpers import tiny_config
from tests.legacy_overlap_add import separate_overlap_add as legacy_overlap_add


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("joint", [False, True])
def test_real_model_is_bit_identical_to_legacy(device, joint):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Requires Metal")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    config = tiny_config(target_instrument=None if joint else "vocals")
    config.model.num_stems = 4 if joint else 1
    model = build_model(config).to(device).eval()
    audio = np.random.default_rng(6).normal(0, .01, (2, 301)).astype(np.float32)
    expected = legacy_overlap_add(model, audio, 128, 31, 2)
    actual = inference.separate_overlap_add(model, audio, 128, 31, 2)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("mono", [False, True])
@pytest.mark.parametrize("backend", ["dsp", "dsp2", "dsp3"])
def test_file_export_matches_legacy_samples(tmp_path, joint, mono, backend):
    config = tiny_config(target_instrument=None if joint else "vocals")
    config.model.dsp_backend = backend
    config.model.num_stems = 4 if joint else 1
    config_path = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pt"
    source = tmp_path / "input.wav"
    OmegaConf.save(config, config_path)
    model = build_model(config).eval()
    torch.save(model.state_dict(), checkpoint)
    audio = np.random.default_rng(9).normal(0, .01, (1 if mono else 2, 301)).astype(np.float32)
    sf.write(source, audio.T, 1600, subtype="FLOAT")
    expected = legacy_overlap_add(model, np.repeat(audio, 2, axis=0) if mono else audio, 128, 32, 2)
    if not joint:
        expected = expected[None]
    paths = inference.separate_file(config_path, checkpoint, str(source), str(tmp_path / "output"), batch_size=2)
    for path, stem in zip(paths, expected):
        actual, rate = sf.read(path, dtype="float32", always_2d=True)
        assert rate == 1600 and sf.info(path).subtype == "FLOAT"
        np.testing.assert_array_equal(actual.T, stem)


@pytest.mark.parametrize("samples", [1000, 100000])
def test_callback_only_allocations_do_not_scale_with_track(samples, monkeypatch):
    class Separator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, audio):
            return torch.stack([audio * self.scale] * 4, dim=1)

    model = Separator()
    audio = np.zeros((2, samples), dtype=np.float32)
    allocations = []
    zeros = np.zeros
    empty = np.empty

    def record(factory, shape, *args, **kwargs):
        allocations.append(shape)
        return factory(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", lambda shape, *args, **kwargs: record(zeros, shape, *args, **kwargs))
    monkeypatch.setattr(np, "empty", lambda shape, *args, **kwargs: record(empty, shape, *args, **kwargs))
    received = 0

    def consume(start, chunk):
        nonlocal received
        assert start == received
        received += chunk.shape[-1]

    result = inference.separate_overlap_add(
        model, audio, 128, 32, 3, on_chunk=consume, collect_output=False,
    )
    assert result is None and received == samples
    assert (4, 2, 192) in allocations
    assert max(np.prod(shape) for shape in allocations) <= 4 * 2 * 192


def test_callback_only_requires_consumer():
    with pytest.raises(ValueError, match="on_chunk"):
        inference.separate_overlap_add(
            torch.nn.Linear(1, 1), np.zeros((2, 128)), 128, 32, 1, collect_output=False,
        )
