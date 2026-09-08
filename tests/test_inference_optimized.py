import pytest
import torch

from benchmarks.reference_model import build_reference_model
from inference_optimized import OPTIMIZATION_STEPS, optimize_for_inference
from model_utils import build_model
from tests.helpers import tiny_config


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("level", OPTIMIZATION_STEPS)
@pytest.mark.parametrize("num_stems", [1, 4])
def test_optimized_forward_is_bitwise_equal(device, level, num_stems):
    torch.manual_seed(123)
    config = tiny_config()
    config.model.num_stems = num_stems
    original = build_reference_model(config).eval().to(device)
    optimized = optimize_for_inference(original, level)
    assert type(original.sb_filter) is not type(optimized.sb_filter)
    assert original.state_dict().keys() == optimized.state_dict().keys()
    for name, tensor in original.state_dict().items():
        assert torch.equal(tensor, optimized.state_dict()[name])
        assert tensor.data_ptr() != optimized.state_dict()[name].data_ptr()
    precisions = [None, torch.float16] if device == "mps" else [None]
    for precision in precisions:
        for batch_size, samples in [(1, 128), (2, 139), (1, 128)]:
            audio = torch.randn(batch_size, 2, samples, device=device) * 0.02
            with torch.inference_mode(), torch.autocast(
                device, dtype=precision, enabled=precision is not None,
            ):
                expected = original(audio)
                actual = optimized(audio)
                repeated = optimized(audio)
            assert torch.isfinite(actual).all()
            assert torch.equal(expected, actual)
            assert torch.equal(actual, repeated)
        with torch.inference_mode(), torch.autocast(
            device, dtype=precision, enabled=precision is not None,
        ):
            silence = torch.zeros(1, 2, 128, device=device)
            assert torch.equal(original(silence), optimized(silence))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("level", OPTIMIZATION_STEPS)
def test_filter_reload_and_mutation_invalidate_cache(device, level):
    original = build_reference_model(tiny_config()).eval().to(device)
    optimized = optimize_for_inference(original, level)
    audio = torch.randn(1, 2, 128, device=device)
    with torch.inference_mode():
        optimized(audio)
        original.sb_filter.w.mul_(0.9)
        optimized.load_state_dict(original.state_dict())
        assert optimized.sb_filter._cache_key is None
        assert torch.equal(original(audio), optimized(audio))
        original.sb_filter.up.mul_(0.9)
        optimized.sb_filter.up.mul_(0.9)
        original.sb_filter.f_center.mul_(0.9)
        optimized.sb_filter.f_center.mul_(0.9)
        assert torch.equal(original(audio), optimized(audio))
    optimized.to("cpu")
    assert optimized.sb_filter._cache_key is None


@pytest.mark.parametrize("level", OPTIMIZATION_STEPS)
def test_gradients_use_original_filter_path(level):
    original = build_reference_model(tiny_config()).eval()
    optimized = optimize_for_inference(original, level)
    audio = torch.randn(1, 2, 128, requires_grad=True)
    expected = original(audio)
    expected_gradient = torch.autograd.grad(expected.sum(), audio)[0]
    actual = optimized(audio)
    actual_gradient = torch.autograd.grad(actual.sum(), audio)[0]
    assert torch.equal(expected, actual)
    assert torch.equal(expected_gradient, actual_gradient)
    assert optimized.sb_filter._cache_key is None


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires MPS")
@pytest.mark.parametrize("level", OPTIMIZATION_STEPS)
def test_mps_half_weights_remain_bitwise_equal(level):
    original = build_reference_model(tiny_config()).eval().half().to("mps")
    optimized = optimize_for_inference(original, level)
    audio = torch.randn(1, 2, 128, device="mps") * 0.02
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
        for _ in range(2):
            assert torch.equal(original(audio), optimized(audio))


def test_cache_is_bounded_to_current_shape_and_resets_for_chunk_size():
    original = build_reference_model(tiny_config()).eval()
    optimized = optimize_for_inference(original, max(OPTIMIZATION_STEPS))
    first_cache_bytes = None
    with torch.inference_mode():
        for samples in (128, 160, 128):
            audio = torch.randn(1, 2, samples)
            assert torch.equal(original(audio), optimized(audio))
            if first_cache_bytes is None:
                first_cache_bytes = optimized.sb_filter.cache_bytes()
        assert first_cache_bytes > 0
        assert optimized.sb_filter.cache_bytes() == first_cache_bytes
        original.sb_filter.chunk_size = 1
        optimized.sb_filter.chunk_size = 1
        assert torch.equal(original(audio), optimized(audio))
    optimized.sb_filter.clear_cache()
    assert optimized.sb_filter.cache_bytes() == 0


@pytest.mark.parametrize("level", OPTIMIZATION_STEPS)
def test_inference_created_buffers_use_safe_uncached_path(level):
    with torch.inference_mode():
        original = build_reference_model(tiny_config()).eval()
        optimized = optimize_for_inference(original, level)
        audio = torch.randn(1, 2, 128)
        assert torch.equal(original(audio), optimized(audio))
        assert optimized.sb_filter.cache_bytes() == 0


def test_default_is_filter_spectrum_cache_without_modulation_cache():
    original = build_reference_model(tiny_config())
    optimized = optimize_for_inference(original)
    assert original.training
    assert not optimized.training
    assert optimized.sb_filter.level == 3
    with torch.inference_mode():
        optimized(torch.randn(1, 2, 128))
    assert optimized.sb_filter._filter_spectra
    assert not optimized.sb_filter._modulations


@pytest.mark.parametrize("device", DEVICES)
def test_default_model_matches_legacy_and_clears_cache_for_training(device):
    config = tiny_config()
    config.model.num_stems = 4
    original = build_reference_model(config).to(device)
    merged = build_model(config).to(device)
    merged.load_state_dict(original.state_dict(), strict=True)
    assert merged.sb_filter.level == 3
    assert merged.state_dict().keys() == original.state_dict().keys()
    audio = torch.randn(2, 2, 139, device=device) * 0.02
    original.eval()
    merged.eval()
    with torch.inference_mode():
        for _ in range(2):
            assert torch.equal(original(audio), merged(audio))
    assert merged.sb_filter.cache_bytes() > 0
    assert not merged.sb_filter._modulations
    original.train()
    merged.train()
    assert merged.sb_filter.cache_bytes() == 0
    reference_audio = audio.detach().requires_grad_()
    merged_audio = audio.detach().clone().requires_grad_()
    original(reference_audio).sum().backward()
    merged(merged_audio).sum().backward()
    if device == "mps":
        torch.testing.assert_close(reference_audio.grad, merged_audio.grad, rtol=1e-5, atol=1e-6)
    else:
        assert torch.equal(reference_audio.grad, merged_audio.grad)
    assert merged.sb_filter.cache_bytes() == 0
