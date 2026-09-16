import pytest
import torch

from dsp2 import SubbandFilter, mel_linear_banks_triangle
from dsp2.inference_optimized import InferenceSubbandFilter


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")


def filters(dtype=torch.float32):
    source = SubbandFilter(1600, mel_linear_banks_triangle(1600, 4, 200), 4,
                           chunk_size=3, bandpass_filter_len=32, upsample_filter_len=16).to("mps", dtype).eval()
    return source, InferenceSubbandFilter(source).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cached_filterbank_is_exact_and_bounded(dtype):
    source, cached = filters(dtype)
    assert source.state_dict().keys() == cached.state_dict().keys()
    cache_sizes = []
    with torch.inference_mode():
        for length in (128, 160, 128):
            audio = torch.randn(2, 2, length, device="mps")
            expected_bands = source.analysis(audio)
            actual_bands = cached.analysis(audio)
            torch.testing.assert_close(actual_bands, expected_bands, atol=0, rtol=0)
            torch.testing.assert_close(cached.synthesis(actual_bands), source.synthesis(expected_bands), atol=0, rtol=0)
            cache_sizes.append(cached.cache_bytes())
    assert cache_sizes[0] > 0 and cache_sizes[0] == cache_sizes[2]
    assert not cached._modulations


@pytest.mark.parametrize("change", ["weights", "reload", "chunk_size", "frequencies", "upsampling"])
def test_cached_filterbank_invalidates_after_changes(change):
    source, cached = filters()
    audio = torch.randn(1, 2, 128, device="mps")
    with torch.inference_mode():
        cached.synthesis(cached.analysis(audio))
        if change == "weights":
            source.w.mul_(0.9)
        elif change == "upsampling":
            source.up.mul_(0.9)
        elif change == "frequencies":
            source.f_center.mul_(0.9)
        elif change == "chunk_size":
            source.chunk_size = cached.chunk_size = 1
        else:
            cached.load_state_dict(source.state_dict(), strict=True)
            assert cached.cache_bytes() == 0
        torch.testing.assert_close(cached.synthesis(cached.analysis(audio)),
                                   source.synthesis(source.analysis(audio)), atol=0, rtol=0)
    cached.to("cpu")
    assert cached.cache_bytes() == 0


def test_training_and_gradient_paths_still_use_dsp2():
    source, cached = filters()
    audio = torch.randn(1, 2, 128, device="mps", requires_grad=True)
    reference_audio = audio.detach().clone().requires_grad_()
    with torch.inference_mode():
        cached.analysis(audio)
    assert cached.cache_bytes() > 0
    cached.train()
    assert cached.cache_bytes() == 0
    actual = cached.synthesis(cached.analysis(audio))
    expected = source.synthesis(source.analysis(reference_audio))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(audio.grad, reference_audio.grad, atol=1e-6, rtol=1e-5)
    assert cached.cache_bytes() == 0
    cached.eval()
    cached.analysis(audio)
    assert cached.cache_bytes() == 0


def test_inference_created_buffers_use_uncached_dsp2():
    with torch.inference_mode():
        source, cached = filters()
        audio = torch.randn(1, 2, 128, device="mps")
        torch.testing.assert_close(cached.synthesis(cached.analysis(audio)),
                                   source.synthesis(source.analysis(audio)), atol=0, rtol=0)
        assert cached.cache_bytes() == 0


def test_cpu_path_retains_uncached_dsp2():
    source, cached = filters()
    source.cpu()
    cached.cpu()
    with torch.inference_mode():
        audio = torch.randn(1, 2, 128)
        torch.testing.assert_close(cached.synthesis(cached.analysis(audio)),
                                   source.synthesis(source.analysis(audio)), atol=0, rtol=0)
        assert cached.cache_bytes() == 0
