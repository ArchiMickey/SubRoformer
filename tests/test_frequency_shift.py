import math

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import subroformer.frontend.subband.dsp2.utils as dsp_utils
from subroformer.frontend.subband.dsp2 import SubbandFilter, mel_linear_banks_triangle
from subroformer.frontend.subband.dsp2.utils import shift_frequency


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


def _reference_oscillator(frequencies, length, sample_rate, stride):
    phase_scale = 1 << 32
    phase_steps = [round(float(frequency) / sample_rate * phase_scale) for frequency in frequencies]
    phase_words = [
        [
            ((phase_step * sample_index * stride + phase_scale // 2) % phase_scale)
            - phase_scale // 2
            for sample_index in range(length)
        ]
        for phase_step in phase_steps
    ]
    phase = torch.tensor(phase_words, dtype=torch.float64) * (2 * math.pi / phase_scale)
    return torch.polar(torch.ones_like(phase), phase)


@pytest.mark.parametrize("stride", [1, 60, (1 << 32) + 60])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_frequency_shift_matches_fixed_point_reference_across_tiles(monkeypatch, device, stride, dtype):
    monkeypatch.setattr(dsp_utils, "_PHASE_CHUNK_ELEMENTS", 128)
    frequencies = torch.tensor([0.0, 10.37, -997.125, 12400.166265, 23980.0, 24000.0])
    signal = torch.ones(2, 3, frequencies.numel(), 257, dtype=dtype, device=device)

    actual = shift_frequency(signal, frequencies.to(device), 48000, stride)
    expected = _reference_oscillator(frequencies, 257, 48000, stride).to(device=device, dtype=dtype)

    assert actual.dtype == dtype
    assert actual.shape == signal.shape
    tolerance = 4e-7 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(actual, expected.expand_as(actual), rtol=0, atol=tolerance)


def test_frequency_shift_remains_accurate_after_thirty_seconds(device):
    frequencies = torch.tensor([10.37, -997.125, 12400.166265, 23980.0])
    signal = torch.ones(1, frequencies.numel(), 24001, dtype=torch.complex64, device=device)

    actual = shift_frequency(signal, frequencies.to(device), 48000, 60)
    expected = _reference_oscillator(frequencies, signal.shape[-1], 48000, 60)

    torch.testing.assert_close(actual[0], expected.to(device=device, dtype=torch.complex64), rtol=0, atol=4e-7)


def test_frequency_shift_analysis_and_synthesis_share_sample_phases(device):
    frequencies = torch.tensor([10.37, 12400.166265, 23980.0], device=device)
    signal = torch.ones(1, frequencies.numel(), 96000, dtype=torch.complex64, device=device)

    synthesis = shift_frequency(signal, frequencies, 48000)
    analysis = shift_frequency(signal[..., ::60], -frequencies, 48000, 60)

    torch.testing.assert_close(analysis, synthesis[..., ::60].conj(), rtol=0, atol=4e-7)


def test_frequency_shift_preserves_signal_gradients(monkeypatch, device):
    monkeypatch.setattr(dsp_utils, "_PHASE_CHUNK_ELEMENTS", 128)
    frequencies = torch.tensor([10.37, -23980.0])
    signal = torch.randn(2, 2, 257, dtype=torch.complex64, device=device, requires_grad=True)
    upstream = torch.randn_like(signal)
    oscillator = _reference_oscillator(frequencies, 257, 48000, 60).to(device=device, dtype=signal.dtype)

    actual = shift_frequency(signal, frequencies.to(device), 48000, 60)
    actual.backward(upstream)

    torch.testing.assert_close(signal.grad, upstream * oscillator.conj(), rtol=1e-6, atol=1e-6)


def test_frequency_shift_uses_bounded_integer_tiles_without_fp64(monkeypatch, device):
    monkeypatch.setattr(dsp_utils, "_PHASE_CHUNK_ELEMENTS", 128)
    frequencies = torch.tensor([10.37, -23980.0], device=device)
    signal = torch.ones(1, 2, 257, dtype=torch.complex64, device=device)
    intermediates = []

    class RecordTensors(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            for tensor in tree_flatten(result)[0]:
                if isinstance(tensor, torch.Tensor):
                    intermediates.append((tensor.dtype, tensor.ndim, tensor.numel()))
            return result

    with RecordTensors():
        actual = dsp_utils._shift_frequency_reference(signal, frequencies, 48000, 60)

    assert actual.dtype == torch.complex64
    assert all(dtype not in (torch.float64, torch.complex128) for dtype, _, _ in intermediates)
    phase_tiles = [size for dtype, dimensions, size in intermediates if dtype == torch.int64 and dimensions == 2]
    assert phase_tiles
    assert max(phase_tiles) <= 128


def test_mel_filterbank_reconstruction_exceeds_seventy_db_away_from_boundaries(device):
    banks = mel_linear_banks_triangle(48000, 118, 390)
    filterbank = SubbandFilter(48000, banks, 60, chunk_size=16).to(device)
    reference = torch.randn(1, 1, 96000, generator=torch.Generator().manual_seed(0)).to(device)

    with torch.inference_mode():
        reconstructed = filterbank.synthesis(filterbank.analysis(reference))

    reference = reference[..., 30000:-30000].double()
    reconstructed = reconstructed[..., 30000:-30000].double()
    sdr = 10 * torch.log10(reference.square().sum() / (reference - reconstructed).square().sum())
    assert float(sdr) > 70


def test_dsp2_preserves_filter_state_and_forward_backward(monkeypatch, device):
    import subroformer.frontend.subband.dsp.subband_fast_triangle as original_module
    import subroformer.frontend.subband.dsp.utils as original_utils

    assert original_module.shift_frequency is original_utils.shift_frequency
    assert original_utils.shift_frequency is not shift_frequency
    banks = mel_linear_banks_triangle(1600, 4, 200)
    original = original_module.SubbandFilter(
        1600, banks, 4, chunk_size=3, bandpass_filter_len=32, upsample_filter_len=16,
    ).to(device)
    modified = SubbandFilter(
        1600, banks, 4, chunk_size=3, bandpass_filter_len=32, upsample_filter_len=16,
    ).to(device)
    assert modified.state_dict().keys() == original.state_dict().keys()
    modified.load_state_dict(original.state_dict(), strict=True)
    signal = torch.randn(2, 2, 128, device=device, requires_grad=True)
    reference_signal = signal.detach().clone().requires_grad_(True)

    actual_bands = modified.analysis(signal)
    actual = modified.synthesis(actual_bands)
    monkeypatch.setattr(original_module, "shift_frequency", shift_frequency)
    expected_bands = original.analysis(reference_signal)
    expected = original.synthesis(expected_bands)

    torch.testing.assert_close(actual_bands, expected_bands)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    assert torch.isfinite(signal.grad).all()
    torch.testing.assert_close(signal.grad, reference_signal.grad)
