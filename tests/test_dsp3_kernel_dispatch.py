import pytest
import torch

import subroformer.frontend.subband.dsp3.subband_fast_triangle as filter_module
from subroformer.frontend.subband.dsp3 import SubbandFilter, mel_linear_banks_triangle
from subroformer.frontend.subband.dsp3.utils import _shift_frequency_reference, shift_frequency


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    return request.param


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("layout", ["contiguous", "sliced", "conjugate", "expanded", "padded", "transposed"])
def test_dispatch_forward_and_backward(device, dtype, layout):
    if device == "mps" and dtype == torch.complex128:
        pytest.skip("MPS does not support complex128")
    signal = torch.randn(2, 3, 514, device=device, dtype=dtype)
    if layout == "sliced":
        signal = signal[..., ::2]
    elif layout == "conjugate":
        signal = signal.conj()
    elif layout == "expanded":
        signal = signal[:1].expand(4, -1, -1)
    elif layout == "padded":
        signal = signal[..., 13:270]
    elif layout == "transposed":
        signal = torch.randn(3, 2, 257, device=device, dtype=dtype).transpose(0, 1)
    signal = signal.detach().requires_grad_()
    reference_signal = signal.detach().clone().requires_grad_()
    frequencies = torch.tensor([10.37, -23980.0, 24000.0], device=device)
    actual = shift_frequency(signal, frequencies, 48000, (1 << 32) + 60)
    expected = _shift_frequency_reference(reference_signal, frequencies, 48000, 60)
    tolerance = 1e-6 if dtype == torch.complex64 else 2e-12
    if device == "mps":
        tolerance = 0
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(signal.grad, reference_signal.grad, atol=tolerance, rtol=tolerance)


def test_dispatch_second_derivatives(device):
    if device == "mps":
        pytest.skip("MPS does not support double-precision gradcheck")
    signal = torch.randn(1, 2, 7, dtype=torch.complex128, device=device, requires_grad=True)
    frequencies = torch.tensor([13.5, -711.0], device=device)
    operation = lambda signal: shift_frequency(signal, frequencies, 1600, 4)
    assert torch.autograd.gradcheck(operation, (signal,))
    assert torch.autograd.gradgradcheck(operation, (signal,))


def test_dispatch_uses_native_kernel_and_falls_back(device, monkeypatch):
    import importlib

    module = importlib.import_module(f"subroformer.frontend.subband.dsp3.{device}_kernel")
    name = f"shift_frequency_{device}"
    native = getattr(module, name)
    calls = []

    def recording_kernel(*args, **kwargs):
        calls.append(args)
        return native(*args, **kwargs)

    assert module.is_available()
    monkeypatch.setattr(module, name, recording_kernel)
    signal = torch.randn(1, 2, 19, dtype=torch.complex64, device=device, requires_grad=True)
    frequencies = torch.tensor([0.0, -800.0], device=device)
    actual = shift_frequency(signal, frequencies, 1600)
    actual.real.sum().backward()
    assert len(calls) == 2
    assert calls[0][-1] is False and calls[1][-1] is True
    monkeypatch.setattr(module, "is_available", lambda: False)
    fallback = shift_frequency(signal, frequencies, 1600)
    assert len(calls) == 2
    torch.testing.assert_close(fallback, _shift_frequency_reference(signal, frequencies, 1600), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_input_fallback(device, dtype):
    if device == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")
    signal = torch.randn(2, 3, 17, device=device, dtype=dtype)
    frequencies = torch.tensor([0.0, 11.5, -700.0], device=device)
    torch.testing.assert_close(
        shift_frequency(signal, frequencies, 1600),
        _shift_frequency_reference(signal, frequencies, 1600),
        atol=0, rtol=0,
    )


def test_filterbank_matches_original_dsp3_forward_and_gradients(device, monkeypatch):
    banks = mel_linear_banks_triangle(1600, 4, 200)
    bank = SubbandFilter(1600, banks, 4, chunk_size=3, bandpass_filter_len=32, upsample_filter_len=16).to(device)
    signal = torch.randn(2, 2, 128, device=device, requires_grad=True)
    reference_signal = signal.detach().clone().requires_grad_()
    actual_bands = bank.analysis(signal)
    actual = bank.synthesis(actual_bands)
    monkeypatch.setattr(filter_module, "shift_frequency", _shift_frequency_reference)
    expected_bands = bank.analysis(reference_signal)
    expected = bank.synthesis(expected_bands)
    torch.testing.assert_close(actual_bands, expected_bands, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(signal.grad, reference_signal.grad, atol=2e-6, rtol=2e-6)
