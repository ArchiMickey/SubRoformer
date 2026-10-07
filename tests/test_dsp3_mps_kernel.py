import pytest
import torch

from subroformer.frontend.dsp3.mps_kernel import is_available, shift_frequency_mps
from subroformer.frontend.dsp3.utils import _phase_steps, _shift_frequency_reference, shift_frequency


pytestmark = pytest.mark.skipif(not is_available(), reason="MPS and shader compilation required")


@pytest.mark.parametrize("shape", [(7, 1), (2, 7, 1031), (2, 3, 7, 8193), (7, 150003)])
@pytest.mark.parametrize("stride", [0, 1, -3, (1 << 32) + 13, -(1 << 40) + 123])
@pytest.mark.parametrize("fused", [False, True])
def test_mps_kernel_matches_original(shape, stride, fused):
    torch.manual_seed(173)
    source = torch.randn(shape, dtype=torch.complex64, device="mps")
    frequencies = torch.tensor([0, 1e-5, -93.12345, 11025, -22050, 22050, 50000.123], device="mps")
    expected = _shift_frequency_reference(source, frequencies, 44100, stride)
    values = frequencies if fused else _phase_steps(source, frequencies, 44100)
    actual = shift_frequency_mps(source, values, stride, sample_rate=44100 if fused else None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mps_kernel_conjugates_original_oscillator():
    source = torch.randn((2, 3, 4099), dtype=torch.complex64, device="mps")
    frequencies = torch.tensor([-22050, 93.12345, 22049], device="mps")
    oscillator = _shift_frequency_reference(torch.ones_like(source), frequencies, 44100, 7)
    actual = shift_frequency_mps(source, frequencies, 7, True, sample_rate=44100)
    torch.testing.assert_close(actual, source * oscillator.conj(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 3, 0), (2, 0, 100), (0, 3, 100)])
def test_mps_kernel_empty(shape):
    source = torch.empty(shape, dtype=torch.complex64, device="mps")
    steps = torch.zeros(shape[-2], dtype=torch.int64, device="mps")
    torch.testing.assert_close(shift_frequency_mps(source, steps), source)


def test_mps_kernel_validates_inputs():
    source = torch.randn((2, 3, 17), dtype=torch.complex64, device="mps")
    steps = torch.zeros(3, dtype=torch.int64, device="mps")
    with pytest.raises(ValueError, match="complex64 MPS"):
        shift_frequency_mps(source.cpu(), steps)
    with pytest.raises(ValueError, match="contiguous input"):
        shift_frequency_mps(source.transpose(-1, -2), steps)
    with pytest.raises(ValueError, match="phase steps"):
        shift_frequency_mps(source, steps[:2])
    with pytest.raises(ValueError, match="phase steps"):
        shift_frequency_mps(source, steps, sample_rate=48000)
    with pytest.raises(ValueError, match="positive integer"):
        shift_frequency_mps(source, steps, sample_rate=0)


@pytest.mark.parametrize("sample_rate", [1601, 44100, 48000])
def test_mps_fused_frequency_quantization(sample_rate):
    frequencies = torch.tensor([
        0.0, 0.5 / (1 << 32), -1.5 / (1 << 32),
        0.5 * sample_rate / (1 << 32), -1.5 * sample_rate / (1 << 32),
        0.5 * sample_rate, -0.5 * sample_rate, 1.25 * sample_rate,
        -97.123456789, 12345.678901234,
    ], device="mps")
    source = torch.randn(2, frequencies.numel(), 4099, device="mps", dtype=torch.complex64)
    expected = _shift_frequency_reference(source, frequencies, sample_rate, (1 << 32) - 13)
    actual = shift_frequency_mps(source, frequencies, (1 << 32) - 13, sample_rate=sample_rate)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mps_dispatch_second_derivatives():
    signal = torch.randn(2, 3, 19, dtype=torch.complex64, device="mps", requires_grad=True)
    reference_signal = signal.detach().clone().requires_grad_()
    frequencies = torch.tensor([0.0, -800.0, 13.57], device="mps")
    upstream = torch.randn_like(signal)

    def derivatives(operation, source):
        output = operation(source, frequencies, 1600, 7)
        gradient, = torch.autograd.grad(output.abs().pow(4).sum(), source, create_graph=True)
        second, = torch.autograd.grad(gradient, source, upstream)
        return gradient, second

    actual = derivatives(shift_frequency, signal)
    expected = derivatives(_shift_frequency_reference, reference_signal)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_mps_dispatch_cpu_frequencies_and_broadcast_fallback():
    signal = torch.randn(2, 3, 17, device="mps", dtype=torch.complex64)
    frequencies = torch.tensor([0.0, 11.5, -700.0], dtype=torch.float64)
    torch.testing.assert_close(
        shift_frequency(signal, frequencies, 1600),
        _shift_frequency_reference(signal, frequencies, 1600),
        atol=0, rtol=0,
    )
    frequencies = frequencies[:1]
    torch.testing.assert_close(
        shift_frequency(signal, frequencies, 1600),
        _shift_frequency_reference(signal, frequencies, 1600),
        atol=0, rtol=0,
    )


def test_mps_frequencies_remain_nonlearnable():
    signal = torch.randn(2, 3, 17, device="mps", dtype=torch.complex64, requires_grad=True)
    frequencies = torch.tensor([0.0, 11.5, -700.0], device="mps", requires_grad=True)
    shift_frequency(signal, frequencies, 1600).real.sum().backward()
    assert signal.grad is not None and torch.isfinite(signal.grad).all()
    assert frequencies.grad is None


def test_mps_shader_errors_are_not_silently_suppressed(monkeypatch):
    import subroformer.frontend.dsp3.mps_kernel as kernel_module

    def fail():
        raise RuntimeError("shader compilation failed")

    monkeypatch.setattr(kernel_module, "_load_library", fail)
    signal = torch.randn(2, 3, 17, device="mps", dtype=torch.complex64)
    frequencies = torch.zeros(3, device="mps")
    with pytest.raises(RuntimeError, match="shader compilation failed"):
        shift_frequency(signal, frequencies, 1600)
