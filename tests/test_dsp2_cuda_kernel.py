import pytest
import torch

from subroformer.frontend.dsp2.cuda_kernel import is_available, shift_frequency_cuda
from subroformer.frontend.dsp2 import utils


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(), reason="CUDA and Triton required"
)


def _phase_steps(frequencies, sample_rate, dtype):
    phase_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    frequencies = frequencies.to(dtype=phase_dtype)
    scaled = frequencies.abs().mul(1 << 32).round().to(torch.int64)
    steps = torch.div(scaled + sample_rate // 2, sample_rate, rounding_mode="floor")
    steps = steps * frequencies.sign().to(torch.int64)
    return ((steps + (1 << 31)) & ((1 << 32) - 1)) - (1 << 31)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", [(7, 1), (2, 7, 1031), (2, 3, 7, 8193), (7, 150003)])
@pytest.mark.parametrize("stride", [0, 1, -3, (1 << 32) + 13, -(1 << 40) + 123])
def test_cuda_kernel_matches_original(dtype, shape, stride):
    torch.manual_seed(173)
    source = torch.randn(shape, dtype=dtype, device="cuda")
    frequencies = torch.tensor(
        [0, 1e-5, -93.12345, 11025, -22050, 22050, 50000.123],
        dtype=torch.float64,
        device="cuda",
    )
    expected = utils._shift_frequency_reference(source, frequencies, 44100, stride)
    actual = shift_frequency_cuda(source, _phase_steps(frequencies, 44100, dtype), stride)
    tolerance = 3e-7 if dtype == torch.complex64 else 1e-14
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_cuda_kernel_conjugates_original_oscillator(dtype):
    source = torch.randn((2, 3, 4099), dtype=dtype, device="cuda")
    frequencies = torch.tensor([-22050, 93.12345, 22049], device="cuda")
    oscillator = utils._shift_frequency_reference(torch.ones_like(source), frequencies, 44100, 7)
    actual = shift_frequency_cuda(source, _phase_steps(frequencies, 44100, dtype), 7, True)
    tolerance = 3e-7 if dtype == torch.complex64 else 1e-14
    torch.testing.assert_close(actual, source * oscillator.conj(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("shape", [(2, 3, 0), (2, 0, 100), (0, 3, 100)])
def test_cuda_kernel_empty(shape):
    source = torch.empty(shape, dtype=torch.complex64, device="cuda")
    steps = torch.zeros(shape[-2], dtype=torch.int64, device="cuda")
    actual = shift_frequency_cuda(source, steps)
    torch.testing.assert_close(actual, source)


def test_cuda_kernel_validates_inputs():
    source = torch.randn((2, 3, 17), dtype=torch.complex64, device="cuda")
    steps = torch.zeros(3, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="contiguous input"):
        shift_frequency_cuda(source.transpose(-1, -2), steps)
    with pytest.raises(ValueError, match="phase steps"):
        shift_frequency_cuda(source, steps[:2])


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("sample_rate", [1601, 44100, 48000])
@pytest.mark.parametrize("frequency_dtype", [torch.float32, torch.float64])
def test_cuda_fused_frequency_quantization(dtype, sample_rate, frequency_dtype):
    frequencies = torch.tensor([
        0.0, 0.5 / (1 << 32), -1.5 / (1 << 32),
        0.5 * sample_rate / (1 << 32), -1.5 * sample_rate / (1 << 32),
        0.5 * sample_rate, -0.5 * sample_rate, 1.25 * sample_rate,
        -97.123456789, 12345.678901234,
    ], device="cuda", dtype=frequency_dtype)
    source = torch.randn(2, frequencies.numel(), 4099, device="cuda", dtype=dtype)
    expected = utils._shift_frequency_reference(source, frequencies, sample_rate, (1 << 32) - 13)
    actual = shift_frequency_cuda(source, frequencies, (1 << 32) - 13, sample_rate=sample_rate)
    tolerance = 3e-7 if dtype == torch.complex64 else 1e-14
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
