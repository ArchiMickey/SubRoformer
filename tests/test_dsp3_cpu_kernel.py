import math

import pytest
import torch

from subroformer.frontend.subband.dsp3.cpu_kernel import is_available, shift_frequency_cpu
from subroformer.frontend.subband.dsp3.utils import _phase_steps, _shift_frequency_reference


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("shape", [(7, 1), (2, 7, 513), (2, 3, 7, 4097)])
@pytest.mark.parametrize("stride", [0, 1, -1, 257, (1 << 32) - 1])
def test_cpu_kernel_matches_reference(dtype, shape, stride):
    generator = torch.Generator().manual_seed(123)
    inputs = torch.randn(shape, generator=generator, dtype=dtype)
    frequencies = torch.tensor(
        [0, 0.125, -0.125, 11025, -22050, 44099.75, -44100.125],
        dtype=torch.float64,
    )
    expected = _shift_frequency_reference(inputs, frequencies, 44100, stride)
    actual = shift_frequency_cpu(inputs, _phase_steps(inputs, frequencies, 44100), stride)
    tolerance = 3e-7 if dtype == torch.complex64 else 1e-15
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("conjugate", [False, True])
def test_cpu_kernel_signed_phase_boundaries_and_large_stride(dtype, conjugate):
    steps = torch.tensor([-(1 << 31), -(1 << 31) + 1, -1, 0, 1, (1 << 31) - 1])
    inputs = torch.randn(2, steps.numel(), 71, dtype=dtype)
    stride = -(1 << 90) + 129
    indices = torch.arange(inputs.shape[-1], dtype=torch.int64) * (stride % (1 << 32))
    words = (steps[:, None] * indices + (1 << 31)) % (1 << 32) - (1 << 31)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    phase = words.to(real_dtype) * (2 * math.pi / (1 << 32))
    oscillator = torch.polar(torch.ones_like(phase), phase)
    expected = inputs * (oscillator.conj() if conjugate else oscillator)
    actual = shift_frequency_cpu(inputs, steps, stride, conjugate=conjugate)
    tolerance = 3e-7 if dtype == torch.complex64 else 1e-15
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("shape", [(2, 3, 0), (2, 0, 17), (0, 3, 17)])
def test_cpu_kernel_empty_tensors(shape):
    inputs = torch.empty(shape, dtype=torch.complex64)
    result = shift_frequency_cpu(inputs, torch.zeros(shape[-2], dtype=torch.int64))
    assert result.shape == inputs.shape
    assert result.dtype == inputs.dtype


def test_cpu_kernel_rejects_invalid_contract():
    with pytest.raises(RuntimeError, match="contiguous"):
        shift_frequency_cpu(torch.ones(3, 7, dtype=torch.complex64).T, torch.zeros(7, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="band dimension"):
        shift_frequency_cpu(torch.ones(3, 7, dtype=torch.complex64), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="int64"):
        shift_frequency_cpu(torch.ones(3, 7, dtype=torch.complex64), torch.zeros(3))


def test_cpu_kernel_available():
    assert is_available()


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_cpu_vector_tiles_and_threading(threads, dtype):
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        inputs = torch.randn(3, 7, 4099, dtype=dtype)
        frequencies = torch.linspace(-23980, 24000, 7)
        expected = _shift_frequency_reference(inputs, frequencies, 48000, 60)
        actual = shift_frequency_cpu(inputs, _phase_steps(inputs, frequencies, 48000), 60)
        tolerance = 3e-7 if dtype == torch.complex64 else 1e-15
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    finally:
        torch.set_num_threads(previous)
