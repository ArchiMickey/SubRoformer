import importlib
from unittest.mock import patch

import pytest
import torch

from dsp import SubbandFilter as OriginalSubbandFilter
from dsp2 import SubbandFilter as ReferenceSubbandFilter
from dsp2.utils import _shift_frequency_reference as reference_shift
from dsp3 import SubbandFilter, mel_linear_banks_triangle, shift_frequency
from dsp3.utils import _shift_frequency_reference
from model import SBRoformer
from inference_optimized import optimize_for_inference


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    return torch.device(request.param)


@pytest.fixture
def native_device(device):
    kernel = importlib.import_module(f"dsp3.{device.type}_kernel")
    if not kernel.is_available():
        pytest.skip(f"The native {device.type} kernel is unavailable")
    return device


def tiny_kwargs():
    return dict(
        sample_rate=1600, subband_n_bands=4, max_half_bandwidth=200,
        subband_factor=4, bandpass_filter_len=32, upsample_filter_len=16,
        n_fft=8, stft_hop_length=2, dim=32, dim_sp=32, dim_head=8,
        n_layers=1, patch_size=[2, 1], analysis_type="stft",
        attn_backend="torch", chunk_size=3,
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("stride", [1, 60, (1 << 32) + 60])
@pytest.mark.parametrize("layout", ["contiguous", "rows", "samples", "four_dimensional", "conjugate", "empty"])
def test_native_modulation_matches_dsp2_outputs_and_gradients(native_device, dtype, stride, layout):
    if native_device.type == "mps" and dtype == torch.complex128:
        pytest.skip("MPS does not support complex128")
    frequencies = torch.tensor([0.0, 10.37, -997.125, 12400.166265, 23980.0, 24000.0], device=native_device)
    signal = torch.randn(2, 12, 257, dtype=dtype, device=native_device)
    if layout == "rows":
        signal = signal[:, ::2]
    else:
        signal = signal[:, :6].contiguous()
    if layout == "samples":
        signal = signal[..., ::2]
    elif layout == "four_dimensional":
        signal = signal.unsqueeze(0).expand(3, -1, -1, -1)
    elif layout == "conjugate":
        signal = signal.conj()
    elif layout == "empty":
        signal = signal[..., :0]
    signal.requires_grad_(True)
    expected = reference_shift(signal, frequencies, 48000, stride)
    with patch("dsp3.utils._shift_frequency_reference", side_effect=AssertionError("Unexpected fallback")):
        actual = shift_frequency(signal, frequencies, 48000, stride)
    tolerance = 1e-6 if dtype == torch.complex64 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.shape == signal.shape
    assert actual.dtype == signal.dtype
    upstream = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, signal, upstream)[0]
    expected_gradient = torch.autograd.grad(expected, signal, upstream)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)


def test_native_modulation_supports_second_derivatives(native_device):
    if native_device.type == "mps":
        pytest.skip("Double-precision gradcheck is unavailable on MPS")
    signal = torch.randn(1, 2, 7, dtype=torch.complex128, device=native_device, requires_grad=True)
    frequencies = torch.tensor([123.45, -987.125], device=native_device, dtype=torch.float64)
    function = lambda value: shift_frequency(value, frequencies, 48000, 60)
    assert torch.autograd.gradcheck(function, (signal,))
    assert torch.autograd.gradgradcheck(function, (signal,))


def test_missing_native_kernel_falls_back_to_fixed_point_reference(monkeypatch, device):
    kernel = importlib.import_module(f"dsp3.{device.type}_kernel")
    monkeypatch.setattr(kernel, "is_available", lambda: False)
    frequencies = torch.tensor([997.125, -23980.0], device=device)
    signal = torch.randn(2, 2, 257, dtype=torch.complex64, device=device, requires_grad=True)
    actual = shift_frequency(signal, frequencies, 48000, 60)
    expected = reference_shift(signal, frequencies, 48000, 60)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.abs().square().mean().backward()
    assert torch.isfinite(signal.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_real_and_broadcast_inputs_use_reference(device, dtype):
    frequencies = torch.tensor([997.125, -23980.0], device=device)
    signal = torch.randn(2, 1, 257, dtype=dtype, device=device, requires_grad=True)
    with patch("dsp3.utils._shift_frequency_reference", wraps=_shift_frequency_reference) as fallback:
        actual = shift_frequency(signal, frequencies, 48000, 60)
    fallback.assert_called_once()
    expected = reference_shift(signal, frequencies, 48000, 60)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_filterbank_preserves_state_outputs_and_signal_gradients(device):
    banks = mel_linear_banks_triangle(1600, 4, 200)
    kwargs = dict(sr=1600, banks=banks, factor=4, chunk_size=3,
                  bandpass_filter_len=32, upsample_filter_len=16)
    original = OriginalSubbandFilter(**kwargs).to(device)
    reference = ReferenceSubbandFilter(**kwargs).to(device)
    fused = SubbandFilter(**kwargs).to(device)
    assert original.state_dict().keys() == fused.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, fused.state_dict()[key], rtol=0, atol=0)
    fused.load_state_dict(original.state_dict(), strict=True)
    reference.load_state_dict(original.state_dict(), strict=True)
    signal = torch.randn(2, 2, 128, device=device, requires_grad=True)
    actual_bands = fused.analysis(signal)
    expected_bands = reference.analysis(signal)
    actual = fused.synthesis(actual_bands)
    expected = reference.synthesis(expected_bands)
    torch.testing.assert_close(actual_bands, expected_bands, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, signal, upstream)[0]
    expected_gradient = torch.autograd.grad(expected, signal, upstream)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6)


def test_long_audio_modulation_matches_dsp2(native_device):
    frequencies = torch.tensor([10.37, -997.125, 12400.166265, 23980.0], device=native_device)
    signal = torch.ones(1, 4, 24001, dtype=torch.complex64, device=native_device)
    torch.testing.assert_close(
        shift_frequency(signal, frequencies, 48000, 60),
        reference_shift(signal, frequencies, 48000, 60), rtol=0, atol=4e-7,
    )


@pytest.mark.parametrize("length", [96000, 1440000])
def test_two_and_thirty_second_filterbanks_match_dsp2(device, length):
    banks = mel_linear_banks_triangle(48000, 118, 390)
    reference = ReferenceSubbandFilter(48000, banks, 60, chunk_size=16).to(device)
    fused = SubbandFilter(48000, banks, 60, chunk_size=16).to(device)
    fused.load_state_dict(reference.state_dict(), strict=True)
    signal = torch.randn(1, 2, length, device=device)
    with torch.inference_mode():
        expected_bands = reference.analysis(signal)
        actual_bands = fused.analysis(signal)
        torch.testing.assert_close(actual_bands, expected_bands, rtol=0, atol=0)
        expected = reference.synthesis(expected_bands)
        actual = fused.synthesis(actual_bands)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reconstruction_exceeds_seventy_db_away_from_boundaries(device):
    filterbank = SubbandFilter(48000, mel_linear_banks_triangle(48000, 118, 390), 60, chunk_size=16).to(device)
    signal = torch.randn(1, 1, 96000, generator=torch.Generator().manual_seed(0)).to(device)
    with torch.no_grad():
        reconstructed = filterbank.synthesis(filterbank.analysis(signal))
    signal = signal[..., 30000:-30000].cpu().double()
    reconstructed = reconstructed[..., 30000:-30000].cpu().double()
    sdr = 10 * torch.log10(signal.square().sum() / (signal - reconstructed).square().sum())
    assert float(sdr) > 70


@pytest.mark.parametrize("backend", ["dsp", "dsp2"])
def test_models_strict_load_existing_backends(backend, device, monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    original = SBRoformer(dsp_backend=backend, num_stems=4, **tiny_kwargs()).to(device).eval()
    fused = SBRoformer(dsp_backend="dsp3", num_stems=4, **tiny_kwargs()).to(device).eval()
    fused.load_state_dict(original.state_dict(), strict=True)
    original.load_state_dict(fused.state_dict(), strict=True)
    assert sum(parameter.numel() for parameter in original.parameters()) == sum(
        parameter.numel() for parameter in fused.parameters()
    )
    signal = torch.randn(1, 2, 133, device=device)
    with torch.no_grad():
        actual = fused(signal)
        expected = original(signal)
    assert actual.shape == (1, 4, 2, 133)
    tolerance = 1e-4 if backend == "dsp" else 1e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("use_bf16", [False, True])
@pytest.mark.parametrize("num_stems", [1, 4])
def test_model_forward_backward_optimizer_step(device, use_bf16, num_stems):
    if device.type == "mps" and use_bf16:
        pytest.skip("BF16 autocast validation targets CPU and CUDA")
    model = SBRoformer(dsp_backend="dsp3", num_stems=num_stems, **tiny_kwargs()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    signal = torch.randn(1, 2, 133, device=device, requires_grad=True)
    before = model.down.weight.detach().clone()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        output = model(signal)
        loss = output.square().mean()
    assert output.shape == ((1, 2, 133) if num_stems == 1 else (1, 4, 2, 133))
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(signal.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()
    assert not torch.equal(before, model.down.weight)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("frontend", ["global_stft_bs", "global_stft_mel"])
def test_dsp3_rejects_non_waveform_frontends(frontend):
    with pytest.raises(ValueError, match="dsp_backend='dsp3'"):
        SBRoformer(dsp_backend="dsp3", frontend_type=frontend, **tiny_kwargs())


def test_legacy_inference_optimizations_do_not_replace_dsp3():
    model = SBRoformer(dsp_backend="dsp3", **tiny_kwargs())
    with pytest.raises(ValueError, match="use dsp3 directly"):
        optimize_for_inference(model)
