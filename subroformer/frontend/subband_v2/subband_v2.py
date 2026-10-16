"""Padded complex FIR subbands with exact band counts and selectable precision.

The interior filters select positive frequencies directly. DC and Nyquist
use symmetric endpoint filters, so taking the real part of the synthesis
sum reconstructs both endpoints without a waveform Hilbert transform.
"""

import math

import numpy as np
import torch
from scipy.fft import next_fast_len
from scipy.signal import firwin, get_window
from torch import Tensor, nn
from torch.nn import functional as F

from subroformer.frontend.subband.dsp3.utils import shift_frequency


def mel_linear_centers(sr: int, bins: int, max_half_bandwidth: float) -> np.ndarray:
    """Return exactly ``bins`` centers, including DC and Nyquist.

    Scale mel-spaced intervals together, capping each at the requested
    width. Solve for the scale, rather than searching for a seed band count.
    An impossible count/width combination is rejected, never silently grown.
    """
    if isinstance(sr, bool) or not isinstance(sr, int) or sr < 2:
        raise ValueError("sr must be an integer >= 2")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2, including DC and Nyquist")
    if not math.isfinite(max_half_bandwidth) or max_half_bandwidth <= 0:
        raise ValueError("max_half_bandwidth must be positive and finite")
    nyquist = sr / 2
    if (bins - 1) * max_half_bandwidth < nyquist:
        raise ValueError("Too few bins to span Nyquist at max_half_bandwidth")
    if (bins - 1) * max_half_bandwidth == nyquist:
        return np.linspace(0, nyquist, bins)
    intervals = np.diff(700 * np.expm1(np.linspace(0, np.log1p(nyquist / 700), bins)))
    low, high = 0.0, 1.0
    while np.minimum(high * intervals, max_half_bandwidth).sum() < nyquist:
        high *= 2
    for _ in range(80):
        scale = (low + high) / 2
        if np.minimum(scale * intervals, max_half_bandwidth).sum() < nyquist:
            low = scale
        else:
            high = scale
    centers = np.r_[0.0, np.cumsum(np.minimum(high * intervals, max_half_bandwidth))]
    centers[-1] = nyquist
    return centers


def _complex_filters(sr, centers, length, window):
    """Sample triangular responses, IFFT once at construction, then window."""
    fft_length = 2 * (1 << (length - 1).bit_length())
    frequencies = np.fft.fftfreq(fft_length, 1 / sr)
    # Leading zero keeps the existing even-length polyphase alignment.
    offsets = np.arange(-(length // 2 - 1), length // 2)
    taper = get_window(window, length - 1, fftbins=False)
    filters = np.zeros((len(centers), length), dtype=np.complex128)
    for band in range(len(centers)):
        if band == 0:
            gain = np.maximum(1 - np.abs(frequencies) / centers[1], 0)
        elif band == len(centers) - 1:
            gain = np.maximum(1 - (sr / 2 - np.abs(frequencies)) / (sr / 2 - centers[-2]), 0)
        else:
            rising = (frequencies - centers[band - 1]) / (centers[band] - centers[band - 1])
            falling = (centers[band + 1] - frequencies) / (centers[band + 1] - centers[band])
            gain = 2 * np.maximum(0, np.minimum(rising, falling))
        filters[band, 1:] = np.fft.ifft(gain)[offsets % fft_length] * taper
    return torch.from_numpy(filters)


class SubbandFilterV2(nn.Module):
    """Complex FIR analysis/synthesis with an explicit ``bins`` count.

    ``analysis``: real [B,C,L] -> complex [B,C,bins,ceil(L/factor)+2*P/factor].
    ``synthesis(z, length=L)``: padded subbands -> real [B,C,L]. Omitting
    length returns the aligned length ceil(L/factor)*factor. The halo P is
    retained in the subbands until synthesis; removing it loses edge context.

    DSP uses the filterbank's dtype: float32/complex64 by default, or
    float64/complex128 with dtype=torch.float64 or module.double(). Autocast
    is disabled internally; half/bfloat16 module casts keep DSP at float32.
    Inputs are converted to the module precision. Filters are fixed, but both
    transforms support input gradients. V1 features are not interchangeable.
    """

    def __init__(self, sr: int = 16000, bins: int = 64, factor: int = 10,
                 max_half_bandwidth: float | None = None, chunk_size: int = 16,
                 bandpass_filter_len: int = 48000, upsample_filter_len: int = 12000,
                 padding: int | None = None, window: str = "kaiser", beta: float = 8.6,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("DSP dtype must be torch.float32 or torch.float64")
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        for name, value in (("factor", factor), ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("bandpass_filter_len", bandpass_filter_len),
                            ("upsample_filter_len", upsample_filter_len)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 4 or value % (2 * factor):
                raise ValueError(f"{name} must be >= 4 and divisible by 2*factor")
        if window not in {"kaiser", "hamming"}:
            raise ValueError("window must be 'kaiser' or 'hamming' (for ablation)")
        if not math.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if max_half_bandwidth is None:
            max_half_bandwidth = sr / (2 * factor) * 0.975
        centers = mel_linear_centers(sr, bins, max_half_bandwidth)
        # Endpoint filters straddle DC/Nyquist, so their full width is twice
        # the adjacent interval. Interior filters span two adjacent intervals.
        if max_half_bandwidth >= sr / (2 * factor):
            raise ValueError("max_half_bandwidth must leave a guard below sr/(2*factor)")
        if padding is None:
            padding = math.ceil((upsample_filter_len // 2) / factor) * factor
        if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0 or padding % factor:
            raise ValueError("padding must be a nonnegative multiple of factor")
        self.sr, self.bins, self.factor = sr, bins, factor
        self.chunk_size = chunk_size
        self.max_half_bandwidth = max_half_bandwidth
        self.bandpass_filter_len = bandpass_filter_len
        self.upsample_filter_len = upsample_filter_len
        self.padding, self.window, self.beta = padding, window, beta
        self.banks = ([centers[:2].tolist()] +
                      [centers[i - 1:i + 2].tolist() for i in range(1, bins - 1)] +
                      [centers[-2:].tolist()])
        design_window = ("kaiser", beta) if window == "kaiser" else window
        self.register_buffer("w", _complex_filters(sr, centers, bandpass_filter_len, design_window).to(complex_dtype))
        carriers = centers.copy()
        carriers[1:-1] = (centers[:-2] + centers[2:]) / 2
        self.register_buffer("f_center", torch.from_numpy(carriers).to(dtype))
        if factor == 1:
            up = np.zeros(upsample_filter_len)
            up[upsample_filter_len // 2] = 1
        else:
            up = np.r_[0.0, firwin(upsample_filter_len - 1, 1 / factor, window=design_window)]
        self.register_buffer("up", torch.from_numpy(up).to(dtype))

    def _apply(self, fn, recurse=True):
        # Convert real/complex buffers as a pair. Module.double() normally
        # ignores complex buffers, while to(float32) discards their imaginary
        # part. Keep FFT arithmetic at least FP32 under half/bfloat16 casts.
        target = fn(self.up.new_empty(0))
        real_dtype = torch.float64 if target.dtype in (torch.float64, torch.complex128) else torch.float32
        complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
        if recurse:
            for child in self.children():
                child._apply(fn)
        for name, buffer in self._buffers.items():
            dtype = complex_dtype if buffer.is_complex() else real_dtype
            self._buffers[name] = buffer.to(device=target.device, dtype=dtype)
        return self

    def get_extra_state(self):
        return {"version": 2, **{key: getattr(self, key) for key in (
            "sr", "bins", "factor", "max_half_bandwidth", "bandpass_filter_len",
            "upsample_filter_len", "padding", "window", "beta")}}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("V2 filter configuration differs from the checkpoint")

    def _check_device(self, x):
        if x.device != self.w.device:
            raise ValueError("Move the filterbank to the input device before calling it")

    def analysis(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or not x.is_floating_point() or min(x.shape) < 1:
            raise ValueError("Expected nonempty real floating-point audio [batch, channels, samples]")
        self._check_device(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._analysis(x, self.up.dtype, self.w.dtype)

    def _analysis(self, x, real_dtype, complex_dtype):
        batch, channels, length = x.shape
        padded = F.pad(x.to(real_dtype), (self.padding, self.padding + (-length) % self.factor))
        down_length = padded.shape[-1] // self.factor
        filter_frames = self.bandpass_filter_len // self.factor
        fft_length = next_fast_len(down_length + filter_frames - 1)
        phases = padded.reshape(batch * channels, down_length, self.factor).transpose(1, 2)
        spectrum = torch.fft.fft(phases, n=fft_length)
        chunks = []
        for start in range(0, self.bins, self.chunk_size):
            stop = min(start + self.chunk_size, self.bins)
            # Polyphase correlation becomes convolution by conjugating our
            # Hermitian FIR. This sign matters for positive-frequency selection.
            filters = self.w[start:stop].to(complex_dtype).conj()
            filters = filters.reshape(stop - start, filter_frames, self.factor).transpose(1, 2).flip(-1)
            response = torch.fft.fft(filters, n=fft_length)
            filtered = torch.fft.ifft(torch.einsum("bpn,kpn->bkn", spectrum, response), n=fft_length)
            offset = filter_frames // 2 - 1
            filtered = filtered[..., offset:offset + down_length]
            chunks.append(shift_frequency(filtered, -self.f_center[start:stop].to(real_dtype), self.sr, self.factor))
        return torch.cat(chunks, dim=1).reshape(batch, channels, self.bins, down_length)

    def synthesis(self, x: Tensor, length: int | None = None) -> Tensor:
        if x.ndim != 4 or not x.is_complex() or min(x.shape) < 1 or x.shape[2] != self.bins:
            raise ValueError("Expected complex subbands [batch, channels, bins, padded samples]")
        self._check_device(x)
        aligned_length = x.shape[-1] * self.factor - 2 * self.padding
        if length is None:
            length = aligned_length
        if (isinstance(length, bool) or not isinstance(length, int) or length < 1
                or not 0 <= aligned_length - length < self.factor):
            raise ValueError("length must match the unpadded analysis length, within alignment slack")
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._synthesis(x, length, self.up.dtype, self.w.dtype)

    def _synthesis(self, x, length, real_dtype, complex_dtype):
        batch, channels, _, down_length = x.shape
        subbands = x.to(complex_dtype).reshape(batch * channels, self.bins, down_length)
        full_length = down_length * self.factor
        filter_frames = self.upsample_filter_len // self.factor
        fft_length = next_fast_len(down_length + filter_frames - 1)
        up = self.up.to(real_dtype).reshape(filter_frames, self.factor).T
        response = torch.fft.fft(up, n=fft_length)
        output = torch.zeros(batch * channels, full_length, dtype=real_dtype, device=x.device)
        for start in range(0, self.bins, self.chunk_size):
            stop = min(start + self.chunk_size, self.bins)
            spectrum = torch.fft.fft(subbands[:, start:stop] * self.factor, n=fft_length)
            interpolated = torch.fft.ifft(spectrum[:, :, None] * response[None, None], n=fft_length)
            interpolated = interpolated.transpose(-1, -2).reshape(batch * channels, stop - start, -1)
            offset = self.upsample_filter_len // 2
            interpolated = interpolated[..., offset:offset + full_length]
            modulated = shift_frequency(interpolated, self.f_center[start:stop].to(real_dtype), self.sr)
            output = output + modulated.real.sum(dim=1)
        return output[..., self.padding:self.padding + length].reshape(batch, channels, length)

    def forward(self, x: Tensor) -> Tensor:
        return self.analysis(x)


def main(argv=None):
    """Compare true FP32/FP64 DSP on identical 10 s and 2 s white noise.

    From the repository root:
        .venv/bin/python -m subroformer.frontend.subband_v2.subband_v2

    Both precisions use the public API with matching filter-buffer dtypes.
    Add --bf16 to compare BF16 autocast and explicitly quantized BF16 inputs.
    """
    import argparse
    import json
    import statistics
    import time
    from pathlib import Path

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--factor", type=int, default=60)
    parser.add_argument("--durations", type=float, nargs="+", default=[10.0, 2.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bf16", action="store_true", help="Also test BF16 autocast and BF16 input quantization")
    parser.add_argument("--output", type=Path, help="Optional JSON result file")
    args = parser.parse_args(argv)
    if args.sr < 2 or any(not math.isfinite(s) or s <= 0 or round(s * args.sr) < 4 for s in args.durations):
        parser.error("sr must be >= 2 and durations must be finite, positive and contain >= 4 samples")
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(1, 1, round(max(args.durations) * args.sr), generator=generator, dtype=torch.float32)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def roundtrip(waveform, real_dtype, complex_dtype, autocast=False):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
            subbands = bank.analysis(waveform)
            restored = bank.synthesis(subbands, length=waveform.shape[-1])
        assert subbands.dtype == complex_dtype and restored.dtype == real_dtype
        assert restored.shape == waveform.shape
        return restored

    def snr(reference, error):
        energy = error.square().sum().clamp_min(torch.finfo(torch.float64).tiny)
        return (10 * torch.log10(reference.square().sum() / energy)).item()

    hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"{hardware}; {args.sr} Hz; {args.bins} bins; factor={args.factor}; seed={args.seed}", flush=True)
    print("Identical FP32 noise values; SNR includes every sample, including both edges.", flush=True)
    print("DSP arithmetic and filter buffers both use the selected precision.", flush=True)
    bf16_support = {}
    if args.bf16:
        probe = torch.ones(96, dtype=torch.bfloat16, device=device)
        for enabled in (False, True):
            key = f"fft_bf16_autocast_{enabled}"
            try:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled):
                    result = torch.fft.fft(probe)
                bf16_support[key] = {"output_dtype": str(result.dtype)}
            except RuntimeError as error:
                bf16_support[key] = {"error": str(error).splitlines()[0]}
        try:
            result = torch.complex(probe, probe)
            bf16_support["complex_bf16"] = {"output_dtype": str(result.dtype)}
        except RuntimeError as error:
            bf16_support["complex_bf16"] = {"error": str(error).splitlines()[0]}
        del probe
        print(f"Raw BF16 support: {bf16_support}", flush=True)
        print("BF16 cases keep DSP at FP32; SNR is relative to original FP32 noise.", flush=True)
    print(f"{'seconds':>7} {'case':>24} {'SNR dB':>10} {'edge dB':>10} {'RMSE':>12} "
          f"{'max error':>12} {'median ms':>11} {'GPU peak MiB':>13}", flush=True)
    cases = [("fp32", torch.float32, torch.complex64, torch.float32, False),
             ("fp64", torch.float64, torch.complex128, torch.float32, False)]
    if args.bf16:
        cases += [("bf16_autocast", torch.float32, torch.complex64, torch.float32, True),
                  ("bf16_input", torch.float32, torch.complex64, torch.bfloat16, False),
                  ("bf16_input_autocast", torch.float32, torch.complex64, torch.bfloat16, True)]
    rows = []
    with torch.inference_mode():
        for seconds in args.durations:
            length = round(seconds * args.sr)
            waveform = noise[..., :length].to(device)
            for label, real_dtype, complex_dtype, input_dtype, autocast in cases:
                bank = SubbandFilterV2(sr=args.sr, bins=args.bins, factor=args.factor, dtype=real_dtype).to(device)
                bank_config = bank.get_extra_state()
                test_input = waveform.to(input_dtype)
                for _ in range(args.warmup):
                    roundtrip(test_input, real_dtype, complex_dtype, autocast)
                synchronize()
                resident = None
                if device.type == "cuda":
                    resident = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                timings = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    restored = roundtrip(test_input, real_dtype, complex_dtype, autocast)
                    synchronize()
                    timings.append(1000 * (time.perf_counter() - started))
                    del restored
                peak = torch.cuda.max_memory_allocated(device) if resident is not None else None
                restored = roundtrip(test_input, real_dtype, complex_dtype, autocast)
                assert torch.isfinite(restored).all(), f"Non-finite {label} reconstruction"
                reference = waveform.double()
                error = restored.double() - reference
                edge = max(1, min(round(args.sr * 0.1), length // 4))
                row = {"seconds": seconds, "samples": length, "precision": label,
                       "real_dtype": str(real_dtype), "complex_dtype": str(complex_dtype),
                       "input_dtype": str(input_dtype), "bf16_autocast": autocast,
                       "snr_db": snr(reference, error),
                       "snr_vs_supplied_input_db": snr(test_input.double(), restored.double() - test_input.double()),
                       "edge_snr_db": snr(torch.cat((reference[..., :edge], reference[..., -edge:]), -1),
                                          torch.cat((error[..., :edge], error[..., -edge:]), -1)),
                       "rmse": error.square().mean().sqrt().item(), "max_abs_error": error.abs().max().item(),
                       "median_ms": statistics.median(timings), "timings_ms": timings,
                       "peak_allocated_mib": peak / 2**20 if peak is not None else None,
                       "incremental_peak_mib": (peak - resident) / 2**20 if peak is not None else None}
                rows.append(row)
                if args.bf16:
                    # Keep the comparison output on CPU so it does not inflate
                    # the next case's GPU peak allocation.
                    if label == "fp32":
                        fp32_output = restored.cpu()
                        with torch.autocast(device.type, dtype=torch.bfloat16):
                            subbands = bank._analysis(test_input, real_dtype, complex_dtype)
                            unguarded = bank._synthesis(subbands, length, real_dtype, complex_dtype)
                        row["without_guard_bf16_autocast_max_abs_error"] = (unguarded - restored).abs().max().item()
                        del subbands, unguarded
                    row["max_abs_difference_vs_fp32"] = (restored.cpu().double() - fp32_output.double()).abs().max().item()
                    if label == "bf16_autocast":
                        assert row["max_abs_difference_vs_fp32"] == 0
                peak_text = f"{row['peak_allocated_mib']:.1f}" if peak is not None else "n/a"
                print(f"{seconds:7g} {label:>24} {row['snr_db']:10.3f} {row['edge_snr_db']:10.3f} "
                      f"{row['rmse']:12.4e} {row['max_abs_error']:12.4e} {row['median_ms']:11.3f} {peak_text:>13}", flush=True)
                del restored, reference, error, bank, test_input
            del waveform
    if args.output is not None:
        report = {"complete": True, "config": bank_config, "device": str(device),
                  "hardware": hardware, "torch": str(torch.__version__), "seed": args.seed,
                  "durations": args.durations, "warmup": args.warmup, "repeats": args.repeats,
                  "reference_dtype": "torch.float32", "bf16_support": bf16_support, "results": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved: {args.output}", flush=True)
    return rows


if __name__ == "__main__":
    main()
