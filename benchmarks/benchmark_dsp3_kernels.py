"""Compare warmed DSP3 dispatch against the retained PyTorch implementation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch
from torch.utils.benchmark import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dsp3.subband_fast_triangle as filter_module
from dsp3 import SubbandFilter, mel_linear_banks_triangle
from dsp3.utils import _shift_frequency_reference, shift_frequency


def measure(operation, threads, duration, device):
    def timer():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    for _ in range(3):
        operation()
    timing = Timer(stmt="operation()", globals={"operation": operation}, num_threads=threads, timer=timer)
    return timing.blocked_autorange(min_run_time=duration).median * 1000


def compare(name, original, optimized, threads, duration):
    expected = original()
    actual = optimized()
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    maximum_error = (actual - expected).abs().max().item()
    relative_rms = ((actual - expected).abs().square().mean() / expected.abs().square().mean()).sqrt().item()
    timings = {"reference": [], "kernel": []}
    for repeat in range(3):
        operations = [("reference", original), ("kernel", optimized)]
        for label, operation in operations[::1 if repeat % 2 == 0 else -1]:
            timings[label].append(measure(operation, threads, duration, actual.device.type))
    baseline = sorted(timings["reference"])[1]
    native = sorted(timings["kernel"])[1]
    result = dict(name=name, reference_ms=baseline, kernel_ms=native, speedup=baseline / native,
                  max_absolute_error=maximum_error, relative_rms_error=relative_rms, trials_ms=timings)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--duration", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-speedup", action="store_true", help="Fail if any median speedup is at most one")
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    torch.manual_seed(173)
    device = arguments.device
    if device == "cpu":
        from dsp3.cpu_kernel import is_available
    elif device == "cuda":
        from dsp3.cuda_kernel import is_available
    else:
        from dsp3.mps_kernel import is_available
    if not is_available():
        raise RuntimeError(f"Native {device} kernel is unavailable")
    results = []
    with torch.inference_mode():
        frequencies = torch.linspace(-23980, 23980, 16, device=device)
        for length, stride, name in [(1600, 60, "analysis_chunk"), (96000, 1, "synthesis_chunk"),
                                     (1440000, 1, "synthesis_chunk_30s")]:
            signal = torch.randn(2, 16, length, device=device, dtype=torch.complex64)
            results.append(compare(name,
                lambda: _shift_frequency_reference(signal, frequencies, 48000, stride),
                lambda: shift_frequency(signal, frequencies, 48000, stride),
                arguments.threads, arguments.duration))
            del signal
        banks = mel_linear_banks_triangle(48000, 118, 390)
        bank = SubbandFilter(48000, banks, 60, chunk_size=16).to(device)
        audio = torch.randn(1, 2, 96000, device=device)

        def filterbank(modulation):
            previous = filter_module.shift_frequency
            try:
                filter_module.shift_frequency = modulation
                return bank.synthesis(bank.analysis(audio))
            finally:
                filter_module.shift_frequency = previous

        results.append(compare("filterbank_2s",
            lambda: filterbank(_shift_frequency_reference), lambda: filterbank(shift_frequency),
            arguments.threads, arguments.duration))
    report = dict(device=device, threads=arguments.threads, torch_version=torch.__version__,
                  warmup=3, min_run_time=arguments.duration,
                  processor=platform.processor(),
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None, results=results)
    if device == "mps":
        report["gpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        report["macos_version"] = platform.mac_ver()[0]
    report["cpu_affinity"] = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    root = Path(__file__).resolve().parents[1]
    paths = [root / "dsp3" / name for name in (
        "utils.py", "cpu_kernel.py", "cuda_kernel.py", "mps_kernel.py",
        "csrc/shift_frequency_cpu.cpp", "csrc/shift_frequency_mps.metal",
    )]
    report["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    if arguments.require_speedup and any(result["speedup"] <= 1 for result in results):
        raise SystemExit("At least one measured kernel path did not beat the reference")


if __name__ == "__main__":
    main()
