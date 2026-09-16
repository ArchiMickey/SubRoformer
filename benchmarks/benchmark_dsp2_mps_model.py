"""Compare complete MPS model forwards with reference versus native DSP2."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_dsp2_kernels import measure
import dsp2.subband_fast_triangle as filter_module
from dsp2 import SubbandFilter
from dsp2.mps_kernel import is_available
from dsp2.utils import _shift_frequency_reference, shift_frequency
from inference import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_audio, load_separator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "coreml"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / DEFAULT_CHECKPOINT)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/coreml-perceptual-200k-fp16")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-speedup", action="store_true")
    arguments = parser.parse_args()
    if not is_available():
        raise RuntimeError("MPS shader compilation is unavailable")
    torch.set_num_threads(4)
    torch.manual_seed(173)
    if arguments.backend == "torch":
        model, config = load_separator(arguments.config, arguments.checkpoint, "mps",
                                       preserve_dtype=True, local_files_only=True)
        bank = model.sb_filter
        files = [arguments.config, arguments.checkpoint]
    else:
        from coreml_inference import load_local_coreml_separator

        model, config = load_local_coreml_separator(arguments.artifact, dsp_device="mps")
        bank = model.dsp.sb_filter
        files = [arguments.artifact / name for name in ("config.yaml", "dsp.safetensors")]
        files.extend(sorted(path for path in (arguments.artifact / "mask.mlmodelc").rglob("*") if path.is_file()))
    if not isinstance(bank, SubbandFilter):
        raise RuntimeError("The benchmark requires the DSP2 filterbank")
    uncached_bank = SubbandFilter(bank.sr, bank.banks, bank.factor, chunk_size=bank.chunk_size,
                                  bandpass_filter_len=bank.bandpass_filter_len,
                                  upsample_filter_len=bank.upsample_filter_len).eval()
    uncached_bank.load_state_dict(bank.state_dict(), strict=True, assign=True)
    parent = model if arguments.backend == "torch" else model.dsp
    operations = [("reference", _shift_frequency_reference, uncached_bank), ("kernel", shift_frequency, bank)]
    if arguments.backend == "coreml":
        from dsp.inference_optimized import InferenceSubbandFilter as LegacyFilter

        operations.append(("legacy", shift_frequency, LegacyFilter(uncached_bank, level=3).eval()))
    samples = int(config.audio.chunk_size)
    noise = torch.randn(1, 2, samples) * 0.05
    waveforms = {"noise": noise, "silence": torch.zeros_like(noise), "quiet": noise * 1e-4}
    if arguments.audio:
        audio = load_audio(arguments.audio, int(config.audio.sample_rate))
        if audio.shape[0] != 2 or audio.shape[-1] < samples:
            raise ValueError("Validation audio must contain at least one full stereo chunk")
        waveforms["music"] = torch.from_numpy(audio[:, :samples].copy())[None]
        files.append(arguments.audio)
    if arguments.backend == "torch":
        waveforms = {name: audio.to("mps") for name, audio in waveforms.items()}

    def forward(audio, modulation, selected_bank):
        previous = filter_module.shift_frequency
        previous_bank = parent.sb_filter
        try:
            filter_module.shift_frequency = modulation
            parent.sb_filter = selected_bank
            with torch.autocast("mps", dtype=torch.float16, enabled=arguments.backend == "torch"):
                return model(audio).cpu()
        finally:
            filter_module.shift_frequency = previous
            parent.sb_filter = previous_bank

    errors = {}
    timings = {label: [] for label, _, _ in operations}
    with torch.inference_mode():
        for name, audio in waveforms.items():
            expected = forward(audio, _shift_frequency_reference, uncached_bank)
            actual = forward(audio, shift_frequency, bank)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise RuntimeError(f"Invalid model output for {name}")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            difference = actual.double() - expected.double()
            absolute_rms = difference.square().mean(dim=(-1, -2)).sqrt()
            reference_rms = expected.double().square().mean(dim=(-1, -2)).sqrt()
            errors[name] = dict(bitwise_equal=bool(torch.equal(actual, expected)),
                                max_absolute_error=difference.abs().max().item(),
                                per_stem_absolute_rms=absolute_rms.tolist(),
                                per_stem_relative_rms=(absolute_rms / reference_rms.clamp_min(1e-8)).tolist())
        audio = waveforms.get("music", noise.to("mps") if arguments.backend == "torch" else noise)
        for repeat in range(3):
            for label, modulation, selected_bank in operations[::1 if repeat % 2 == 0 else -1]:
                timings[label].append(measure(lambda: forward(audio, modulation, selected_bank), 4, arguments.duration, "mps"))
    reference_ms = statistics.median(timings["reference"])
    kernel_ms = statistics.median(timings["kernel"])
    sources = [ROOT / name for name in (
        "dsp2/utils.py", "dsp2/mps_kernel.py", "dsp2/csrc/shift_frequency_mps.metal", "dsp2/inference_optimized.py",
        "dsp/inference_optimized.py",
        "coreml_inference.py", "model.py", "benchmarks/benchmark_dsp2_mps_model.py",
        "benchmarks/benchmark_dsp2_kernels.py",
    )]
    report = dict(backend=arguments.backend, device="mps", platform=platform.platform(),
                  hardware=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                  torch_version=torch.__version__, samples=samples, batch_size=1, threads=4,
                  warmup=3, min_run_time=arguments.duration, trials_ms=timings,
                  reference_ms=reference_ms, kernel_ms=kernel_ms,
                  speedup=reference_ms / kernel_ms, errors=errors,
                  reference_filterbank="uncached DSP2 reference", kernel_filterbank="DSP2 Metal with filter FFT cache",
                  filter_cache_bytes=bank.cache_bytes(),
                  timing_input="music" if arguments.audio else "noise", output_device="cpu",
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  input_sha256={str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name:
                                hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    if "legacy" in timings:
        report["legacy_ms"] = statistics.median(timings["legacy"])
        report["legacy_speedup"] = report["legacy_ms"] / kernel_ms
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if arguments.require_speedup and (report["speedup"] <= 1 or report.get("legacy_speedup", float("inf")) <= 1):
        raise SystemExit("The complete model did not beat the reference")


if __name__ == "__main__":
    main()
