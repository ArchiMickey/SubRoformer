from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]

from subroformer.inference.coreml import CoreMLSeparator
from subroformer.inference.export_coreml import validate_waveform
from subroformer.inference.engine import load_audio, load_separator, resolve_pretrained_paths
from subroformer.model_utils import load_config


def main():
    import coremltools as ct

    parser = argparse.ArgumentParser(description="Compare CPU and MPS DSP with an unchanged Core ML artifact")
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/coreml-perceptual-200k-fp16")
    parser.add_argument("--validation-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    config = load_config(args.artifact / "config.yaml")
    samples = int(config.audio.chunk_size)
    music = load_audio(args.validation_audio, int(config.audio.sample_rate))
    if music.shape[0] != 2 or music.shape[-1] < samples:
        raise ValueError("Validation requires at least one full stereo chunk")
    waveforms = {"noise": torch.randn(1, 2, samples) * 0.05,
                 "silence": torch.zeros(1, 2, samples)}
    for name, start in (("music", 0), ("music_middle", (music.shape[-1] - samples) // 2),
                        ("music_end", music.shape[-1] - samples)):
        waveforms[name] = torch.from_numpy(music[:, start:start + samples].copy())[None]
    config_path, checkpoint_path = resolve_pretrained_paths(local_files_only=True)
    reference, _ = load_separator(config_path, checkpoint_path, backend="torch", device="cpu")
    with torch.inference_mode():
        expected = {name: reference(audio).numpy() for name, audio in waveforms.items()}
    del reference
    report = {"platform": platform.platform(), "torch": torch.__version__,
              "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
              "coremltools": ct.__version__, "cpu_threads": torch.get_num_threads(),
              "batch_size": 1, "warmup": 2, "repeats": args.repeats,
              "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
              "validation_audio_sha256": hashlib.sha256(args.validation_audio.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("subroformer/inference/coreml.py", "subroformer/model.py", "benchmarks/benchmark_coreml_dsp.py")},
              "artifact_sha256": {str(path.relative_to(args.artifact)):
                  hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(args.artifact.rglob("*")) if path.is_file()},
              "devices": {}}
    baseline = {}
    for device in ("cpu", "mps"):
        start = time.perf_counter()
        compiled = ct.models.CompiledMLModel(str(args.artifact / "mask.mlmodelc"),
                                            compute_units=ct.ComputeUnit.CPU_AND_GPU)
        separator = CoreMLSeparator(compiled, config, load_file(str(args.artifact / "dsp.safetensors")),
                                    dsp_device=device).eval()
        load_seconds = time.perf_counter() - start
        durations = []
        with torch.inference_mode():
            for iteration in range(2 + args.repeats):
                start = time.perf_counter()
                separator(waveforms["music"])
                durations.append(time.perf_counter() - start)
            errors = {}
            for name, audio in waveforms.items():
                actual = separator(audio).numpy()
                relative, absolute = validate_waveform(expected[name], actual)
                errors[name] = {"fp32_relative_rms": relative, "fp32_absolute_rms": absolute}
                if device == "cpu":
                    baseline[name] = actual
                else:
                    relative, absolute = validate_waveform(baseline[name], actual)
                    errors[name].update(cpu_relative_rms=relative, cpu_absolute_rms=absolute,
                                        bitwise_equal=bool(np.array_equal(baseline[name], actual)))
        median = statistics.median(durations[2:])
        report["devices"][device] = {"load_s": load_seconds, "first_forward_s": durations[0],
                                     "warm_seconds": durations[2:], "median_s": median,
                                     "errors": errors}
        print(device, json.dumps(report["devices"][device]), flush=True)
        del separator, compiled
    report["speedup"] = report["devices"]["cpu"]["median_s"] / report["devices"]["mps"]["median_s"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
