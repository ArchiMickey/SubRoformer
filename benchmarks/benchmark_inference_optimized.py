from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

from subroformer.checkpoint_utils import load_model_checkpoint
from benchmarks.reference_model import build_reference_model
from subroformer.inference.optimized import OPTIMIZATION_STEPS, optimize_for_inference
from subroformer.model_utils import build_model, load_config


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_hashes():
    paths = [ROOT / name for name in ("subroformer/model.py", "subroformer/inference/engine.py", "subroformer/model_utils.py")]
    paths.extend(sorted((ROOT / "subroformer/frontend/dsp").glob("*.py")))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def timed_forward(model, audio, device):
    context = torch.autocast("mps", dtype=torch.float16) if device == "mps" else contextlib.nullcontext()
    synchronize(device)
    with torch.inference_mode(), context:
        start = time.perf_counter()
        output = model(audio)
        synchronize(device)
        elapsed = time.perf_counter() - start
    return elapsed, output


def compare(expected, actual):
    expected = expected.float().cpu()
    actual = actual.float().cpu()
    difference = actual - expected
    return {
        "bitwise_equal": torch.equal(expected, actual),
        "max_abs_error": difference.abs().max().item(),
        "rms_error": difference.square().mean().sqrt().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


def summarize(values):
    return {"median_s": statistics.median(values), "min_s": min(values), "max_s": max(values)}


def run_device(args, config, device):
    torch.manual_seed(args.seed)
    baseline = build_reference_model(config).eval()
    load_model_checkpoint(baseline, args.checkpoint)
    if args.merged:
        optimized = build_model(config).eval()
        optimized.load_state_dict(baseline.state_dict(), strict=True)
    else:
        optimized = optimize_for_inference(baseline, args.level)
    previous = optimize_for_inference(baseline, args.level - 1) if args.level > 1 and not args.merged else None
    baseline.to(device)
    optimized.to(device)
    if previous is not None:
        previous.to(device)
    generator = torch.Generator().manual_seed(args.seed + 1)
    audio = torch.randn(1, 2, config.audio.chunk_size, generator=generator).mul_(0.02).to(device)
    timings = {"baseline": [], "optimized": []}
    if previous is not None:
        timings["previous"] = []
    comparisons = []
    cold = {}
    reference = None
    baseline_repeatable = True
    for iteration in range(args.warmup + args.repeats):
        order = [("baseline", baseline), ("optimized", optimized)]
        if previous is not None:
            order.insert(1, ("previous", previous))
        rotation = iteration % len(order)
        order = order[rotation:] + order[:rotation]
        if iteration % 2:
            order.reverse()
        outputs = {}
        for name, model in order:
            elapsed, output = timed_forward(model, audio, device)
            outputs[name] = output.cpu()
            if iteration == 0:
                cold[name] = elapsed
            if iteration >= args.warmup:
                timings[name].append(elapsed)
        if reference is None:
            reference = outputs["baseline"].clone()
        baseline_repeatable &= torch.equal(reference, outputs["baseline"])
        comparisons.append(compare(outputs["baseline"], outputs["optimized"]))
        if previous is not None:
            comparisons.append(compare(outputs["baseline"], outputs["previous"]))
        print(device, iteration, comparisons[-1], flush=True)
    baseline_summary = summarize(timings["baseline"])
    optimized_summary = summarize(timings["optimized"])
    result = {
        "device": device,
        "precision": "autocast_fp16_fp32_weights" if device == "mps" else "fp32",
        "input_shape": list(audio.shape),
        "cold_s": cold,
        "timings_s": timings,
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "speedup": baseline_summary["median_s"] / optimized_summary["median_s"],
        "latency_reduction_pct": 100 * (1 - optimized_summary["median_s"] / baseline_summary["median_s"]),
        "baseline_repeatable": baseline_repeatable,
        "bitwise_equal": all(item["bitwise_equal"] for item in comparisons),
        "finite": all(item["finite"] for item in comparisons),
        "max_abs_error": max(item["max_abs_error"] for item in comparisons),
        "rms_error": max(item["rms_error"] for item in comparisons),
        "paired_latency_reduction_pct": statistics.median([
            100 * (1 - optimized_time / baseline_time)
            for baseline_time, optimized_time in zip(timings["baseline"], timings["optimized"])
        ]),
    }
    previous_times = timings.get("previous", timings["baseline"])
    result["previous"] = summarize(previous_times)
    result["incremental_latency_reduction_pct"] = statistics.median([
        100 * (1 - optimized_time / previous_time)
        for previous_time, optimized_time in zip(previous_times, timings["optimized"])
    ])
    if hasattr(optimized.sb_filter, "cache_bytes"):
        result["cache_bytes"] = optimized.sb_filter.cache_bytes()
    if args.verify_probes:
        probes = []
        for name, batch_size, samples in (("silence", 1, 96000), ("tones", 2, 12347), ("noise", 1, 48000)):
            if name == "silence":
                probe = torch.zeros(batch_size, 2, samples)
            elif name == "tones":
                positions = torch.arange(samples) / config.audio.sample_rate
                mono = 0.03 * torch.sin(2 * torch.pi * 440 * positions)
                mono += 0.02 * torch.sin(2 * torch.pi * 3700 * positions)
                probe = torch.stack((mono, mono * 0.7)).repeat(batch_size, 1, 1)
            else:
                probe = torch.randn(batch_size, 2, samples, generator=generator) * 0.02
            probe = probe.to(device)
            _, expected = timed_forward(baseline, probe, device)
            _, actual = timed_forward(optimized, probe, device)
            probes.append({"name": name, "input_shape": list(probe.shape), **compare(expected, actual)})
        result["verification_probes"] = probes
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=OPTIMIZATION_STEPS)
    parser.add_argument("--config", type=Path, default=ROOT / "subroformer/configs/sbroformer_base_4stem.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/subroformer_base_4stem_1400000_ema.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--verify-probes", action="store_true")
    parser.add_argument("--merged", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    if args.merged and args.level != 3:
        parser.error("the merged model uses level 3")
    if args.output.exists():
        parser.error("output already exists; choose a new result path")
    if not torch.backends.mps.is_available():
        parser.error("this comparison requires both CPU and MPS")
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    result = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": args.level,
        "merged_model": args.merged,
        "optimization": OPTIMIZATION_STEPS[args.level],
        "cumulative_steps": [OPTIMIZATION_STEPS[level] for level in range(1, args.level + 1)],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "config": str(args.config.resolve()),
        "config_sha256": file_hash(args.config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_hash(args.checkpoint),
        "original_source_sha256": source_hashes(),
        "optimized_source_sha256": file_hash(ROOT / "subroformer/inference/optimized.py"),
        "reference_builder_sha256": file_hash(ROOT / "benchmarks/reference_model.py"),
        "benchmark_source_sha256": file_hash(Path(__file__)),
        "devices": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for device in ("cpu", "mps"):
        result["devices"].append(run_device(args, config, device))
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if device == "mps":
            torch.mps.empty_cache()
    result["original_sources_unchanged"] = result["original_source_sha256"] == source_hashes()
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["original_sources_unchanged"]
    assert all(item["finite"] and item["bitwise_equal"] and item["baseline_repeatable"] for item in result["devices"])
    assert all(
        probe["finite"] and probe["bitwise_equal"]
        for device in result["devices"]
        for probe in device.get("verification_probes", [])
    )


if __name__ == "__main__":
    main()
