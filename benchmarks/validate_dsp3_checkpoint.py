"""Matched DSP2/DSP3 perceptual-checkpoint validation on real audio excerpts."""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


from subroformer.checkpoint_utils import load_model_checkpoint
from subroformer.inference.engine import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, separate_audio
from subroformer.configs import get_config_path
from subroformer.model_utils import build_model, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", type=Path, default=get_config_path(Path(DEFAULT_CONFIG).name))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tracks", type=int, default=50)
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp16", "bf16"], default=["fp32", "fp16", "bf16"])
    parser.add_argument("--long-seconds", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(args.config)
    models = {}
    for backend in ("dsp2", "dsp3"):
        config.model.dsp_backend = backend
        model = build_model(config)
        load_model_checkpoint(model, args.checkpoint)
        models[backend] = model.to(args.device).eval()
    tracks = sorted(args.dataset.glob("*/mixture.wav"))[:args.tracks]
    if len(tracks) != args.tracks or not tracks:
        raise ValueError(f"Expected {args.tracks} tracks, found {len(tracks)}")
    results = []
    sample_rate = int(config.audio.sample_rate)
    for track_index, path in enumerate(tracks):
        durations = [2.0]
        if track_index == 0 and args.long_seconds > 2:
            durations.append(args.long_seconds)
        for duration in durations:
            info = sf.info(path)
            audio, original_rate = sf.read(path, start=10 * info.samplerate,
                                          frames=round(duration * info.samplerate),
                                          dtype="float32", always_2d=True)
            if len(audio) != round(duration * original_rate):
                raise ValueError(f"Insufficient audio: {path}")
            audio = librosa.resample(audio.T, orig_sr=original_rate, target_sr=sample_rate)
            for precision in args.precisions:
                dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
                outputs = [separate_audio(model, audio, config, amp_dtype=dtype)
                           for model in models.values()]
                metrics = {}
                for stem in config.training.instruments:
                    expected, actual = (output[stem].astype(np.float64) for output in outputs)
                    difference = actual - expected
                    maximum = float(np.abs(difference).max())
                    relative = float(np.sqrt(np.mean(difference ** 2) / max(np.mean(expected ** 2), 1e-30)))
                    metrics[stem] = {"max_absolute_error": maximum, "relative_rms_error": relative}
                    if not np.isfinite(actual).all() or maximum > 1e-5 or relative > 1e-5:
                        raise AssertionError(f"Parity failed: {path.parent.name}, {precision}, {stem}: {metrics[stem]}")
                result = {"track": path.parent.name, "seconds": duration, "precision": precision, "stems": metrics}
                results.append(result)
                print(json.dumps(result), flush=True)
    source_paths = [Path("subroformer/model.py"), Path("subroformer/inference/engine.py")]
    for backend in models:
        source_paths.extend(path for path in Path("subroformer/frontend", backend).rglob("*")
                            if path.suffix in {".py", ".cpp", ".metal"})
    report = {
        "device": args.device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "weights": "FP32 model parameters loaded strictly from checkpoint",
        "protocol": "MUSDB mixture excerpts at 10s; 48kHz; two-second windows; hop ratio 4; batch 1; TF32 disabled",
        "scope": "Output parity, not a full-song SDR or listening evaluation",
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_paths)},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(results)} matched comparisons", flush=True)


if __name__ == "__main__":
    main()
