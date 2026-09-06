from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from einops import rearrange
from scipy.signal import get_window
from huggingface_hub import hf_hub_download

from checkpoint_utils import load_model_checkpoint
from model_utils import build_model, load_config


DEFAULT_REPO_ID = "archimickey/SubRoformer"
DEFAULT_CONFIG = "configs/sbroformer_base_4stem.yaml"
DEFAULT_CHECKPOINT = "checkpoints/subroformer_base_4stem_1400000_ema.safetensors"


def _resolve_pretrained_file(requested, filename, **download_options) -> Path:
    local_path = Path(requested if requested is not None else filename).expanduser()
    if local_path.is_file():
        return local_path
    if local_path.exists():
        raise ValueError(f"Expected a file, got {local_path}")
    return Path(hf_hub_download(filename=filename, **download_options))


def resolve_pretrained_paths(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[Path, Path]:
    """Prefer local files; fetch missing Base artifacts into the HF cache.

    Missing explicit paths also fall back to the canonical Base filenames.
    Existing files are never replaced, and invalid local files fail at load time.
    """
    resolved = []
    for requested, filename in (
        (config_path, DEFAULT_CONFIG), (checkpoint_path, DEFAULT_CHECKPOINT),
    ):
        resolved.append(_resolve_pretrained_file(
            requested, filename, repo_id=repo_id, revision=revision,
            cache_dir=cache_dir, local_files_only=local_files_only,
        ))
    return tuple(resolved)


def normalize_model_output(estimate: torch.Tensor) -> torch.Tensor:
    if estimate.ndim == 4 and estimate.shape[1] == 1:
        estimate = estimate[:, 0]
    if estimate.ndim not in {3, 4}:
        raise ValueError(
            "Expected separator output [batch, channels, samples] or "
            f"[batch, stems, channels, samples], got {tuple(estimate.shape)}"
        )
    return estimate


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.T
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate, axis=-1)
    return audio


def separate_overlap_add(model, audio, segment_samples, hop_length, batch_size, amp_dtype=None):
    """Return stems as [stems, channels, samples], or [channels, samples] for one stem."""
    if segment_samples < 1 or not 1 <= hop_length <= segment_samples:
        raise ValueError("Require 1 <= hop_length <= segment_samples")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or min(audio.shape) == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected finite nonempty audio [channels, samples]")
    device = next(model.parameters()).device
    audio_samples = audio.shape[-1]
    full_samples = max(segment_samples, segment_samples + math.ceil(max(audio_samples - segment_samples, 0) / hop_length) * hop_length)
    padded = librosa.util.fix_length(audio, size=full_samples, axis=-1)
    segments = librosa.util.frame(padded, frame_length=segment_samples, hop_length=hop_length)
    segments = rearrange(segments, "c l n -> n c l")
    outputs = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(segments), batch_size):
            batch = torch.as_tensor(segments[start:start + batch_size].copy(), device=device)
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast("cuda", dtype=amp_dtype):
                    estimate = model(batch)
            else:
                estimate = model(batch)
            estimate = normalize_model_output(estimate)
            outputs.append(estimate.float().cpu().numpy())

    outputs = np.concatenate(outputs)
    window = get_window("hamming", segment_samples).astype(np.float32)
    output_shape = (*outputs.shape[1:-1], full_samples)
    estimate = np.zeros(output_shape, dtype=outputs.dtype)
    weights = np.zeros(output_shape, dtype=outputs.dtype)
    for index, segment in enumerate(outputs):
        start = index * hop_length
        estimate[..., start:start + segment_samples] += segment * window
        weights[..., start:start + segment_samples] += window
    return (estimate / np.maximum(weights, 1e-8))[..., :audio_samples]


def load_separator(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
):
    """Load once for serving; return an eval-mode model and its resolved config."""
    config_path, checkpoint_path = resolve_pretrained_paths(
        config_path, checkpoint_path, repo_id=repo_id, revision=revision,
        cache_dir=cache_dir, local_files_only=local_files_only,
    )
    config = load_config(config_path)
    model = build_model(config)
    load_model_checkpoint(model, checkpoint_path)
    return model.to(device).eval(), config


def separate_audio(
    model, audio: np.ndarray, config, hop_ratio: int = 4,
    batch_size: int = 1, amp_dtype=None,
) -> dict[str, np.ndarray]:
    """Separate audio at config.audio.sample_rate into named [channels, samples] arrays.

    Mono is duplicated for stereo models. No references, SDR, or file writes are used.
    The caller owns model placement; returned arrays are float32 on CPU.
    """
    segment_samples = int(config.audio.chunk_size)
    if not 1 <= hop_ratio <= segment_samples:
        raise ValueError("hop_ratio must be between 1 and audio.chunk_size")
    audio = np.asarray(audio, dtype=np.float32)
    channels = int(config.model.get("audio_channels", config.audio.num_channels))
    if audio.ndim == 2 and audio.shape[0] == 1 and channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    if audio.ndim != 2 or audio.shape[0] != channels:
        raise ValueError(f"Expected audio [channels, samples] with {channels} channels")
    target = config.training.get("target_instrument")
    stems = [target] if target else list(config.training.instruments)
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("Stem names must be nonempty and unique")
    estimate = separate_overlap_add(
        model, audio, segment_samples, segment_samples // hop_ratio, batch_size, amp_dtype
    )
    if estimate.ndim == 2:
        estimate = estimate[None]
    if estimate.shape != (len(stems), channels, audio.shape[-1]):
        raise ValueError(f"Unexpected model output shape: {estimate.shape}")
    if not np.isfinite(estimate).all():
        raise ValueError("Model output contains non-finite samples")
    return dict(zip(stems, estimate))


def separate_file(
    config_path: str | Path | None,
    checkpoint_path: str | Path | None,
    input_path: str,
    output_dir: str,
    device: str = "cpu",
    precision: str = "fp32",
    hop_ratio: int = 4,
    batch_size: int = 1,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> list[Path]:
    config_path = _resolve_pretrained_file(
        config_path, DEFAULT_CONFIG, repo_id=repo_id, revision=revision,
        cache_dir=cache_dir, local_files_only=local_files_only,
    )
    config = load_config(config_path)
    segment_samples = int(config.audio.chunk_size)
    if not 1 <= hop_ratio <= segment_samples:
        raise ValueError("hop_ratio must be between 1 and audio.chunk_size")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    requested_device = torch.device(device)
    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must select cpu or cuda")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if requested_device.type == "cpu" and precision != "fp32":
        raise ValueError("Use fp32 precision on CPU")

    sample_rate = int(config.audio.sample_rate)
    audio = load_audio(Path(input_path), sample_rate)
    channels = int(config.model.get("audio_channels", config.audio.num_channels))
    if audio.shape[0] == 1 and channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    if audio.shape[0] != channels or audio.shape[-1] == 0:
        raise ValueError(f"Expected nonempty audio with {channels} channels (or mono for stereo)")
    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains non-finite samples")

    target = config.training.get("target_instrument")
    stems = [target] if target else list(config.training.instruments)
    if len(stems) != len(set(stems)) or any(
        not isinstance(stem, str) or not stem or stem in {".", ".."}
        or "/" in stem or "\\" in stem for stem in stems
    ):
        raise ValueError("Stem names must be unique safe filenames")
    destination = Path(output_dir)
    paths = [destination / f"{stem}.wav" for stem in stems]
    if any(path.exists() for path in paths):
        raise FileExistsError("Output stem files already exist; choose a new output directory")

    checkpoint_path = _resolve_pretrained_file(
        checkpoint_path, DEFAULT_CHECKPOINT, repo_id=repo_id, revision=revision,
        cache_dir=cache_dir, local_files_only=local_files_only,
    )
    model = build_model(config)
    load_model_checkpoint(model, checkpoint_path)
    model.to(requested_device)
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    estimates = separate_audio(
        model, audio, config, hop_ratio, batch_size, amp_dtype
    )
    destination.mkdir(parents=True, exist_ok=True)
    for path, stem in zip(paths, stems):
        sf.write(path, estimates[stem].T, sample_rate, subtype="FLOAT")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate an audio file without reference stems")
    parser.add_argument("--config", help="Local config; missing files fall back to the HF Base config")
    parser.add_argument("--checkpoint", help="Local weights; missing files fall back to the HF Base checkpoint")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", help="HF branch, tag, or commit")
    parser.add_argument("--cache_dir", help="Hugging Face download cache")
    parser.add_argument("--local_files_only", action="store_true", help="Use local files/cache without network")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--hop_ratio", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    paths = separate_file(
        args.config, args.checkpoint, args.input, args.output_dir,
        args.device, args.precision, args.hop_ratio, args.batch_size,
        repo_id=args.repo_id, revision=args.revision, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
