from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from scipy.signal import get_window
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from checkpoint_utils import load_model_checkpoint
from model_utils import build_model, load_config


DEFAULT_REPO_ID = "archimickey/SubRoformer"
DEFAULT_CONFIG = "configs/sbroformer_base_4stem.yaml"
DEFAULT_CHECKPOINT = "checkpoints/subroformer_base_4stem_1400000_ema_fp16.safetensors"


def resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(device)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must select auto, cpu, cuda, or mps")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use an MPS-enabled PyTorch on a supported Mac or --device cpu")
    return device


def resolve_amp_dtype(device: torch.device, precision: str = "auto") -> torch.dtype | None:
    if precision not in {"auto", "fp32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, fp32, bf16, or fp16")
    if precision == "fp32":
        return None
    if device.type == "cpu":
        if precision != "auto":
            raise ValueError("Use fp32 precision on CPU")
        return None
    if device.type == "cuda":
        with torch.cuda.device(device):
            bf16_supported = torch.cuda.is_bf16_supported()
    else:
        bf16_supported = torch.backends.mps.is_macos_or_newer(15, 0)
    if precision == "auto":
        if device.type == "mps":
            return torch.float16
        return torch.bfloat16 if bf16_supported else torch.float16
    if precision == "bf16" and not bf16_supported:
        raise ValueError(f"BF16 is unavailable on {device}; use auto or fp16")
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def _resolve_pretrained_file(requested, filename, download_progress=None, **download_options) -> Path:
    local_path = Path(requested if requested is not None else filename).expanduser()
    if local_path.is_file():
        return local_path
    if local_path.exists():
        raise ValueError(f"Expected a file, got {local_path}")
    if download_progress is None:
        return Path(hf_hub_download(filename=filename, **download_options))
    download_progress(filename, 0, None)
    with open(os.devnull, "w") as output:
        class DownloadProgress(tqdm):
            def __init__(self, *args, **kwargs):
                self.transfer = kwargs.get("desc", "").endswith(": downloading bytes")
                kwargs.update(disable=False, file=output)
                super().__init__(*args, **kwargs)
                if not self.transfer:
                    download_progress(filename, self.n, self.total)

            def update(self, amount=1):
                result = super().update(amount)
                if not self.transfer:
                    download_progress(filename, self.n, self.total)
                return result

        path = Path(hf_hub_download(filename=filename, tqdm_class=DownloadProgress, **download_options))
    size = path.stat().st_size
    download_progress(filename, size, size)
    return path


def resolve_pretrained_paths(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    download_progress=None,
) -> tuple[Path, Path]:
    """Prefer local files; fetch missing Base artifacts into the HF cache.

    Missing explicit paths also fall back to the canonical Base filenames.
    Existing files are never replaced, and invalid local files fail at load time.
    download_progress(filename, completed_bytes, total_bytes) reports resolution
    and file reconstruction; total_bytes is None while checking the cache/Hub.
    """
    resolved = []
    for requested, filename in (
        (config_path, DEFAULT_CONFIG), (checkpoint_path, DEFAULT_CHECKPOINT),
    ):
        resolved.append(_resolve_pretrained_file(
            requested, filename, repo_id=repo_id, revision=revision,
            cache_dir=cache_dir, local_files_only=local_files_only,
            download_progress=download_progress,
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


def separate_overlap_add(
    model, audio, segment_samples, hop_length, batch_size, amp_dtype=None,
    *, on_chunk=None, should_cancel=None, collect_output=True,
):
    """Overlap-add in a bounded working buffer, optionally returning the full result.

    on_chunk(start, chunk) receives owned, finalized float32 CPU samples.
    Set collect_output=False with a callback to avoid retaining the full output.
    """
    if segment_samples < 1 or not 1 <= hop_length <= segment_samples:
        raise ValueError("Require 1 <= hop_length <= segment_samples")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not collect_output and on_chunk is None:
        raise ValueError("on_chunk is required when collect_output=False")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or min(audio.shape) == 0 or not np.isfinite(audio).all():
        raise ValueError("Expected finite nonempty audio [channels, samples]")
    device = next(model.parameters()).device
    audio_samples = audio.shape[-1]
    segment_count = 1 + math.ceil(max(audio_samples - segment_samples, 0) / hop_length)
    capacity = segment_samples + (min(batch_size, segment_count) - 1) * hop_length
    window = get_window("hamming", segment_samples).astype(np.float32)
    weights = np.zeros(capacity, dtype=np.float32)
    emitted = 0
    result = None

    model.eval()
    with torch.inference_mode():
        for start in range(0, segment_count, batch_size):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("Separation cancelled")
            count = min(batch_size, segment_count - start)
            batch_audio = np.zeros((count, audio.shape[0], segment_samples), dtype=np.float32)
            for offset in range(count):
                position = (start + offset) * hop_length
                length = min(segment_samples, audio_samples - position)
                batch_audio[offset, :, :length] = audio[:, position:position + length]
            batch = torch.as_tensor(batch_audio, device=device)
            if amp_dtype is not None and device.type in {"cpu", "cuda", "mps"}:
                with torch.autocast(device.type, dtype=amp_dtype):
                    estimate = model(batch)
            else:
                estimate = model(batch)
            outputs = normalize_model_output(estimate).float().cpu().numpy()
            if not np.isfinite(outputs).all():
                raise ValueError("Model output contains non-finite samples")
            if start == 0:
                accumulated = np.zeros((*outputs.shape[1:-1], capacity), dtype=np.float32)
                if collect_output:
                    result = np.empty((*outputs.shape[1:-1], audio_samples), dtype=np.float32)
            for offset, segment in enumerate(outputs):
                position = offset * hop_length
                accumulated[..., position:position + segment_samples] += segment * window
                weights[position:position + segment_samples] += window
            next_segment = start + count
            finalized = min(audio_samples, next_segment * hop_length) if next_segment < segment_count else audio_samples
            length = finalized - emitted
            accumulated[..., :length] /= np.maximum(weights[:length], 1e-8)
            if collect_output:
                result[..., emitted:finalized] = accumulated[..., :length]
            if on_chunk is not None:
                on_chunk(emitted, accumulated[..., :length].copy())
            remaining = capacity - length
            accumulated[..., :remaining] = accumulated[..., length:].copy()
            accumulated[..., remaining:] = 0
            weights[:remaining] = weights[length:].copy()
            weights[remaining:] = 0
            emitted = finalized
    return result


def load_separator(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    preserve_dtype: bool = False,
    backend: str = "torch",
):
    """Load once for serving; return an eval-mode model and its resolved config."""
    from coreml_inference import load_coreml_separator, use_coreml

    if use_coreml(backend, device, config_path, checkpoint_path):
        return load_coreml_separator(
            repo_id=repo_id, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only,
            dsp_device="mps" if device == "mps" else "cpu",
        )
    device = resolve_device(device)
    config_path, checkpoint_path = resolve_pretrained_paths(
        config_path, checkpoint_path, repo_id=repo_id, revision=revision,
        cache_dir=cache_dir, local_files_only=local_files_only,
    )
    config = load_config(config_path)
    model = build_model(config)
    load_model_checkpoint(model, checkpoint_path, preserve_dtype=preserve_dtype)
    return model.to(device).eval(), config


def separate_audio(
    model, audio: np.ndarray, config, hop_ratio: int = 4,
    batch_size: int = 1, amp_dtype=None, *, on_chunk=None, should_cancel=None,
    collect_output=True,
) -> dict[str, np.ndarray] | None:
    """Separate audio at config.audio.sample_rate into named [channels, samples] arrays.

    Mono is duplicated for stereo models. No references, SDR, or file writes are used.
    The caller owns model placement; returned arrays are float32 on CPU.
    With collect_output=False, emit named chunks through on_chunk and return None.
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

    def emit_chunk(start, chunk):
        if chunk.ndim == 2:
            chunk = chunk[None]
        if chunk.shape[:2] != (len(stems), channels):
            raise ValueError(f"Unexpected model output shape: {chunk.shape}")
        on_chunk(start, dict(zip(stems, chunk)))

    estimate = separate_overlap_add(
        model, audio, segment_samples, segment_samples // hop_ratio, batch_size, amp_dtype,
        on_chunk=emit_chunk if on_chunk is not None else None,
        should_cancel=should_cancel, collect_output=collect_output,
    )
    if not collect_output:
        return None
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
    precision: str = "auto",
    hop_ratio: int = 4,
    batch_size: int = 1,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    backend: str = "torch",
) -> list[Path]:
    from coreml_inference import load_coreml_separator, use_coreml

    coreml = use_coreml(backend, device, config_path, checkpoint_path)
    if coreml:
        if precision not in {"auto", "fp16"}:
            raise ValueError("Core ML uses compiled FP16 precision; use --backend torch for other precisions")
        model, config = load_coreml_separator(
            repo_id=repo_id, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only,
            dsp_device="mps" if device == "mps" else "cpu",
        )
        amp_dtype = None
    else:
        requested_device = resolve_device(device)
        amp_dtype = resolve_amp_dtype(requested_device, precision)
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

    if not coreml:
        checkpoint_path = _resolve_pretrained_file(
            checkpoint_path, DEFAULT_CHECKPOINT, repo_id=repo_id, revision=revision,
            cache_dir=cache_dir, local_files_only=local_files_only,
        )
        model = build_model(config)
        load_model_checkpoint(model, checkpoint_path)
        model.to(requested_device)
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        writers = {
            stem: stack.enter_context(sf.SoundFile(
                path, mode="x", samplerate=sample_rate, channels=channels, subtype="FLOAT",
            ))
            for path, stem in zip(paths, stems)
        }

        def write_chunk(start, chunks):
            for stem, chunk in chunks.items():
                writers[stem].write(chunk.T)

        separate_audio(
            model, audio, config, hop_ratio, batch_size, amp_dtype,
            on_chunk=write_chunk, collect_output=False,
        )
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
    parser.add_argument("--device", default="auto", help="auto (CUDA, then MPS, then CPU), cpu, cuda[:N], or mps")
    parser.add_argument("--backend", choices=["auto", "torch", "coreml"], default="torch",
                        help="torch (default); auto also selects PyTorch; coreml is opt-in on macOS 15+")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                        help="auto uses FP16 on MPS, BF16/FP16 on CUDA, and FP32 on CPU; Core ML uses FP16")
    parser.add_argument("--hop_ratio", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    paths = separate_file(
        args.config, args.checkpoint, args.input, args.output_dir,
        args.device, args.precision, args.hop_ratio, args.batch_size,
        repo_id=args.repo_id, revision=args.revision, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        backend=args.backend,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
