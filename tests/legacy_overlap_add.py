import math

import librosa
import numpy as np
import torch
from einops import rearrange
from scipy.signal import get_window

from inference import normalize_model_output


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
            if amp_dtype is not None and device.type in {"cuda", "mps"}:
                with torch.autocast(device.type, dtype=amp_dtype):
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


