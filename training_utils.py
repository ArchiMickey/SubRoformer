from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import librosa
import museval
import numpy as np
import torch
import torch.nn as nn

from model_utils import build_model, load_config




class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for name, ema_value in self.model.state_dict().items():
            value = source[name].detach()
            if torch.is_floating_point(ema_value) or torch.is_complex(ema_value):
                ema_value.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.decay = float(state_dict["decay"])
        self.model.load_state_dict(state_dict["model"])


def multiresolution_stft_l1(
    output: torch.Tensor,
    target: torch.Tensor,
    window_sizes: list[int],
    n_fft: int,
    hop_size: int,
    normalized: bool,
) -> torch.Tensor:
    # Treat every batch, stem, and audio-channel slice as one equally
    # weighted waveform for torch.stft, which accepts only 1D or 2D input.
    output = output.flatten(0, -2)
    target = target.flatten(0, -2)
    losses = []
    for window_size in window_sizes:
        window_size = int(window_size)
        resolution_n_fft = max(window_size, int(n_fft))
        window = torch.hann_window(window_size, device=output.device, dtype=output.dtype)
        output_stft = torch.stft(
            output,
            n_fft=resolution_n_fft,
            hop_length=int(hop_size),
            win_length=window_size,
            window=window,
            normalized=bool(normalized),
            return_complex=True,
        )
        target_stft = torch.stft(
            target,
            n_fft=resolution_n_fft,
            hop_length=int(hop_size),
            win_length=window_size,
            window=window,
            normalized=bool(normalized),
            return_complex=True,
        )
        losses.append(torch.nn.functional.l1_loss(output_stft, target_stft))
    return torch.stack(losses).sum() if losses else output.new_zeros(())


def separation_loss(output: torch.Tensor, target: torch.Tensor, config) -> torch.Tensor:
    waveform = torch.nn.functional.l1_loss(output, target)
    loss_name = config.training.get("loss", "l1")
    if loss_name == "l1":
        return waveform
    if loss_name == "l1_wav_l1_multistft":
        resolutions = list(config.training.get("multistft_window_sizes", [4096, 2048, 1024, 512, 256]))
        n_fft = int(config.training.get("multistft_n_fft", 2048))
        hop_size = int(config.training.get("multistft_hop_size", 147))
        normalized = bool(config.training.get("multistft_normalized", False))
        weight = float(config.training.get("multistft_weight", 1.0))
        mrstft = multiresolution_stft_l1(
            output.float(),
            target.float(),
            resolutions,
            n_fft=n_fft,
            hop_size=hop_size,
            normalized=normalized,
        )
        return waveform + weight * mrstft
    raise ValueError(f"Unsupported loss: {loss_name}")


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def museval_sdr(output: np.ndarray, target: np.ndarray, sample_rate: int) -> float:
    museval_sample_rate = 44_100
    output = librosa.resample(output, orig_sr=sample_rate, target_sr=museval_sample_rate, axis=-1)
    target = librosa.resample(target, orig_sr=sample_rate, target_sr=museval_sample_rate, axis=-1)
    sdrs, _, _, _ = museval.evaluate(
        references=target.T[None, :, :],
        estimates=output.T[None, :, :],
    )
    return float(np.nanmedian(sdrs))


def save_ema_weights(ema: ModelEMA, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ema.model.state_dict(), path)
