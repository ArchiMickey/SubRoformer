from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf


STEMS = ["vocals", "bass", "drums", "other"]


def make_dataset(root: Path, songs: int = 6, sample_rate: int = 1600, samples: int = 256) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    time = np.arange(samples, dtype=np.float32) / sample_rate
    for song_index in range(songs):
        song_dir = root / f"song_{song_index:02d}"
        song_dir.mkdir()
        mixture = np.zeros((samples, 2), dtype=np.float32)
        for stem_index, stem in enumerate(STEMS, 1):
            frequency = 20 + song_index * 3 + stem_index * 5
            mono = 0.02 * stem_index * np.sin(2 * np.pi * frequency * time)
            audio = np.stack([mono, mono * 0.9], axis=-1).astype(np.float32)
            sf.write(song_dir / f"{stem}.wav", audio, sample_rate, subtype="FLOAT")
            mixture += audio
        sf.write(song_dir / "mixture.wav", mixture, sample_rate, subtype="FLOAT")
    return root


def tiny_config(num_steps: int = 2, target_instrument=None):
    return OmegaConf.create(
        {
            "audio": {
                "sample_rate": 1600,
                "chunk_size": 128,
                "num_channels": 2,
                "min_mean_abs": 0.0,
            },
            "model": {
                "name": "SBRoformer",
                "audio_channels": 2,
                "sample_rate": 1600,
                "analysis_type": "stft",
                "mask_mode": "features",
                "n_fft": 8,
                "stft_hop_length": 2,
                "subband_n_bands": 4,
                "max_half_bandwidth": 200,
                "dim": 32,
                "dim_sp": 32,
                "dim_head": 8,
                "patch_size": [2, 1],
                "n_layers": 1,
                "n_pre_layers": 1,
                "n_post_layers": 1,
                "use_fusion": True,
                "bandpass_filter_len": 32,
                "upsample_filter_len": 16,
                "chunk_size": 2,
            },
            "training": {
                "batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_workers": 0,
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": 2,
                "instruments": STEMS,
                "target_instrument": target_instrument,
                "num_epochs": 1,
                "num_steps": num_steps,
                "lr": 1e-3,
                "warmup_steps": 0,
                "mixed_precision": "no",
                "max_grad_norm": 1.0,
                "ema_decay": 0.9,
                "loss": "l1",
                "save_every_updates": 0,
                "seed": 123,
                "read_metadata_procs": 1,
            },
            "augmentations": {
                "enable": True,
                "loudness": False,
                "random_mixing": {
                    "enable": True,
                    "min_sources": 2,
                    "max_sources": 2,
                    "time_align": "random",
                },
                "all": {"channel_shuffle": 0.0},
            },
        }
    )
