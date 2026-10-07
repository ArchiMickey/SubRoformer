from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler

from subroformer.training.dataset import RandomMixingSampler, prepare_data
from tests.helpers import make_dataset, tiny_config


def test_random_mixing_loader_preserves_upstream_output_contract(tmp_path: Path):
    data_root = make_dataset(tmp_path / "train")
    config = tiny_config(target_instrument=None)
    args = SimpleNamespace(
        data_path=[str(data_root)],
        results_path=str(tmp_path / "results"),
        dataset_type=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        seed=7,
    )
    (tmp_path / "results").mkdir()

    loader = prepare_data(config, args, batch_size=1)
    stems, mixture = next(iter(loader))

    assert isinstance(loader.sampler, RandomMixingSampler)
    assert stems.shape == (1, 4, 2, 128)
    assert mixture.shape == (1, 2, 128)
    torch.testing.assert_close(mixture, stems.sum(dim=1))


def test_disabled_random_mixing_uses_zfturbo_sampling_and_mixup(tmp_path: Path):
    data_root = make_dataset(tmp_path / "train")
    config = tiny_config(target_instrument=None)
    config.augmentations.random_mixing.enable = False
    config.augmentations.mixup = True
    config.augmentations.mixup_probs = [1.0]
    config.augmentations.loudness_min = 0.5
    config.augmentations.loudness_max = 1.5
    args = SimpleNamespace(
        data_path=[str(data_root)],
        results_path=str(tmp_path / "results"),
        dataset_type=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        seed=7,
    )
    (tmp_path / "results").mkdir()
    loader = prepare_data(config, args, batch_size=1)

    with (
        patch.object(loader.dataset, "load_random_mix", wraps=loader.dataset.load_random_mix) as zfturbo_mix,
        patch.object(loader.dataset, "load_random_mixing", wraps=loader.dataset.load_random_mixing) as mini_mix,
        patch.object(loader.dataset, "load_source", wraps=loader.dataset.load_source) as load_source,
    ):
        stems, mixture = next(iter(loader))

    assert isinstance(loader.sampler, RandomSampler)
    assert zfturbo_mix.call_count == 1
    assert mini_mix.call_count == 0
    assert load_source.call_count == 8
    assert stems.shape == (1, 4, 2, 128)
    assert mixture.shape == (1, 2, 128)
    torch.testing.assert_close(mixture, stems.sum(dim=1))


def test_random_mixing_sampler_is_reproducible_per_epoch(tmp_path: Path):
    data_root = make_dataset(tmp_path / "train")
    config = tiny_config(target_instrument=None)
    args = SimpleNamespace(
        data_path=[str(data_root)],
        results_path=str(tmp_path / "results"),
        dataset_type=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        seed=11,
    )
    (tmp_path / "results").mkdir()
    loader = prepare_data(config, args, batch_size=1)

    loader.sampler.set_epoch(3)
    first = list(loader.sampler)
    loader.sampler.set_epoch(3)
    second = list(loader.sampler)
    loader.sampler.set_epoch(4)
    third = list(loader.sampler)

    assert first == second
    assert first != third


def test_stem_augmentation_does_not_mutate_shared_config(tmp_path: Path):
    data_root = make_dataset(tmp_path / "train")
    config = tiny_config(target_instrument=None)
    config.augmentations.vocals = {
        "pitch_shift": 0.0,
        "pitch_shift_min_semitones": -5,
        "pitch_shift_max_semitones": 5,
    }
    config.augmentations.other = {
        "gaussian_noise": 0.0,
        "gaussian_noise_min_amplitude": 0.001,
        "gaussian_noise_max_amplitude": 0.015,
    }
    args = SimpleNamespace(
        data_path=[str(data_root)],
        results_path=str(tmp_path / "results"),
        dataset_type=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        seed=7,
    )
    (tmp_path / "results").mkdir()
    dataset = prepare_data(config, args, batch_size=1).dataset
    shared_before = OmegaConf.to_container(config.augmentations["all"], resolve=True)
    source = np.zeros((2, config.audio.chunk_size), dtype=np.float32)

    dataset.augm_data(source, "vocals")
    dataset.augm_data(source, "other")

    assert OmegaConf.to_container(config.augmentations["all"], resolve=True) == shared_before
