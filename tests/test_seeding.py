"""Training data stream depends only on the seed, not on global RNG use."""

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from subroformer.training.dataset import prepare_data
from subroformer.training.train import build_arg_parser
from tests.helpers import make_dataset, tiny_config


def _batches(tmp_path: Path, seed: int, num_workers: int, n: int = 4):
    data_root = tmp_path / "train"
    if not data_root.exists():
        make_dataset(data_root, songs=6, samples=1024)
    config = tiny_config(target_instrument=None)
    config.augmentations.random_mixing.enable = False
    config.augmentations.loudness = True
    config.augmentations.loudness_min = 0.5
    config.augmentations.loudness_max = 1.5
    results = tmp_path / f"results_{len(list(tmp_path.glob('results_*')))}"
    results.mkdir()
    args = SimpleNamespace(
        data_path=[str(data_root)],
        results_path=str(results),
        dataset_type=1,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        seed=seed,
    )
    loader = prepare_data(config, args, batch_size=2)
    # Stand-in for model construction consuming a model-dependent amount of RNG.
    torch.randn(int(torch.randint(1, 1000, ()).item()))
    random.random()
    np.random.rand(3)
    out = []
    for i, (stems, mixture) in enumerate(loader):
        out.append(mixture)
        if i + 1 == n:
            break
    return torch.stack(out)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_stream_depends_only_on_seed(tmp_path, num_workers):
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    a = _batches(tmp_path, seed=3, num_workers=num_workers)
    if num_workers == 0:
        # Main-process augmentations draw from the global random/NumPy RNGs,
        # which train() seeds; reset them to the same state.
        random.seed(1)
        np.random.seed(1)
    torch.manual_seed(2)  # different global torch state, e.g. a different model
    b = _batches(tmp_path, seed=3, num_workers=num_workers)
    c = _batches(tmp_path, seed=4, num_workers=num_workers)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)


def test_seed_flag_parses():
    args = build_arg_parser().parse_args(["--config", "x.yaml", "--seed", "5"])
    assert args.seed == 5
    assert build_arg_parser().parse_args(["--config", "x.yaml"]).seed is None


def test_metadata_order_is_canonical(tmp_path):
    import pickle

    data_root = make_dataset(tmp_path / "train", songs=6, samples=1024)
    config = tiny_config(target_instrument=None)
    results = tmp_path / "results"
    results.mkdir()
    # A cache written in completion order (reversed here) must not change the order.
    tracks = sorted(str(p) for p in data_root.iterdir())
    with open(results / "metadata_1.pkl", "wb") as f:
        pickle.dump([[t, 1024] for t in reversed(tracks)], f)
    args = SimpleNamespace(
        data_path=[str(data_root)], results_path=str(results), dataset_type=1, num_workers=0,
        pin_memory=False, persistent_workers=False, prefetch_factor=2, seed=0,
    )
    metadata = prepare_data(config, args, batch_size=1).dataset.metadata
    assert metadata == [(t, 1024) for t in tracks]
