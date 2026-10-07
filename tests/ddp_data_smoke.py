from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from subroformer.training.dataset import RandomMixingSampler, prepare_data
from tests.helpers import make_dataset, tiny_config


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shared_root = Path(tempfile.gettempdir()) / f"sbroformer_ddp_{os.environ['MASTER_PORT']}"
    data_root = shared_root / "train"
    results_root = shared_root / "results"

    if rank == 0:
        make_dataset(data_root)
        results_root.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    random_mixing = os.environ.get("RANDOM_MIXING", "1") != "0"
    config = tiny_config(num_steps=3, target_instrument=None)
    config.augmentations.random_mixing.enable = random_mixing
    args = SimpleNamespace(
        data_path=[str(data_root)],
        results_path=str(results_root),
        dataset_type=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        seed=19,
    )
    loader = prepare_data(config, args, batch_size=1)
    expected_sampler = RandomMixingSampler if random_mixing else DistributedSampler
    assert isinstance(loader.sampler, expected_sampler)
    assert len(loader) == 3

    stems, mixture = next(iter(loader))
    assert stems.shape == (1, 4, 2, 128)
    torch.testing.assert_close(mixture, stems.sum(dim=1))

    first_indexes = list(loader.sampler)[:2]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, first_indexes)
    if rank == 0:
        assert gathered[0] != gathered[1]
        (shared_root / "ddp_result.json").write_text(
            json.dumps(
                {
                    "world_size": world_size,
                    "loader_length": len(loader),
                    "random_mixing": random_mixing,
                    "sampler": type(loader.sampler).__name__,
                }
            ),
            encoding="utf-8",
        )
        print((shared_root / "ddp_result.json").read_text(), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
