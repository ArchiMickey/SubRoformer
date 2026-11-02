"""Accelerate smoke test for song sharding and global SDR aggregation."""

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf


from subroformer import evaluate


def fake_evaluate_songs(
    config,
    model,
    songs,
    hop_ratio,
    batch_size,
    amp_dtype=None,
    split=None,
    process_index=None,
    executor=None,
):
    del model, batch_size, amp_dtype, process_index, executor
    rows = [
        {"song": song.name, "sdr": float(song.name.removeprefix("song_"))}
        for song in songs
    ]
    return evaluate.summarize_evaluation_rows(config, rows, hop_ratio, split)


def main():
    accelerator = Accelerator(cpu=True)
    expected_world_size = int(os.environ.get("EXPECTED_WORLD_SIZE", "2"))
    num_songs = int(os.environ.get("NUM_SONGS", "7"))
    if expected_world_size <= 0 or num_songs <= 0:
        raise AssertionError("EXPECTED_WORLD_SIZE and NUM_SONGS must be positive")
    if accelerator.num_processes != expected_world_size:
        raise AssertionError(
            f"Expected {expected_world_size} processes, got {accelerator.num_processes}"
        )

    config = OmegaConf.create(
        {
            "training": {
                "target_instrument": "vocals",
                "instruments": ["vocals", "bass", "drums", "other"],
            }
        }
    )
    model = torch.nn.Linear(1, 1).to(accelerator.device)
    songs = [Path(f"song_{index}") for index in range(num_songs)]
    evaluate.evaluate_songs = fake_evaluate_songs

    result = evaluate.evaluate_songs_distributed(
        config=config,
        model=model,
        songs=songs,
        hop_ratio=4,
        batch_size=1,
        accelerator=accelerator,
        split="test",
    )

    assert result["songs"] == num_songs
    assert result["median_sdr"] == (num_songs - 1) / 2
    assert [row["song"] for row in result["results"]] == [
        f"song_{index}" for index in range(num_songs)
    ]
    accelerator.wait_for_everyone()
    accelerator.print("Distributed evaluation smoke test passed")


if __name__ == "__main__":
    main()
