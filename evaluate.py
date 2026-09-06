from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from checkpoint_utils import extract_model_state, load_checkpoint
from inference import load_audio, normalize_model_output, separate_overlap_add

from training_utils import build_model, load_config, museval_sdr




def select_subset_songs(root: str | Path, split: str, audios_num: int) -> list[Path]:
    songs = sorted(path for path in Path(root, split).iterdir() if path.is_dir())
    if not songs:
        raise ValueError(f"No songs found in {Path(root, split)}")
    if audios_num <= 0:
        raise ValueError(f"audios_num must be positive, got {audios_num}")

    skip_n = max(1, len(songs) // audios_num)
    return songs[::skip_n]


def summarize_evaluation_rows(config, rows, hop_ratio, split=None):
    if not rows:
        raise ValueError("Cannot summarize an evaluation with no song results")

    target_stem = config.training.target_instrument
    stems = list(config.training.instruments)
    if target_stem is not None:
        median_sdr = float(np.nanmedian([row["sdr"] for row in rows]))
        stem_median_sdr = {target_stem: median_sdr}
    else:
        stem_median_sdr = {
            stem: float(np.nanmedian([row["sdr"][stem] for row in rows]))
            for stem in stems
        }
        median_sdr = float(np.mean(list(stem_median_sdr.values())))

    return {
        "split": split,
        "songs": len(rows),
        "hop_ratio": hop_ratio,
        "median_sdr": median_sdr,
        "stem_median_sdr": stem_median_sdr,
        "results": rows,
    }


def order_distributed_rows(rows, songs):
    song_order = {song.name: index for index, song in enumerate(songs)}
    row_names = [row["song"] for row in rows]
    if len(row_names) != len(set(row_names)):
        raise ValueError("Distributed evaluation produced duplicate song results")

    missing = sorted(set(song_order) - set(row_names))
    unexpected = sorted(set(row_names) - set(song_order))
    if missing or unexpected:
        raise ValueError(
            "Distributed evaluation results do not match the requested songs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return sorted(rows, key=lambda row: song_order[row["song"]])


def evaluate_songs(
    config,
    model,
    songs,
    hop_ratio,
    batch_size,
    amp_dtype=None,
    split=None,
    process_index=None,
):
    sample_rate = int(config.audio.sample_rate)
    segment_samples = int(config.audio.chunk_size)
    if segment_samples % hop_ratio != 0:
        raise ValueError(f"audio.chunk_size={segment_samples} must be divisible by hop_ratio={hop_ratio}")
    hop_length = segment_samples // hop_ratio
    target_stem = config.training.target_instrument
    stems = list(config.training.instruments)
    rows = []

    for index, song in enumerate(songs, 1):
        sources = {stem: load_audio(song / f"{stem}.wav", sample_rate) for stem in stems}
        mixture = sum(sources.values())
        estimate = separate_overlap_add(
            model,
            mixture,
            segment_samples,
            hop_length,
            batch_size,
            amp_dtype,
        )
        prefix = f"{split} " if split else ""
        if process_index is not None:
            prefix += f"rank={process_index} "
        if target_stem is not None:
            score = museval_sdr(estimate, sources[target_stem], sample_rate)
            rows.append({"song": song.name, "sdr": score})
            print(f"{prefix}{index}/{len(songs)} {song.name}: SDR={score:.4f} dB", flush=True)
            continue

        if estimate.ndim != 3 or estimate.shape[0] != len(stems):
            raise ValueError(
                f"Joint separator must emit [{len(stems)}, channels, samples], got {estimate.shape}"
            )
        scores = {
            stem: museval_sdr(estimate[stem_index], sources[stem], sample_rate)
            for stem_index, stem in enumerate(stems)
        }
        rows.append({"song": song.name, "sdr": scores})
        score_text = " ".join(f"{stem}={scores[stem]:.4f}" for stem in stems)
        print(f"{prefix}{index}/{len(songs)} {song.name}: {score_text} dB", flush=True)

    return summarize_evaluation_rows(config, rows, hop_ratio, split)


def evaluate_songs_distributed(
    config,
    model,
    songs,
    hop_ratio,
    batch_size,
    accelerator,
    amp_dtype=None,
    split=None,
):
    """Evaluate disjoint whole songs on every rank and aggregate global SDRs."""
    songs = list(songs)
    if not songs:
        raise ValueError("Cannot run distributed evaluation without songs")

    # DDP forward may synchronize buffers. Ranks evaluate different numbers of
    # variable-length segments, so inference must use each rank's local module.
    local_model = accelerator.unwrap_model(model)
    was_training = local_model.training
    local_songs = songs[accelerator.process_index :: accelerator.num_processes]
    try:
        if local_songs:
            local_metrics = evaluate_songs(
                config=config,
                model=local_model,
                songs=local_songs,
                hop_ratio=hop_ratio,
                batch_size=batch_size,
                amp_dtype=amp_dtype,
                split=split,
                process_index=(
                    accelerator.process_index
                    if accelerator.num_processes > 1
                    else None
                ),
            )
            local_rows = local_metrics["results"]
        else:
            local_rows = []
    finally:
        local_model.train(was_training)

    rows = gather_object(local_rows)
    rows = order_distributed_rows(rows, songs)
    return summarize_evaluation_rows(config, rows, hop_ratio, split)


def evaluate_subset(
    config,
    model,
    root,
    split,
    audios_num,
    hop_ratio,
    batch_size,
    amp_dtype=None,
    accelerator=None,
):
    songs = select_subset_songs(root=root, split=split, audios_num=audios_num)
    if accelerator is not None:
        return evaluate_songs_distributed(
            config=config,
            model=model,
            songs=songs,
            hop_ratio=hop_ratio,
            batch_size=batch_size,
            accelerator=accelerator,
            amp_dtype=amp_dtype,
            split=split,
        )
    return evaluate_songs(
        config=config,
        model=model,
        songs=songs,
        hop_ratio=hop_ratio,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
        split=split,
    )


def evaluate(args: argparse.Namespace) -> dict:
    requested_device = torch.device(args.device)
    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError(f"--device must select cpu or cuda, got {args.device!r}")
    mixed_precision = {
        "bf16": "bf16",
        "fp16": "fp16",
        "fp32": "no",
    }[args.precision]
    accelerator = Accelerator(
        cpu=requested_device.type == "cpu",
        mixed_precision=mixed_precision,
    )

    config = load_config(args.config)
    model = build_model(config)
    if args.target_instrument:
        config.training.target_instrument = args.target_instrument
    checkpoint = load_checkpoint(args.checkpoint)
    checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    model.load_state_dict(extract_model_state(checkpoint))
    model_device = (
        requested_device
        if accelerator.num_processes == 1 and requested_device.index is not None
        else accelerator.device
    )
    model.to(model_device)

    songs = sorted(path for path in Path(args.root).iterdir() if path.is_dir())
    if args.limit is not None:
        songs = songs[:args.limit]
    if not songs:
        raise ValueError(f"No songs found in {args.root}")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]
    metrics = evaluate_songs_distributed(
        config=config,
        model=model,
        songs=songs,
        hop_ratio=args.hop_ratio,
        batch_size=args.batch_size,
        accelerator=accelerator,
        amp_dtype=amp_dtype,
    )
    payload = metrics | {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint_epoch,
        "model_type": "sbroformer",
        "root": args.root,
        "num_processes": accelerator.num_processes,
        "mixture_source": "reference_sum",
    }
    if args.output_json and accelerator.is_main_process:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    accelerator.print(f"Median SDR: {payload['median_sdr']:.4f} dB")
    accelerator.wait_for_everyone()
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Evaluation device type. Accelerate selects each rank's CUDA device.",
    )
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--hop_ratio", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--target_instrument", choices=["vocals", "bass", "drums", "other"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output_json")
    evaluate(parser.parse_args())
