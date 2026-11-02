from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from threadpoolctl import threadpool_limits
from .checkpoint_utils import extract_model_state, load_checkpoint
from .inference.engine import load_audio, normalize_model_output, separate_overlap_add

from .training.training_utils import build_model, load_config, museval_sdr




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
        "median_of_stem_medians_sdr": float(np.median(list(stem_median_sdr.values()))),
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


def _limit_worker_threads():
    # museval is single-thread bound; stop BLAS starting one thread per core in every worker.
    threadpool_limits(1)


def _scoring_executor(scoring_workers: int) -> ProcessPoolExecutor | None:
    if scoring_workers <= 0:
        return None
    # Spawn: the parent holds CUDA state and threads that must not be forked.
    return ProcessPoolExecutor(
        max_workers=scoring_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_limit_worker_threads,
    )


def _load_sources(song, stems, sample_rate):
    return {stem: load_audio(song / f"{stem}.wav", sample_rate) for stem in stems}


def _separate_songs(config, model, songs, hop_ratio, batch_size, amp_dtype=None, executor=None):
    """Separate songs in order, returning (song, {stem: SDR}) pairs.

    Without ``executor`` each stem is scored inline. With one, each stem's
    museval is submitted to the worker processes and its value is a Future,
    so separation of later songs overlaps scoring. The next song's audio is
    loaded on a thread while the current one is separated.
    """
    sample_rate = int(config.audio.sample_rate)
    segment_samples = int(config.audio.chunk_size)
    if segment_samples % hop_ratio != 0:
        raise ValueError(f"audio.chunk_size={segment_samples} must be divisible by hop_ratio={hop_ratio}")
    hop_length = segment_samples // hop_ratio
    target_stem = config.training.target_instrument
    stems = list(config.training.instruments)
    items = []
    if not songs:
        return items

    with ThreadPoolExecutor(max_workers=1) as loader:
        next_sources = loader.submit(_load_sources, songs[0], stems, sample_rate)
        for index, song in enumerate(songs):
            sources = next_sources.result()
            if index + 1 < len(songs):
                next_sources = loader.submit(_load_sources, songs[index + 1], stems, sample_rate)
            mixture = sum(sources.values())
            estimate = separate_overlap_add(
                model,
                mixture,
                segment_samples,
                hop_length,
                batch_size,
                amp_dtype,
            )
            if target_stem is not None:
                pairs = {target_stem: (estimate, sources[target_stem])}
            else:
                if estimate.ndim != 3 or estimate.shape[0] != len(stems):
                    raise ValueError(
                        f"Joint separator must emit [{len(stems)}, channels, samples], got {estimate.shape}"
                    )
                pairs = {stem: (estimate[stem_index], sources[stem]) for stem_index, stem in enumerate(stems)}
            scores = {
                stem: (
                    museval_sdr(output, reference, sample_rate)
                    if executor is None
                    else executor.submit(museval_sdr, output, reference, sample_rate)
                )
                for stem, (output, reference) in pairs.items()
            }
            items.append((song, scores))
            print(f"Separated {index + 1}/{len(songs)}: {song.name}", flush=True)
    return items


def _collect_rows(items, target_stem, split=None, process_index=None) -> list[dict]:
    prefix = f"{split} " if split else ""
    if process_index is not None:
        prefix += f"rank={process_index} "
    rows = []
    for index, (song, scores) in enumerate(items, 1):
        scores = {
            stem: score.result() if isinstance(score, Future) else score
            for stem, score in scores.items()
        }
        if target_stem is not None:
            score = scores[target_stem]
            rows.append({"song": song.name, "sdr": score})
            print(f"{prefix}{index}/{len(items)} {song.name}: SDR={score:.4f} dB", flush=True)
            continue
        rows.append({"song": song.name, "sdr": scores})
        score_text = " ".join(f"{stem}={score:.4f}" for stem, score in scores.items())
        print(f"{prefix}{index}/{len(items)} {song.name}: {score_text} dB", flush=True)
    return rows


def evaluate_songs(
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
    items = _separate_songs(config, model, songs, hop_ratio, batch_size, amp_dtype, executor)
    rows = _collect_rows(items, config.training.target_instrument, split, process_index)
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
    scoring_workers=0,
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
    executor = _scoring_executor(scoring_workers)
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
                executor=executor,
            )
            local_rows = local_metrics["results"]
        else:
            local_rows = []
    finally:
        local_model.train(was_training)
        if executor is not None:
            executor.shutdown(cancel_futures=True)

    rows = gather_object(local_rows)
    rows = order_distributed_rows(rows, songs)
    return summarize_evaluation_rows(config, rows, hop_ratio, split)


def evaluate_subsets(
    config,
    model,
    root,
    splits,
    audios_num,
    hop_ratio,
    batch_size,
    amp_dtype=None,
    accelerator=None,
    scoring_workers=0,
) -> dict:
    """Evaluate the subset of every split with one museval scoring pool.

    All splits are separated before any score is collected, so with
    scoring_workers > 0 scoring overlaps all later separation. Returns
    ``{split: summary}``; scores match inline scoring.
    """
    local_model = accelerator.unwrap_model(model) if accelerator is not None else model
    rank, world = (
        (accelerator.process_index, accelerator.num_processes) if accelerator is not None else (0, 1)
    )
    target_stem = config.training.target_instrument
    was_training = local_model.training
    executor = _scoring_executor(scoring_workers)
    try:
        pending = {}
        for split in splits:
            songs = select_subset_songs(root=root, split=split, audios_num=audios_num)
            items = _separate_songs(
                config, local_model, songs[rank::world], hop_ratio, batch_size, amp_dtype, executor,
            )
            pending[split] = (songs, items)

        results = {}
        for split, (songs, items) in pending.items():
            rows = _collect_rows(items, target_stem, split, rank if world > 1 else None)
            if accelerator is not None:
                rows = order_distributed_rows(gather_object(rows), songs)
            results[split] = summarize_evaluation_rows(config, rows, hop_ratio, split)
        return results
    finally:
        local_model.train(was_training)
        if executor is not None:
            executor.shutdown(cancel_futures=True)


def format_evaluation_table(metrics: dict) -> str:
    rows = [(stem, score) for stem, score in metrics["stem_median_sdr"].items()]
    if len(rows) > 1:
        rows += [("Mean of stem medians", metrics["median_sdr"]),
                 ("Median of stem medians", metrics["median_of_stem_medians_sdr"])]
    width = max(len("Stem / summary"), *(len(label) for label, _ in rows))
    border = f"+-{'-' * width}-+--------------+"
    lines = [border, f"| {'Stem / summary':<{width}} | Median SDR   |", border]
    lines.extend(f"| {label:<{width}} | {score:>9.4f} dB |" for label, score in rows)
    return "\n".join([*lines, border])


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
        scoring_workers=args.scoring_workers,
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
    accelerator.print(format_evaluation_table(payload))
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
    parser.add_argument("--scoring_workers", type=int, default=16, help="Processes for parallel museval scoring (0: inline)")
    parser.add_argument("--output_json")
    evaluate(parser.parse_args())
