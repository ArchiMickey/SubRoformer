from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from torch.optim import AdamW

from .dataset import prepare_data
from .evaluate import evaluate_subset
from .training_utils import (
    ModelEMA,
    build_model,
    load_config,
    make_scheduler,
    save_ema_weights,
    separation_loss,
)


def resolve_max_grad_norm(value):
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    value = float(value)
    if value <= 0:
        raise ValueError("training.max_grad_norm must be positive or null")
    return value


def resolve_distributed_timeout(value):
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    seconds = float(value)
    if seconds <= 0:
        raise ValueError("training.distributed_timeout_seconds must be positive or null")
    return timedelta(seconds=seconds)


def validate_training_data_sources(config, data_path) -> str:
    """Validate the configured dataset source without touching the filesystem."""
    data_config = config.get("data", {}) or {}
    data_kind = str(data_config.get("kind", "mss")).strip().lower()

    if data_kind != "mss":
        raise ValueError(f"Unsupported data.kind={data_kind!r}; expected 'mss'")
    if not data_path:
        raise ValueError("--data_path is required when data.kind=mss")

    return data_kind


def build_eval_metrics(train_eval: dict, test_eval: dict) -> dict[str, float]:
    """Flatten subset SDR results into tracker-friendly train/test metrics."""
    metrics = {
        "eval/train_sdr": float(train_eval["median_sdr"]),
        "eval/test_sdr": float(test_eval["median_sdr"]),
    }
    for split, result in (("train", train_eval), ("test", test_eval)):
        metrics[f"eval/{split}/stem_average_sdr"] = float(result["median_sdr"])
        for stem, score in result["stem_median_sdr"].items():
            metrics[f"eval/{split}/{stem}_sdr"] = float(score)
    return metrics


def train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    validate_training_data_sources(config, getattr(args, "data_path", None))
    if getattr(args, "target_instrument", None):
        config.training.target_instrument = args.target_instrument
    if getattr(args, "log_with", None):
        config.training.log_with = args.log_with
    if getattr(args, "project_name", None):
        config.training.project_name = args.project_name
    if args.max_updates is not None:
        config.training.max_updates = int(args.max_updates)
    if args.seed is not None:
        config.training.seed = int(args.seed)
    if args.eval_every_updates is not None:
        if args.eval_every_updates < 0:
            raise ValueError("--eval_every_updates must be non-negative")
        config.training.eval_every_updates = int(args.eval_every_updates)
    training = config.training
    seed = int(training.get("seed", 0))
    config.training.seed = seed
    # Seeds model initialization and dataset construction (random, NumPy,
    # torch, CUDA). The loader shuffles with its own generator from the same
    # seed, so models that consume different amounts of RNG at init still see
    # the same data stream.
    set_seed(seed)
    output_dir = Path(args.output_dir or training.get("output_dir", "checkpoints/sbroformer"))
    output_dir.mkdir(parents=True, exist_ok=True)

    process_group_timeout = resolve_distributed_timeout(
        training.get("distributed_timeout_seconds", None)
    )
    kwargs_handlers = []
    if process_group_timeout is not None:
        kwargs_handlers.append(InitProcessGroupKwargs(timeout=process_group_timeout))

    accelerator = Accelerator(
        mixed_precision=training.get("mixed_precision", "no"),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 1)),
        step_scheduler_with_optimizer=False,
        log_with=training.get("log_with", None),
        project_dir=str(output_dir),
        kwargs_handlers=kwargs_handlers,
    )
    if training.get("log_with", None):
        accelerator.init_trackers(
            project_name=training.get("project_name", "sbroformer"),
            config=OmegaConf.to_container(config, resolve=True),
        )

    loader_args = SimpleNamespace(
        data_path=args.data_path,
        results_path=str(output_dir),
        dataset_type=args.dataset_type,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True)),
        persistent_workers=bool(training.get("persistent_workers", True)),
        prefetch_factor=int(training.get("prefetch_factor", 2)),
        multiprocessing_context=training.get("multiprocessing_context", None),
        seed=seed,
    )
    train_loader = prepare_data(config, loader_args, int(training.batch_size))

    model = build_model(config)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training.lr),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    total_updates = int(training.num_epochs) * math.ceil(
        len(train_loader) / int(training.get("gradient_accumulation_steps", 1))
    )
    max_updates = int(training.get("max_updates", 0))
    if max_updates > 0:
        total_updates = min(total_updates, max_updates)
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=int(training.get("warmup_steps", 0)),
        total_steps=total_updates,
        min_lr_ratio=float(training.get("min_lr_ratio", 0.0)),
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    ema = ModelEMA(accelerator.unwrap_model(model), decay=float(training.get("ema_decay", 0.999)))
    ema.model.to(accelerator.device)
    accelerator.register_for_checkpointing(ema)

    global_update = 0
    start_epoch = 0
    if args.resume:
        accelerator.load_state(args.resume)
        metadata_path = Path(args.resume, "training_state.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            global_update = int(metadata.get("global_update", 0))
            start_epoch = int(metadata.get("epoch", 0))
        # load_state restores the global RNGs; reseed the loader's generator
        # too, or a resumed run would replay the data stream from update 0.
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(seed + 1_000_003 * (global_update + 1))

    max_grad_norm = resolve_max_grad_norm(training.get("max_grad_norm", None))
    save_every = int(training.get("save_every_updates", 0))
    eval_every = int(training.get("eval_every_updates", 20_000))
    validation = config.get("validation", {})
    eval_root = args.eval_root or validation.get("root", "data/musdb18hq")
    eval_audios_num = int(validation.get("audios_num", 5))
    eval_hop_ratio = int(validation.get("hop_ratio", 1))
    eval_batch_size = int(validation.get("batch_size", training.batch_size))
    eval_amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": None,
        "fp32": None,
    }[training.get("mixed_precision", "no")]
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, int(training.num_epochs)):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        model.train()

        for stems, mixture in train_loader:
            mixture = mixture.to(accelerator.device, non_blocking=True)
            target = stems.to(accelerator.device, non_blocking=True)
            if target.ndim == 4 and target.shape[1] == 1:
                target = target[:, 0]

            with accelerator.accumulate(model):
                output = model(mixture)
                loss = separation_loss(output, target, config)
                accelerator.backward(loss)

                grad_norm = None
                if accelerator.sync_gradients:
                    clip_value = max_grad_norm if max_grad_norm is not None else float("inf")
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_value)

                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_update += 1
            ema.update(accelerator.unwrap_model(model))
            metrics = {
                "train/loss": float(loss.detach()),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": float(grad_norm.detach()) if torch.is_tensor(grad_norm) else float(grad_norm),
            }
            accelerator.log(metrics, step=global_update)
            accelerator.print(
                f"epoch={epoch} update={global_update}/{total_updates} "
                f"loss={metrics['train/loss']:.6f} grad_norm={metrics['train/grad_norm']:.4f}"
            )

            if eval_every > 0 and global_update % eval_every == 0:
                del output, mixture, target, loss
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                train_eval = evaluate_subset(
                    config=config,
                    model=ema.model,
                    root=eval_root,
                    split="train",
                    audios_num=eval_audios_num,
                    hop_ratio=eval_hop_ratio,
                    batch_size=eval_batch_size,
                    amp_dtype=eval_amp_dtype,
                    accelerator=accelerator,
                )
                test_eval = evaluate_subset(
                    config=config,
                    model=ema.model,
                    root=eval_root,
                    split="test",
                    audios_num=eval_audios_num,
                    hop_ratio=eval_hop_ratio,
                    batch_size=eval_batch_size,
                    amp_dtype=eval_amp_dtype,
                    accelerator=accelerator,
                )
                if accelerator.is_main_process:
                    eval_metrics = build_eval_metrics(train_eval, test_eval)
                    accelerator.log(eval_metrics, step=global_update)
                    eval_summary = " ".join(
                        f"{name.removeprefix('eval/').replace('/', '_')}={score:.4f}"
                        for name, score in eval_metrics.items()
                        if name.startswith(("eval/train/", "eval/test/"))
                    )
                    accelerator.print(
                        f"subset_eval update={global_update} {eval_summary}"
                    )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                accelerator.wait_for_everyone()

            if save_every > 0 and global_update % save_every == 0:
                checkpoint_dir = output_dir / f"step_{global_update}"
                accelerator.save_state(str(checkpoint_dir))
                if accelerator.is_main_process:
                    (checkpoint_dir / "training_state.json").write_text(
                        json.dumps({"global_update": global_update, "epoch": epoch}),
                        encoding="utf-8",
                    )

            if max_updates > 0 and global_update >= max_updates:
                break

        if max_updates > 0 and global_update >= max_updates:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_ema_weights(ema, output_dir / "model_ema.pt")
        torch.save(accelerator.get_state_dict(model), output_dir / "model.pt")
    accelerator.end_training()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_path", nargs="+")
    parser.add_argument("--dataset_type", type=int, default=1)
    parser.add_argument("--output_dir")
    parser.add_argument("--eval_root")
    parser.add_argument("--resume")
    parser.add_argument("--max_updates", type=int)
    parser.add_argument("--seed", type=int, help="overrides training.seed")
    parser.add_argument("--eval_every_updates", type=int)
    parser.add_argument("--target_instrument", choices=["vocals", "bass", "drums", "other"])
    parser.add_argument("--log_with")
    parser.add_argument("--project_name")
    return parser


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
