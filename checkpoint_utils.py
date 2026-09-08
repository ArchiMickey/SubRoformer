from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model", "model_state_dict", "state_dict", "state"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def load_checkpoint(path: str | Path) -> dict:
    """Load safetensors weights or a weights-only PyTorch checkpoint on CPU."""
    path = Path(path)
    if path.suffix.lower() == ".safetensors":
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
        result = {"model": load_file(path, device="cpu")}
        for key in ("epoch", "global_update"):
            if key in metadata:
                result[key] = int(metadata[key])
        return result
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model_checkpoint(model: torch.nn.Module, path: str | Path, *, preserve_dtype: bool = False):
    """Strictly load model weights, rejecting missing or unexpected tensors."""
    return model.load_state_dict(
        extract_model_state(load_checkpoint(path)), strict=True, assign=preserve_dtype,
    )


def export_safetensors(source: str | Path, destination: str | Path, metadata=None, *, fp16: bool = False) -> Path:
    """Export inference weights, optionally casting floating tensors to FP16."""
    destination = Path(destination)
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("Destination must use the .safetensors extension")
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = load_checkpoint(source)
    state = extract_model_state(checkpoint)
    tensors = {
        name: (value.half() if fp16 and value.is_floating_point() else value).detach().cpu().contiguous().clone()
        for name, value in state.items()
    }
    details = {}
    if Path(source).suffix.lower() == ".safetensors":
        with safe_open(source, framework="pt", device="cpu") as original:
            details.update(original.metadata() or {})
    details.update({str(key): str(value) for key, value in (metadata or {}).items()})
    if fp16:
        details["precision"] = "fp16"
    for key in ("epoch", "global_update", "decay"):
        if key in checkpoint:
            details[key] = str(checkpoint[key])
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, destination, metadata=details)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inference weights to safetensors")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true", help="Convert floating tensors to FP16")
    args = parser.parse_args()
    print(export_safetensors(args.checkpoint, args.output, fp16=args.fp16))


if __name__ == "__main__":
    main()
