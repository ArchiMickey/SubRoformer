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


def load_model_checkpoint(model: torch.nn.Module, path: str | Path):
    """Strictly load model weights, rejecting missing or unexpected tensors."""
    return model.load_state_dict(extract_model_state(load_checkpoint(path)), strict=True)


def export_safetensors(source: str | Path, destination: str | Path, metadata=None) -> Path:
    """Export inference weights without changing tensor dtypes or values."""
    destination = Path(destination)
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("Destination must use the .safetensors extension")
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = load_checkpoint(source)
    state = extract_model_state(checkpoint)
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in state.items()}
    details = {str(key): str(value) for key, value in (metadata or {}).items()}
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
    args = parser.parse_args()
    print(export_safetensors(args.checkpoint, args.output))


if __name__ == "__main__":
    main()

