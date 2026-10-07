from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from omegaconf import OmegaConf

from .model import SBRoformer


def load_config(path: str):
    path = Path(path)
    config = OmegaConf.load(path)
    extends = config.pop("extends", None)
    if extends is None:
        return config
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return OmegaConf.merge(load_config(str(base_path)), config)


def build_model(config) -> nn.Module:
    model_config = OmegaConf.to_container(config.model, resolve=True)
    name = model_config.pop("name", "SBRoformer")
    if name != "SBRoformer":
        raise ValueError(f"Unsupported model: {name}")
    return SBRoformer(**model_config)

