from __future__ import annotations

import copy
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from model_utils import build_model, load_config


COREML_DIRECTORY = "coreml/base-4stem-fp16-v1"


def use_coreml(backend, device, config_path=None, checkpoint_path=None):
    if backend not in {"auto", "torch", "coreml"}:
        raise ValueError("backend must be auto, torch, or coreml")
    supported = sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0]) >= 15
    if backend == "coreml":
        if not supported:
            raise ValueError("Core ML requires macOS 15 or newer")
        if config_path or checkpoint_path:
            raise ValueError("Core ML supports the published Base model, not custom configs/checkpoints")
        return True
    return False


class CoreMLSeparator(torch.nn.Module):
    def __init__(self, compiled, config, dsp_state=None):
        super().__init__()
        self.compiled = compiled
        self.segment_samples = int(config.audio.chunk_size)
        dsp_config = copy.deepcopy(config)
        for key in ("n_layers", "n_pre_layers", "n_post_layers"):
            dsp_config.model[key] = 0
        self.dsp = build_model(dsp_config).eval()
        if self.dsp.frontend_type != "waveform_subband" or self.dsp.analysis_type != "stft" or self.dsp.stft_backend != "torch":
            raise ValueError("Core ML artifact requires the Base waveform-subband/STFT frontend")
        if dsp_state is not None:
            self.dsp.sb_filter.load_state_dict(dsp_state, strict=True)
        for name in ("pre_blocks", "blocks", "post_blocks", "down", "up", "fusion", "stem_mask_heads", "rope"):
            delattr(self.dsp, name)
        self.dsp.predict_mask = self.predict_mask
        self.register_parameter("device_marker", torch.nn.Parameter(torch.empty(0), requires_grad=False))

    def predict_mask(self, features):
        prediction = self.compiled.predict({"features": features.contiguous().numpy()})["mask"]
        if not np.isfinite(prediction).all():
            raise ValueError("Core ML output contains non-finite values")
        return torch.from_numpy(np.asarray(prediction, dtype=np.float32))

    def forward(self, audio):
        if audio.device.type != "cpu" or audio.ndim != 3 or audio.shape[1:] != (2, self.segment_samples):
            raise ValueError("Core ML expects CPU stereo audio at the compiled chunk size")
        return torch.cat([self.dsp(chunk.float()) for chunk in audio.split(1)], dim=0)


def load_coreml_separator(*, repo_id="archimickey/SubRoformer", revision=None, cache_dir=None,
                          local_files_only=False, download_progress=None):
    import coremltools as ct

    if download_progress is not None:
        download_progress("Core ML model", 0, None)
    snapshot = Path(snapshot_download(
        repo_id=repo_id, revision=revision, cache_dir=cache_dir,
        local_files_only=local_files_only,
        allow_patterns=[f"{COREML_DIRECTORY}/config.yaml", f"{COREML_DIRECTORY}/dsp.safetensors",
                        f"{COREML_DIRECTORY}/mask.mlmodelc/**"],
    ))
    directory = snapshot / COREML_DIRECTORY
    config = load_config(directory / "config.yaml")
    compiled = ct.models.CompiledMLModel(str(directory / "mask.mlmodelc"), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    if download_progress is not None:
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        download_progress("Core ML model", size, size)
    return CoreMLSeparator(compiled, config, load_file(str(directory / "dsp.safetensors"))).eval(), config
