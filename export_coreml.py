from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from inference import DEFAULT_REPO_ID, load_audio, load_separator, resolve_pretrained_paths
from coreml_inference import COREML_DIRECTORY, CoreMLSeparator


class MaskNetwork(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features):
        return self.model.predict_mask(features)


def use_fp16_for_op(op):
    return not any(
        module in {"RMSNorm", "RMSNorm2d", "GRN"}
        for scopes in op.scopes.values() for module in scopes
    )


def validate_waveform(expected, actual):
    if not np.isfinite(actual).all():
        raise RuntimeError("Core ML waveform contains non-finite values")
    reference_rms = float(np.sqrt(np.mean(expected.astype(np.float64) ** 2)))
    absolute_rms = float(np.sqrt(np.mean((actual.astype(np.float64) - expected) ** 2)))
    relative_rms = absolute_rms / max(reference_rms, 1e-8)
    if absolute_rms > max(0.001 * reference_rms, 1e-6):
        raise RuntimeError("Core ML waveform parity failed")
    return relative_rms, absolute_rms


def main():
    import coremltools as ct

    parser = argparse.ArgumentParser(description="Compile the Base mask network for Core ML")
    parser.add_argument("--output", type=Path, default=Path("artifacts/coreml-fp16"))
    parser.add_argument("--validation-audio", type=Path, required=True)
    parser.add_argument("--upload", action="store_true", help="Publish validated runtime artifacts to the default HF repo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    config_path, checkpoint_path = resolve_pretrained_paths()
    model, config = load_separator(config_path, checkpoint_path, backend="torch")
    OmegaConf.save(config, args.output / "config.yaml")
    samples = int(config.audio.chunk_size)
    audio = torch.randn(1, 2, samples) * 0.05
    with torch.inference_mode():
        features = model.analysis(model.sb_filter.analysis(model.pad_audio(audio, model.sb_factor)))
        network = MaskNetwork(model).eval()
        print("Tracing features", tuple(features.shape), flush=True)
        traced = torch.jit.trace(network, features, check_trace=False)
    converted = ct.convert(
        traced, inputs=[ct.TensorType(name="features", shape=features.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="mask", dtype=np.float32)],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.transform.FP16ComputePrecision(op_selector=use_fp16_for_op),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    converted.save(str(args.output / "mask.mlpackage"))
    compiled_path = args.output / "mask.mlmodelc"
    shutil.copytree(converted.get_compiled_model_path(), compiled_path, dirs_exist_ok=True)
    compiled = ct.models.CompiledMLModel(str(compiled_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    with torch.inference_mode():
        reference = network(features).numpy()
    prediction = compiled.predict({"features": features.numpy()})["mask"]
    relative_rms = float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))
    print("Mask relative RMS error:", relative_rms, flush=True)
    if not np.isfinite(prediction).all() or relative_rms > 0.01:
        raise RuntimeError("Core ML mask parity failed")
    dsp_state = model.sb_filter.state_dict()
    save_file(dsp_state, str(args.output / "dsp.safetensors"))
    separator = CoreMLSeparator(compiled, config, dsp_state).eval()
    music = load_audio(args.validation_audio, int(config.audio.sample_rate))
    if music.shape[0] != 2 or music.shape[-1] < samples:
        raise ValueError("Validation requires at least one full stereo chunk")
    errors = {}
    absolute_errors = {}
    for name, waveform in (
        ("noise", audio), ("silence", torch.zeros_like(audio)),
        ("music", torch.from_numpy(music[:, :samples].copy())[None]),
        ("music_middle", torch.from_numpy(music[:, (music.shape[-1] - samples) // 2:(music.shape[-1] - samples) // 2 + samples].copy())[None]),
        ("music_end", torch.from_numpy(music[:, -samples:].copy())[None]),
    ):
        with torch.inference_mode():
            expected = model(waveform).numpy()
            actual = separator(waveform).numpy()
        error, absolute_error = validate_waveform(expected, actual)
        print(name, "waveform relative/absolute RMS error:", error, absolute_error, flush=True)
        errors[name] = error
        absolute_errors[name] = absolute_error
    (args.output / "conversion.json").write_text(json.dumps({
        "feature_shape": list(features.shape), "chunk_size": samples,
        "mask_relative_rms": relative_rms, "coremltools": ct.__version__,
        "torch": torch.__version__, "precision": "float16",
        "fp32_modules": ["RMSNorm", "RMSNorm2d", "GRN"],
        "waveform_relative_rms": errors,
        "waveform_absolute_rms": absolute_errors,
        "waveform_rms_tolerance": {"relative": 0.001, "absolute_floor": 1e-6},
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    if args.upload:
        from huggingface_hub import HfApi

        commit = HfApi().upload_folder(
            repo_id=DEFAULT_REPO_ID, folder_path=args.output, path_in_repo=COREML_DIRECTORY,
            allow_patterns=["mask.mlmodelc/**", "config.yaml", "dsp.safetensors", "conversion.json"],
            commit_message="Add validated macOS 15+ FP16 Core ML Base mask network and exact DSP buffers",
        )
        print("Uploaded:", commit.commit_url, flush=True)


if __name__ == "__main__":
    main()
