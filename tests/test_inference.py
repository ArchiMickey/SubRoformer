import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from subroformer.inference.engine import separate_file
from subroformer.inference.engine import load_separator, separate_audio, separate_overlap_add
from subroformer.checkpoint_utils import export_safetensors
from tests.helpers import STEMS, tiny_config
from subroformer.training.training_utils import build_model


@pytest.mark.parametrize("samples", [20, 128, 277])
@pytest.mark.parametrize("joint", [False, True])
def test_named_inference_matches_evaluation_core(tmp_path, samples, joint):
    config = tiny_config(target_instrument=None if joint else "vocals")
    config.model.num_stems = 4 if joint else 1
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    source = tmp_path / "model.pt"
    torch.save(build_model(config).state_dict(), source)
    destination = export_safetensors(source, tmp_path / "model.safetensors")
    model, config = load_separator(str(config_path), str(destination))
    assert not model.training
    audio = np.random.default_rng(42).standard_normal((1, samples)).astype(np.float32) * 0.01
    outputs = separate_audio(model, audio, config, batch_size=2)
    expected = separate_overlap_add(model, np.repeat(audio, 2, axis=0), 128, 32, 2)
    if not joint:
        expected = expected[None]
    assert list(outputs) == (STEMS if joint else ["vocals"])
    for index, value in enumerate(outputs.values()):
        assert value.shape == (2, samples)
        np.testing.assert_array_equal(value, expected[index])


def test_inference_import_does_not_load_evaluation_dependencies():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import subroformer.inference.engine; "
         "assert all(name not in sys.modules for name in ('subroformer.training.evaluate', 'museval', 'accelerate'))"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("audio,segment,hop,batch", [
    (np.zeros((2, 0)), 8, 4, 1),
    (np.full((2, 10), np.nan), 8, 4, 1),
    (np.zeros((2, 10)), 8, 0, 1),
    (np.zeros((2, 10)), 8, 9, 1),
    (np.zeros((2, 10)), 8, 4, 0),
])
def test_overlap_add_rejects_invalid_input(audio, segment, hop, batch):
    with pytest.raises(ValueError):
        separate_overlap_add(torch.nn.Linear(1, 1), audio, segment, hop, batch)


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("safe", [False, True])
def test_file_inference_resamples_mono_and_preserves_duration(tmp_path: Path, joint: bool, safe: bool):
    config = tiny_config(target_instrument=None if joint else "vocals")
    config.model.num_stems = 4 if joint else 1
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    checkpoint_path = tmp_path / "model.pt"
    state = build_model(config).state_dict()
    torch.save({"decay": 0.999, "model": state} if joint else state, checkpoint_path)
    if safe:
        checkpoint_path = export_safetensors(checkpoint_path, tmp_path / "model.safetensors")
    input_path = tmp_path / "input.wav"
    sf.write(input_path, np.ones(40, dtype=np.float32) * 0.01, 800, subtype="FLOAT")

    paths = separate_file(
        str(config_path), str(checkpoint_path), str(input_path), str(tmp_path / "output")
    )

    assert [path.stem for path in paths] == (STEMS if joint else ["vocals"])
    for path in paths:
        audio, sample_rate = sf.read(path, always_2d=True)
        assert sample_rate == 1600
        assert audio.shape == (80, 2)
        assert np.isfinite(audio).all()
    with pytest.raises(FileExistsError):
        separate_file(
            str(config_path), str(checkpoint_path), str(input_path), str(tmp_path / "output")
        )


@pytest.mark.parametrize("hop_ratio,batch_size", [(0, 1), (129, 1), (4, 0)])
def test_file_inference_rejects_invalid_window_settings(tmp_path, hop_ratio, batch_size):
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(tiny_config(), config_path)
    with pytest.raises(ValueError):
        separate_file(
            str(config_path), "missing.pt", "missing.wav", str(tmp_path / "output"),
            hop_ratio=hop_ratio, batch_size=batch_size,
        )


def test_training_and_inference_cli_without_private_dependencies(tmp_path):
    from tests.helpers import make_dataset

    config = tiny_config(num_steps=1, target_instrument=None)
    config.model.num_stems = 4
    config.training.save_every_updates = 1
    config_path = tmp_path / "config.yaml"
    OmegaConf.save(config, config_path)
    data_root = make_dataset(tmp_path / "train", songs=2)
    output_root = tmp_path / "training"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "subroformer.training.train", "--config", str(config_path),
         "--data_path", str(data_root), "--output_dir", str(output_root),
         "--max_updates", "1", "--eval_every_updates", "0"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "update=1/1" in result.stdout
    metrics = re.search(r"loss=([\d.eE+-]+) grad_norm=([\d.eE+-]+)", result.stdout)
    assert metrics is not None
    assert all(np.isfinite(float(value)) for value in metrics.groups())
    saved_state = json.loads((output_root / "step_1" / "training_state.json").read_text())
    assert saved_state["global_update"] == 1
    raw_state = torch.load(output_root / "model.pt", weights_only=True)
    ema_state = torch.load(output_root / "model_ema.pt", weights_only=True)
    assert all(torch.isfinite(value).all() for value in raw_state.values())
    assert any(not torch.equal(value, ema_state[name]) for name, value in raw_state.items())
    result = subprocess.run(
        [sys.executable, "-m", "subroformer.inference", "--config", str(config_path),
         "--checkpoint", str(output_root / "model_ema.pt"),
         "--input", str(data_root / "song_00" / "mixture.wav"),
         "--output_dir", str(tmp_path / "separated"), "--device", "cpu"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / "separated").glob("*.wav"))) == 4
