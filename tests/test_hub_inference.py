from pathlib import Path
import sys

import pytest
import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

import inference
from checkpoint_utils import export_safetensors
from model_utils import build_model
from tests.helpers import STEMS, tiny_config


@pytest.mark.parametrize("config_exists,checkpoint_exists", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_resolve_downloads_only_missing_files(tmp_path, monkeypatch, config_exists, checkpoint_exists):
    config = tmp_path / "custom.yaml"
    checkpoint = tmp_path / "custom.safetensors"
    for path, exists in ((config, config_exists), (checkpoint, checkpoint_exists)):
        if exists:
            path.write_text("local")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path / "cache" / kwargs["filename"])

    monkeypatch.setattr(inference, "hf_hub_download", download)
    resolved = inference.resolve_pretrained_paths(
        config, checkpoint, repo_id="owner/model", revision="commit",
        cache_dir=tmp_path / "cache", local_files_only=True,
    )
    expected_downloads = [
        filename for filename, exists in (
            (inference.DEFAULT_CONFIG, config_exists),
            (inference.DEFAULT_CHECKPOINT, checkpoint_exists),
        ) if not exists
    ]
    assert [call["filename"] for call in calls] == expected_downloads
    assert all(call["repo_id"] == "owner/model" and call["revision"] == "commit"
               and call["local_files_only"] and call["cache_dir"] == tmp_path / "cache"
               for call in calls)
    assert resolved == (
        config if config_exists else tmp_path / "cache" / inference.DEFAULT_CONFIG,
        checkpoint if checkpoint_exists else tmp_path / "cache" / inference.DEFAULT_CHECKPOINT,
    )
    for path, exists in ((config, config_exists), (checkpoint, checkpoint_exists)):
        if exists:
            assert path.read_text() == "local"


def test_default_paths_are_local_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for filename in (inference.DEFAULT_CONFIG, inference.DEFAULT_CHECKPOINT):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(inference, "hf_hub_download", lambda **kwargs: pytest.fail("Unexpected download"))
    assert inference.resolve_pretrained_paths() == (
        Path(inference.DEFAULT_CONFIG), Path(inference.DEFAULT_CHECKPOINT),
    )


def test_directory_is_not_treated_as_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Expected a file"):
        inference.resolve_pretrained_paths(tmp_path, None)


def test_download_failures_are_not_hidden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def download(**kwargs):
        raise RuntimeError("Authentication required")

    monkeypatch.setattr(inference, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="Authentication required"):
        inference.resolve_pretrained_paths()


def test_download_progress_uses_file_bytes_and_ignores_xet_transfer_bar(tmp_path, monkeypatch):
    local_config = tmp_path / "local.yaml"
    local_config.write_bytes(b"config")
    cached = tmp_path / "cached.safetensors"
    cached.write_bytes(b"checkpoint")
    updates = []

    def download(**kwargs):
        with kwargs["tqdm_class"](total=10, initial=2, disable=True) as bar:
            bar.update(3)
            bar.update(5)
        with kwargs["tqdm_class"](total=50, desc="checkpoint: downloading bytes") as bar:
            bar.update(50)
        return str(cached)

    monkeypatch.setattr(inference, "hf_hub_download", download)
    result = inference.resolve_pretrained_paths(local_config, tmp_path / "missing", download_progress=lambda *values: updates.append(values))
    assert result == (local_config, cached)
    assert updates == [(inference.DEFAULT_CHECKPOINT, completed, total) for completed, total in [(0, None), (2, 10), (5, 10), (10, 10), (10, 10)]]


def test_offline_cache_miss_does_not_download(tmp_path):
    from huggingface_hub.errors import LocalEntryNotFoundError

    with pytest.raises(LocalEntryNotFoundError):
        inference.resolve_pretrained_paths(tmp_path / "missing.yaml", tmp_path / "missing.pt", cache_dir=tmp_path / "empty-cache", local_files_only=True)


def test_failed_download_does_not_report_completion(tmp_path, monkeypatch):
    updates = []

    def download(**kwargs):
        with kwargs["tqdm_class"](total=100) as bar:
            bar.update(20)
        raise RuntimeError("network interrupted")

    monkeypatch.setattr(inference, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="network interrupted"):
        inference.resolve_pretrained_paths(tmp_path / "missing.yaml", tmp_path / "missing.pt", download_progress=lambda *values: updates.append(values))
    assert updates[-1] == (inference.DEFAULT_CONFIG, 20, 100)


def test_default_checkpoint_is_fp16():
    assert inference.DEFAULT_CONFIG == "configs/sbroformer_base_4stem_silu.yaml"
    assert inference.DEFAULT_CHECKPOINT == "checkpoints/subroformer_base_4stem_1400000_ema_perceptual_finetuned_200000_fp16.safetensors"
    config = inference.load_config(Path(__file__).resolve().parents[1] / inference.DEFAULT_CONFIG)
    assert config.model.dsp_backend == "dsp3"
    assert config.model.fusion_type == "silu"


@pytest.mark.parametrize("backend", [None, "dsp", "dsp2", "dsp3"])
def test_inference_dsp_default_and_explicit_override(tmp_path, backend):
    config = tiny_config()
    if backend is not None:
        config.model.dsp_backend = backend
    path = tmp_path / "config.yaml"
    OmegaConf.save(config, path)
    resolved = inference.load_inference_config(path)
    assert resolved.model.dsp_backend == (backend or "dsp3")
    assert inference.load_config(path) == config


def test_inference_dsp_default_preserves_global_frontend(tmp_path):
    config = tiny_config()
    config.model.frontend_type = "global_stft_bs"
    path = tmp_path / "config.yaml"
    OmegaConf.save(config, path)
    assert "dsp_backend" not in inference.load_inference_config(path).model


def test_cli_config_and_checkpoint_are_optional(monkeypatch):
    calls = []
    monkeypatch.setattr(inference, "separate_file", lambda *args, **kwargs: calls.append((args, kwargs)) or [])
    monkeypatch.setattr(sys, "argv", [
        "inference.py", "--input", "song.wav", "--output_dir", "stems",
        "--revision", "commit", "--local_files_only",
    ])
    inference.main()
    args, kwargs = calls[0]
    assert args[:4] == (None, None, "song.wav", "stems")
    assert kwargs["revision"] == "commit"
    assert kwargs["local_files_only"]


def test_downloaded_artifacts_run_file_inference(tmp_path, monkeypatch):
    config = tiny_config()
    config.model.num_stems = 4
    cached_config = tmp_path / "cached.yaml"
    OmegaConf.save(config, cached_config)
    source = tmp_path / "source.pt"
    torch.save(build_model(config).state_dict(), source)
    cached_checkpoint = export_safetensors(source, tmp_path / "cached.safetensors")
    downloads = {
        inference.DEFAULT_CONFIG: str(cached_config),
        inference.DEFAULT_CHECKPOINT: str(cached_checkpoint),
    }
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(inference, "hf_hub_download", lambda **kwargs: downloads[kwargs["filename"]])
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, np.zeros((160, 2), dtype=np.float32), 1600)
    paths = inference.separate_file(None, None, str(audio_path), str(tmp_path / "output"))
    assert [path.stem for path in paths] == STEMS
    for path in paths:
        audio, sample_rate = sf.read(path)
        assert audio.shape == (160, 2) and sample_rate == 1600
        assert np.isfinite(audio).all()
