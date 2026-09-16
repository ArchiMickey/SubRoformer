from types import SimpleNamespace

import numpy as np
import pytest
import torch
import soundfile as sf
from omegaconf import OmegaConf

import coreml_inference
import inference
from model_utils import build_model
from dsp2 import SubbandFilter
from dsp.inference_optimized import InferenceSubbandFilter
from tests.helpers import tiny_config


def test_default_artifact_is_fp16():
    assert coreml_inference.COREML_DIRECTORY == "coreml/base-4stem-perceptual-200k-fp16-v1"


@pytest.mark.parametrize("backend,expected", [(None, SubbandFilter), ("dsp2", SubbandFilter), ("dsp", InferenceSubbandFilter)])
def test_coreml_defaults_to_dsp2_without_overriding_explicit_backend(backend, expected):
    config = tiny_config()
    if backend is not None:
        config.model.dsp_backend = backend
    model = coreml_inference.CoreMLSeparator(None, config)
    assert isinstance(model.dsp.sb_filter, expected)
    assert config.model.get("dsp_backend") == backend


def test_export_preserves_input_block_and_norm_precision():
    from export_coreml import use_fp16_for_op

    for module in ("RMSNorm", "RMSNorm2d", "GRN", "TimeMixBlock"):
        assert not use_fp16_for_op(SimpleNamespace(scopes={"module": [module]}))
    assert use_fp16_for_op(SimpleNamespace(scopes={"module": ["blocks_0", "Linear"]}))


def test_snapshot_reports_bytes_not_file_counts_or_transfer_bytes(tmp_path, monkeypatch):
    directory = tmp_path / coreml_inference.COREML_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "data").write_bytes(bytes(100))

    def download(**kwargs):
        progress = kwargs["tqdm_class"]
        with progress(total=7, desc="Fetching 7 files") as bar:
            bar.update(1)
        with progress(total=100, unit="B", desc="Downloading bytes") as bar:
            bar.update(80)
        with progress(total=100, unit="B", desc="Reconstructing (incomplete total...)", disable=True) as bar:
            bar.update(25)
            bar.update(50)
        return str(tmp_path)

    monkeypatch.setattr(coreml_inference, "snapshot_download", download)
    updates = []
    assert coreml_inference.resolve_coreml_artifact(download_progress=lambda *args: updates.append(args)) == directory
    assert updates == [("Core ML model", 0, None), ("Core ML model", 25, 100),
                       ("Core ML model", 75, 100), ("Core ML model", 100, 100)]


@pytest.mark.parametrize("module,expected", [("RMSNorm", False), ("RMSNorm2d", False), ("GRN", False), ("Linear", True), ("Conv2d", True)])
def test_export_preserves_normalization_precision(module, expected):
    from export_coreml import use_fp16_for_op

    assert use_fp16_for_op(SimpleNamespace(scopes={"module_type": ["MaskNetwork", module]})) is expected


def test_export_waveform_validation():
    from export_coreml import validate_waveform

    relative, absolute = validate_waveform(np.zeros(16), np.full(16, 2e-7))
    assert absolute == pytest.approx(2e-7)
    assert relative > 0.001
    validate_waveform(np.ones(16), np.full(16, 1.0005))
    for expected, actual in ((np.zeros(16), np.full(16, 2e-6)),
                             (np.ones(16), np.full(16, 1.01)),
                             (np.ones(16), np.full(16, np.nan))):
        with pytest.raises(RuntimeError):
            validate_waveform(expected, actual)


@pytest.mark.parametrize("precision", ["auto", "fp16", "fp32", "bf16"])
@pytest.mark.parametrize("device", ["auto", "cpu", "mps"])
def test_cli_coreml_precision(tmp_path, monkeypatch, precision, device):
    config = tiny_config()
    config.model.num_stems = 4
    monkeypatch.setattr(coreml_inference, "use_coreml", lambda *args: True)
    calls = []
    monkeypatch.setattr(coreml_inference, "load_coreml_separator", lambda **kwargs:
                        calls.append(kwargs) or (build_model(config).eval(), config))
    monkeypatch.setattr(inference, "resolve_amp_dtype", lambda *args: pytest.fail("Core ML DSP must not use autocast"))
    audio_path = tmp_path / "audio.wav"
    sf.write(audio_path, np.zeros((128, 2), dtype=np.float32), 1600)
    if precision in {"fp32", "bf16"}:
        with pytest.raises(ValueError, match="compiled FP16"):
            inference.separate_file(None, None, str(audio_path), str(tmp_path / "out"), precision=precision, device=device)
    else:
        paths = inference.separate_file(None, None, str(audio_path), str(tmp_path / "out"), precision=precision, device=device)
        assert len(paths) == 4 and all(path.is_file() for path in paths)
        assert calls[0]["dsp_device"] == ("mps" if device == "mps" else "cpu")


@pytest.mark.parametrize("system,version,device,custom,expected", [
    ("darwin", "15.0", "auto", False, False),
    ("darwin", "14.7", "auto", False, False),
    ("linux", "", "auto", False, False),
    ("win32", "", "auto", False, False),
    ("darwin", "26.0", "cpu", False, False),
    ("darwin", "26.0", "mps", False, False),
    ("darwin", "26.0", "auto", True, False),
])
def test_automatic_backend(monkeypatch, system, version, device, custom, expected):
    monkeypatch.setattr(coreml_inference, "sys", SimpleNamespace(platform=system))
    monkeypatch.setattr(coreml_inference.platform, "mac_ver", lambda: (version, (), ""))
    assert coreml_inference.use_coreml("auto", device, "custom.yaml" if custom else None) is expected
    assert not coreml_inference.use_coreml("torch", device)


def test_explicit_backend_errors(monkeypatch):
    monkeypatch.setattr(coreml_inference, "sys", SimpleNamespace(platform="linux"))
    with pytest.raises(ValueError, match="macOS"):
        coreml_inference.use_coreml("coreml", "auto")
    with pytest.raises(ValueError, match="backend"):
        coreml_inference.use_coreml("unknown", "auto")
    monkeypatch.setattr(coreml_inference, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(coreml_inference.platform, "mac_ver", lambda: ("15.0", (), ""))
    assert coreml_inference.use_coreml("coreml", "auto")
    with pytest.raises(ValueError, match="custom"):
        coreml_inference.use_coreml("coreml", "auto", checkpoint_path="custom.pt")


@pytest.mark.parametrize("dsp_device", ["cpu", "mps"])
def test_hybrid_waveform_and_batched_streaming(dsp_device):
    if dsp_device == "mps" and (not torch.backends.mps.is_available()
                               or torch.torch_version.TorchVersion(torch.__version__) < "2.14"):
        pytest.skip("MPS DSP requires PyTorch 2.14 and MPS hardware")
    config = tiny_config()
    config.model.num_stems = 4
    config.model.dsp_backend = "dsp2"
    reference = build_model(config).eval()
    reference.sb_filter.w.mul_(0.99)
    calls = []

    def predict(inputs):
        features = torch.from_numpy(inputs["features"])
        calls.append(features.shape[0])
        return {"mask": reference.predict_mask(features).numpy()}

    model = coreml_inference.CoreMLSeparator(SimpleNamespace(predict=predict), config,
                                           reference.sb_filter.state_dict(), dsp_device=dsp_device).eval()
    assert next(model.parameters()).device.type == "cpu"
    assert model.dsp.sb_filter.w.device.type == dsp_device
    audio = np.random.default_rng(0).normal(0, 0.05, (2, 279)).astype(np.float32)
    expected = inference.separate_audio(reference, audio, config, batch_size=2)
    chunks = []
    actual = inference.separate_audio(model, audio, config, batch_size=3, on_chunk=lambda start, chunk: chunks.append(chunk))
    assert calls and set(calls) == {1}
    for stem in config.training.instruments:
        np.testing.assert_allclose(actual[stem], expected[stem], atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(np.concatenate([chunk[stem] for chunk in chunks], axis=-1), actual[stem])
    with pytest.raises(ValueError, match="chunk size"):
        model(torch.zeros(1, 2, 17))


@pytest.mark.parametrize("dsp_device", ["cpu", "mps"])
def test_snapshot_loading_without_checkpoint(tmp_path, monkeypatch, dsp_device):
    import sys

    directory = tmp_path / coreml_inference.COREML_DIRECTORY
    directory.mkdir(parents=True)
    OmegaConf.save(tiny_config(), directory / "config.yaml")
    calls = []
    compiled = object()
    monkeypatch.setattr(coreml_inference, "snapshot_download", lambda **kwargs: calls.append(kwargs) or str(tmp_path))
    devices = []
    monkeypatch.setattr(coreml_inference, "CoreMLSeparator", lambda model, config, state, **kwargs:
                        devices.append(kwargs["dsp_device"]) or SimpleNamespace(eval=lambda: model))
    monkeypatch.setattr(coreml_inference, "load_file", lambda path: {})
    monkeypatch.setitem(sys.modules, "coremltools", SimpleNamespace(
        models=SimpleNamespace(CompiledMLModel=lambda path, **kwargs: compiled),
        ComputeUnit=SimpleNamespace(CPU_AND_GPU="cpu_gpu"),
    ))
    progress = []
    model, _ = coreml_inference.load_coreml_separator(
        repo_id="owner/repo", revision="commit", cache_dir=tmp_path, local_files_only=True,
        download_progress=lambda *args: progress.append(args),
        dsp_device=dsp_device,
    )
    assert devices == [dsp_device]
    assert model is compiled
    assert calls[0]["local_files_only"] and calls[0]["revision"] == "commit"
    assert calls[0]["repo_id"] == "owner/repo"
    assert calls[0]["allow_patterns"] == [f"{coreml_inference.COREML_DIRECTORY}/config.yaml", f"{coreml_inference.COREML_DIRECTORY}/dsp.safetensors", f"{coreml_inference.COREML_DIRECTORY}/mask.mlmodelc/**"]
    assert progress[0][2] is None and progress[-1][1] == progress[-1][2]


@pytest.mark.parametrize("device,expected", [("auto", "cpu"), ("cpu", "cpu"), ("mps", "mps")])
def test_load_separator_routes_without_pytorch_download(monkeypatch, device, expected):
    monkeypatch.setattr(coreml_inference, "use_coreml", lambda *args: True)
    calls = []
    monkeypatch.setattr(coreml_inference, "load_coreml_separator", lambda **kwargs: calls.append(kwargs) or ("compiled", "config"))
    monkeypatch.setattr(inference, "resolve_pretrained_paths", lambda *args, **kwargs: pytest.fail("Checkpoint download"))
    assert inference.load_separator(device=device) == ("compiled", "config")
    assert calls[0]["dsp_device"] == expected


def test_dsp_device_validation(monkeypatch):
    with pytest.raises(ValueError, match="cpu or mps"):
        coreml_inference.CoreMLSeparator(None, None, dsp_device="cuda")
    monkeypatch.setattr(torch, "__version__", "2.7.0")
    with pytest.raises(ValueError, match="2.14"):
        coreml_inference.CoreMLSeparator(None, None, dsp_device="mps")
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="available MPS"):
        coreml_inference.CoreMLSeparator(None, None, dsp_device="mps")
