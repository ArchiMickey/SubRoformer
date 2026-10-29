"""Training reads must deliver audio at config.audio.sample_rate."""

import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch

import subroformer.training.dataset as dataset_module
from subroformer.training.dataset import load_chunk, prepare_data, source_frames_for
from tests.helpers import STEMS, tiny_config

STEM_FREQS = {"vocals": 1000.0, "bass": 220.0, "drums": 3000.0, "other": 5500.0}


def _write(path: Path, audio: np.ndarray, sample_rate: int) -> Path:
    sf.write(path, audio, sample_rate, subtype="FLOAT")
    return path


def _sine(freq, sample_rate, frames, amplitude=0.5):
    t = np.arange(frames, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(2 * np.pi * freq * t)
    return np.stack([mono, 0.8 * mono], axis=-1).astype(np.float32)


def _noise(sample_rate, frames, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((frames, 2)).astype(np.float32) * 0.1
    # Keep content below both Nyquist limits so resampler choice is not the test.
    spec = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(frames, 1.0 / sample_rate)
    spec[freqs > 18000] = 0
    return np.fft.irfft(spec, n=frames, axis=0).astype(np.float32)


def _peak_hz(x, sample_rate):
    mono = x[0].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    return np.argmax(spectrum) * sample_rate / mono.size


def _full_resample(path, target_sr):
    # Same call as subroformer.inference.engine.load_audio (evaluation path).
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    return librosa.resample(audio.T, orig_sr=source_rate, target_sr=target_sr, axis=-1)


def test_load_chunk_resamples_and_preserves_frequency(tmp_path):
    path = _write(tmp_path / "sine.wav", _sine(1000.0, 44100, 3 * 44100), 44100)
    length = sf.info(path).frames

    np.random.seed(0)
    chunk = load_chunk(str(path), length, 48000, target_sr=48000)
    assert chunk.shape == (2, 48000) and chunk.dtype == np.float32
    assert abs(_peak_hz(chunk, 48000) - 1000.0) <= 1.0

    # The pre-fix read path plays the 44.1 kHz file 8.8% fast at 48 kHz.
    raw = load_chunk(str(path), length, 48000, offset=0)
    assert abs(_peak_hz(raw, 48000) - 1000.0 * 48000 / 44100) <= 1.0


@pytest.mark.parametrize("offset", [0, 147, 44100, 88200 - 147, 88200])
def test_resampled_chunk_matches_whole_file_resample(tmp_path, offset):
    # 3 s at 44.1 kHz; offsets are source frames on the 147-frame grid, so the
    # chunk starts on an output sample of the whole-file resample.
    path = _write(tmp_path / "noise.wav", _noise(44100, 3 * 44100), 44100)
    length = sf.info(path).frames
    chunk_size = 48000
    assert source_frames_for(chunk_size, 44100, 48000) == 44100

    chunk = load_chunk(str(path), length, chunk_size, offset=offset, target_sr=48000)
    full = _full_resample(path, 48000)
    start = offset * 160 // 147
    expected = full[:, start:start + chunk_size]
    # soxr's own start-of-stream transient differs from zero padding in the
    # first few output samples of the file; elsewhere the chunk is exact.
    skip = 8 if offset == 0 else 0
    np.testing.assert_allclose(chunk[:, skip:], expected[:, skip:], atol=2e-6, rtol=0)
    np.testing.assert_allclose(chunk[:, :skip], expected[:, :skip], atol=5e-3, rtol=0)


def test_chunk_running_past_file_end_is_zero_padded(tmp_path):
    path = _write(tmp_path / "noise.wav", _noise(44100, 2 * 44100), 44100)
    length = sf.info(path).frames
    offset = length - 14700  # 14700 source frames left -> 16000 output samples
    chunk = load_chunk(str(path), length, 48000, offset=offset, target_sr=48000)
    full = _full_resample(path, 48000)
    start = offset * 160 // 147

    assert chunk.shape == (2, 48000)
    np.testing.assert_allclose(chunk[:, :15800], full[:, start:start + 15800], atol=2e-6, rtol=0)
    assert np.all(chunk[:, 16000:] == 0)


def test_file_shorter_than_chunk(tmp_path):
    path = _write(tmp_path / "short.wav", _noise(44100, 22050), 44100)
    chunk = load_chunk(str(path), 22050, 48000, target_sr=48000)
    full = _full_resample(path, 48000)

    assert chunk.shape == (2, 48000)
    np.testing.assert_allclose(chunk[:, 8:23900], full[:, 8:23900], atol=2e-6, rtol=0)
    assert np.all(chunk[:, 24000:] == 0)


def test_downsampling_and_mono(tmp_path):
    mono = _sine(1000.0, 96000, 2 * 96000)[:, 0]
    path = _write(tmp_path / "mono96k.wav", mono, 96000)
    np.random.seed(1)
    chunk = load_chunk(str(path), sf.info(path).frames, 48000, target_sr=48000)

    assert chunk.shape == (2, 48000)
    np.testing.assert_array_equal(chunk[0], chunk[1])
    assert abs(_peak_hz(chunk, 48000) - 1000.0) <= 1.0


@pytest.mark.parametrize("frames", [3 * 48000, 20000])
@pytest.mark.parametrize("offset", [None, 0, 1234])
def test_matching_rate_is_bit_identical(tmp_path, frames, offset):
    path = str(_write(tmp_path / "noise48k.wav", _noise(48000, frames), 48000))
    for seed in range(3):
        np.random.seed(seed)
        expected = load_chunk(path, frames, 48000, offset=offset)
        expected_next = np.random.randint(1 << 30)
        np.random.seed(seed)
        actual = load_chunk(path, frames, 48000, offset=offset, target_sr=48000)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        assert np.random.randint(1 << 30) == expected_next  # same RNG consumption


def _make_folder_dataset(root: Path, sample_rate: int, seconds: float, songs: int = 2) -> Path:
    root.mkdir(parents=True)
    frames = int(seconds * sample_rate)
    for song in range(songs):
        song_dir = root / f"song_{song}"
        song_dir.mkdir()
        mixture = np.zeros((frames, 2), dtype=np.float32)
        for stem in STEMS:
            audio = _sine(STEM_FREQS[stem] + 10 * song, sample_rate, frames, amplitude=0.2)
            _write(song_dir / f"{stem}.wav", audio, sample_rate)
            mixture += audio
        _write(song_dir / "mixture.wav", mixture, sample_rate)
    return root


def _loader(tmp_path, file_sr, dataset_type=1, random_mixing=False, time_align="random",
            seconds=1.5, chunk_size=24000):
    data_root = _make_folder_dataset(tmp_path / "train", file_sr, seconds)
    config = tiny_config(target_instrument=None)
    config.audio.sample_rate = 48000
    config.audio.chunk_size = chunk_size
    config.augmentations.random_mixing.enable = random_mixing
    config.augmentations.random_mixing.min_sources = 1
    config.augmentations.random_mixing.max_sources = 1
    config.augmentations.random_mixing.time_align = time_align
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    args = SimpleNamespace(
        data_path=[str(data_root)], results_path=str(results), dataset_type=dataset_type,
        num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2, seed=3,
    )
    return prepare_data(config, args, batch_size=1)


@pytest.mark.parametrize(
    "dataset_type,random_mixing,time_align",
    [(1, False, "random"), (1, True, "random"), (1, True, "strict"), (4, False, "random"),
     (5, False, "random"), (6, False, "random")],
)
def test_dataset_paths_deliver_target_rate(tmp_path, dataset_type, random_mixing, time_align):
    loader = _loader(tmp_path, 44100, dataset_type, random_mixing, time_align)
    dataset = loader.dataset
    for index in range(3):
        np.random.seed(index)
        item = next(iter(loader))
        stems, mixture = item[0][0], item[1][0]
        assert stems.shape == (4, 2, 24000)
        assert mixture.shape == (2, 24000)
        for stem_index, stem in enumerate(STEMS):
            peak = _peak_hz(stems[stem_index].numpy(), 48000)
            assert min(abs(peak - (STEM_FREQS[stem] + 10 * song)) for song in range(2)) <= 2.0, (stem, peak)
        if dataset_type == 6:  # mixture.wav is read and resampled at the same offset
            torch.testing.assert_close(mixture, stems.sum(0), atol=1e-4, rtol=0)
    assert dataset._source_chunk_size(dataset.metadata[0][0]) == 22050


def test_matching_rate_dataset_items_are_unchanged(tmp_path):
    loader = _loader(tmp_path, 48000, dataset_type=4)
    dataset = loader.dataset
    original = dataset_module.load_chunk

    def pre_fix_load_chunk(*args, target_sr=None, **kwargs):
        return original(*args, **kwargs)

    for seed in range(4):
        np.random.seed(seed)
        random.seed(seed)
        with patch.object(dataset_module, "load_chunk", pre_fix_load_chunk):
            reference = dataset[0]
        np.random.seed(seed)
        random.seed(seed)
        actual = dataset[0]
        for a, r in zip(actual, reference):
            torch.testing.assert_close(a, r, atol=0, rtol=0)


def test_chunk_cache_without_sample_rate_is_recomputed(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    stale = {
        "chunks_metadata": [("stale", 0)],
        "config": {"chunk_size": 24000, "min_mean_abs": 0.0, "instruments": sorted(STEMS)},
    }
    with open(results / "metadata_5_chunks.pkl", "wb") as handle:
        pickle.dump(stale, handle)

    dataset = _loader(tmp_path, 44100, dataset_type=5).dataset

    assert ("stale", 0) not in dataset.chunks_metadata
    # 1.5 s at 44.1 kHz = 66150 frames, 22050-frame chunks with 50% overlap.
    assert sorted(offset for _, offset in dataset.chunks_metadata) == sorted([0, 11025, 22050, 33075, 44100] * 2)
    with open(results / "metadata_5_chunks.pkl", "rb") as handle:
        assert pickle.load(handle)["config"]["sample_rate"] == 48000
