from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

import evaluate
from evaluate import (
    evaluate_songs,
    evaluate_songs_distributed,
    extract_model_state,
    normalize_model_output,
    order_distributed_rows,
    select_subset_songs,
    separate_overlap_add,
    summarize_evaluation_rows,
)
from train import build_eval_metrics
from training_utils import museval_sdr


def test_subset_selection_matches_mini_source_separation_spacing(tmp_path: Path):
    split_root = tmp_path / "train"
    split_root.mkdir()
    for index in range(100):
        (split_root / f"song_{index:03d}").mkdir()

    selected = select_subset_songs(tmp_path, split="train", audios_num=5)

    assert [path.name for path in selected] == [
        "song_000",
        "song_020",
        "song_040",
        "song_060",
        "song_080",
    ]


def test_museval_sdr_is_finite():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((2, 44_100)).astype(np.float32)
    output = target + 0.1 * rng.standard_normal(target.shape).astype(np.float32)

    score = museval_sdr(output, target, sample_rate=44_100)

    assert np.isfinite(score)


def test_extract_model_state_supports_nested_training_checkpoint():
    model_state = {"weight": torch.ones(2)}
    checkpoint = {"epoch": 9, "model_state_dict": model_state}

    assert extract_model_state(checkpoint) is model_state


def test_normalize_model_output_squeezes_single_stem_axis():
    estimate = torch.zeros(2, 1, 2, 128)

    normalized = normalize_model_output(estimate)

    assert normalized.shape == (2, 2, 128)


def test_normalize_model_output_preserves_joint_stem_axis():
    estimate = torch.zeros(2, 4, 2, 128)

    normalized = normalize_model_output(estimate)

    assert normalized.shape == (2, 4, 2, 128)


def test_overlap_add_preserves_joint_stem_axis():
    class JointSeparator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, audio):
            return audio[:, None].expand(-1, 4, -1, -1)

    rng = np.random.default_rng(1)
    audio = rng.standard_normal((2, 20)).astype(np.float32)

    estimate = separate_overlap_add(
        JointSeparator(),
        audio,
        segment_samples=8,
        hop_length=4,
        batch_size=2,
    )

    assert estimate.shape == (4, 2, 20)
    np.testing.assert_allclose(estimate, np.broadcast_to(audio, estimate.shape), atol=1e-6)


def test_joint_evaluation_records_each_stem_and_stem_average(monkeypatch):
    stems = ["vocals", "bass", "drums", "other"]
    config = OmegaConf.create(
        {
            "audio": {"sample_rate": 44_100, "chunk_size": 8},
            "training": {"target_instrument": None, "instruments": stems},
        }
    )
    scores = iter([1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0])
    monkeypatch.setattr(evaluate, "load_audio", lambda path, sample_rate: np.zeros((2, 8)))
    monkeypatch.setattr(
        evaluate,
        "separate_overlap_add",
        lambda *args, **kwargs: np.zeros((4, 2, 8)),
    )
    monkeypatch.setattr(evaluate, "museval_sdr", lambda *args, **kwargs: next(scores))

    result = evaluate_songs(
        config=config,
        model=torch.nn.Identity(),
        songs=[Path("song_a"), Path("song_b")],
        hop_ratio=1,
        batch_size=1,
        split="test",
    )

    assert result["stem_median_sdr"] == {
        "vocals": 2.0,
        "bass": 3.0,
        "drums": 4.0,
        "other": 5.0,
    }
    assert result["median_sdr"] == 3.5
    assert result["results"][0]["sdr"] == dict(zip(stems, [1.0, 2.0, 3.0, 4.0]))


def test_evaluation_uses_reference_sum_not_mixture_file(tmp_path, monkeypatch):
    stems = ["vocals", "bass", "drums", "other"]
    config = OmegaConf.create({
        "audio": {"sample_rate": 44_100, "chunk_size": 8},
        "training": {"target_instrument": None, "instruments": stems},
    })
    source = np.full((8, 2), 0.125, dtype=np.float32)
    sf.write(tmp_path / "mixture.wav", np.zeros_like(source), 44_100, subtype="FLOAT")
    for stem in stems:
        sf.write(tmp_path / f"{stem}.wav", source, 44_100, subtype="FLOAT")

    def separate(model, audio, *args, **kwargs):
        np.testing.assert_array_equal(audio, len(stems) * source.T)
        return np.zeros((4, 2, 8))

    monkeypatch.setattr(evaluate, "separate_overlap_add", separate)
    monkeypatch.setattr(evaluate, "museval_sdr", lambda *args: 1.0)
    result = evaluate_songs(config, torch.nn.Identity(), [tmp_path], 1, 1)
    assert result["songs"] == 1


def test_distributed_rows_are_restored_to_song_order():
    songs = [Path("song_a"), Path("song_b"), Path("song_c")]
    gathered_rows = [
        {"song": "song_a", "sdr": 1.0},
        {"song": "song_c", "sdr": 3.0},
        {"song": "song_b", "sdr": 2.0},
    ]

    rows = order_distributed_rows(gathered_rows, songs)

    assert [row["song"] for row in rows] == ["song_a", "song_b", "song_c"]


def test_distributed_rows_reject_missing_results():
    songs = [Path("song_a"), Path("song_b")]

    with np.testing.assert_raises_regex(ValueError, "missing=\\['song_b'\\]"):
        order_distributed_rows([{"song": "song_a", "sdr": 1.0}], songs)


def test_distributed_rows_reject_duplicate_results():
    songs = [Path("song_a"), Path("song_b")]
    rows = [
        {"song": "song_a", "sdr": 1.0},
        {"song": "song_a", "sdr": 1.0},
    ]

    with np.testing.assert_raises_regex(ValueError, "duplicate"):
        order_distributed_rows(rows, songs)


def test_summarize_evaluation_rows_uses_global_song_median():
    config = OmegaConf.create(
        {
            "training": {
                "target_instrument": "vocals",
                "instruments": ["vocals", "bass", "drums", "other"],
            }
        }
    )
    rows = [
        {"song": "song_a", "sdr": 1.0},
        {"song": "song_b", "sdr": 100.0},
        {"song": "song_c", "sdr": 2.0},
    ]

    result = summarize_evaluation_rows(config, rows, hop_ratio=4)

    assert result["median_sdr"] == 2.0
    assert result["songs"] == 3


def test_distributed_evaluation_splits_and_aggregates_all_songs(monkeypatch):
    class FakeAccelerator:
        num_processes = 2
        process_index = 0

        @staticmethod
        def unwrap_model(model):
            return model

    config = OmegaConf.create(
        {
            "audio": {"sample_rate": 44_100, "chunk_size": 8},
            "training": {
                "target_instrument": "vocals",
                "instruments": ["vocals", "bass", "drums", "other"],
            },
        }
    )
    songs = [Path("song_a"), Path("song_b"), Path("song_c")]
    monkeypatch.setattr(evaluate, "load_audio", lambda path, sample_rate: np.zeros((2, 8)))
    monkeypatch.setattr(
        evaluate,
        "separate_overlap_add",
        lambda *args, **kwargs: np.zeros((2, 8)),
    )
    local_scores = iter([1.0, 2.0])
    monkeypatch.setattr(evaluate, "museval_sdr", lambda *args, **kwargs: next(local_scores))

    def fake_gather_object(local_rows):
        assert [row["song"] for row in local_rows] == ["song_a", "song_c"]
        return local_rows + [{"song": "song_b", "sdr": 100.0}]

    monkeypatch.setattr(evaluate, "gather_object", fake_gather_object)

    result = evaluate_songs_distributed(
        config=config,
        model=torch.nn.Identity(),
        songs=songs,
        hop_ratio=1,
        batch_size=1,
        accelerator=FakeAccelerator(),
        split="test",
    )

    assert result["songs"] == 3
    assert result["median_sdr"] == 2.0
    assert [row["song"] for row in result["results"]] == [
        "song_a",
        "song_b",
        "song_c",
    ]


def test_distributed_evaluation_restores_model_training_mode(monkeypatch):
    class FakeAccelerator:
        num_processes = 1
        process_index = 0

        @staticmethod
        def unwrap_model(model):
            return model

    config = OmegaConf.create(
        {
            "audio": {"sample_rate": 44_100, "chunk_size": 8},
            "training": {
                "target_instrument": "vocals",
                "instruments": ["vocals", "bass", "drums", "other"],
            },
        }
    )
    model = torch.nn.Linear(1, 1)
    model.train()
    monkeypatch.setattr(evaluate, "load_audio", lambda path, sample_rate: np.zeros((2, 8)))

    def fake_separate_overlap_add(model, *args, **kwargs):
        model.eval()
        return np.zeros((2, 8))

    monkeypatch.setattr(
        evaluate,
        "separate_overlap_add",
        fake_separate_overlap_add,
    )
    monkeypatch.setattr(evaluate, "museval_sdr", lambda *args, **kwargs: 1.0)

    evaluate_songs_distributed(
        config=config,
        model=model,
        songs=[Path("song_a")],
        hop_ratio=1,
        batch_size=1,
        accelerator=FakeAccelerator(),
    )

    assert model.training


def test_in_training_eval_metrics_include_each_stem_and_stem_average():
    train_eval = {
        "median_sdr": 2.5,
        "stem_median_sdr": {
            "vocals": 4.0,
            "bass": 1.0,
            "drums": 3.0,
            "other": 2.0,
        },
    }
    test_eval = {
        "median_sdr": 3.5,
        "stem_median_sdr": {
            "vocals": 5.0,
            "bass": 2.0,
            "drums": 4.0,
            "other": 3.0,
        },
    }

    metrics = build_eval_metrics(train_eval, test_eval)

    assert metrics == {
        "eval/train_sdr": 2.5,
        "eval/test_sdr": 3.5,
        "eval/train/stem_average_sdr": 2.5,
        "eval/train/vocals_sdr": 4.0,
        "eval/train/bass_sdr": 1.0,
        "eval/train/drums_sdr": 3.0,
        "eval/train/other_sdr": 2.0,
        "eval/test/stem_average_sdr": 3.5,
        "eval/test/vocals_sdr": 5.0,
        "eval/test/bass_sdr": 2.0,
        "eval/test/drums_sdr": 4.0,
        "eval/test/other_sdr": 3.0,
    }
