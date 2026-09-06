from pathlib import Path

import pytest

from training_utils import load_config


def test_release_contains_only_base_configs():
    root = Path(__file__).resolve().parents[1]
    expected = {f"sbroformer_base_{stem}.yaml" for stem in ["4stem", "vocals", "bass", "drums", "other"]}
    assert {path.name for path in (root / "configs").glob("*.yaml")} == expected


def test_release_four_stem_config_contract():
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "configs" / "sbroformer_base_4stem.yaml"))
    assert "extends" not in config
    assert config.model.num_stems == 4
    assert config.training.target_instrument is None
    assert config.training.eval_every_updates == 0
    assert list(config.training.instruments) == ["vocals", "bass", "drums", "other"]


@pytest.mark.parametrize("stem", ["vocals", "bass", "drums", "other"])
def test_release_single_stem_config_contract(stem):
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "configs" / f"sbroformer_base_{stem}.yaml"))
    assert "extends" not in config
    assert config.model.num_stems == 1
    assert config.model.dim == 576
    assert config.model.n_layers == 12
    assert config.training.target_instrument == stem
    assert config.training.output_dir == f"checkpoints/sbroformer_base_{stem}"
    assert config.training.batch_size == 4
    assert config.training.eval_every_updates == 0
    assert config.training.multiprocessing_context == "spawn"
    assert config.validation.root == "data/musdb18hq"
    assert not config.augmentations.random_mixing.enable
    assert list(config.training.instruments) == ["vocals", "bass", "drums", "other"]
