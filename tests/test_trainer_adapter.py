from pathlib import Path

import pytest
from autrainer.transforms import SmartCompose
from omegaconf import OmegaConf

from noisegap.training import trainer as trainer_module
from noisegap.training.trainer import (
    NoiseGapTrainer,
    PhaseAwareAugmentationManager,
    extract_test_tracking_value,
)


def _pipeline(snr_db: float) -> dict:
    return {
        "_target_": "autrainer.augmentations.AugmentationPipeline",
        "id": f"synthetic-{snr_db}",
        "pipeline": [
            {
                "noisegap.augmentations.SyntheticLogMelNoise": {
                    "snr_db": snr_db,
                    "generator_seed": 0,
                }
            }
        ],
    }


def test_phase_bundle_maps_train_dev_test_independently() -> None:
    manager = PhaseAwareAugmentationManager(
        {
            "id": "phase-aware",
            "train": _pipeline(0),
            "dev": None,
            "test": _pipeline(20),
        }
    )

    train, dev, test = manager.get_augmentations()

    assert all(isinstance(item, SmartCompose) for item in (train, dev, test))
    assert len(train.transforms) == 1
    assert len(dev.transforms) == 0
    assert len(test.transforms) == 1
    assert train.transforms[0].snr_db == 0
    assert test.transforms[0].snr_db == 20


def test_legacy_single_pipeline_remains_train_only() -> None:
    train, dev, test = PhaseAwareAugmentationManager(_pipeline(10)).get_augmentations()

    assert len(train.transforms) == 1
    assert len(dev.transforms) == 0
    assert len(test.transforms) == 0


def test_trainer_installs_phase_manager_only_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_init(self: object, **kwargs: object) -> None:
        observed.append(trainer_module.training_module.AugmentationManager)

    original = trainer_module.training_module.AugmentationManager
    monkeypatch.setattr(trainer_module.ModularTaskTrainer, "__init__", fake_init)
    cfg = OmegaConf.create({"iterations": 1, "model": {}})

    NoiseGapTrainer(cfg=cfg, output_directory="unused")

    assert observed == [PhaseAwareAugmentationManager]
    assert trainer_module.training_module.AugmentationManager is original


def test_evaluation_refuses_missing_checkpoint(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "iterations": 0,
            "model": {"model_checkpoint": str(tmp_path / "missing.pt")},
        }
    )

    with pytest.raises(FileNotFoundError, match="missing.pt"):
        NoiseGapTrainer(cfg=cfg, output_directory="unused")


def test_tracking_metric_keeps_autrainer_test_prefix() -> None:
    results = {"test_loss": 1.0, "test_accuracy": 0.75}

    assert extract_test_tracking_value(results, "accuracy") == 0.75


def test_evaluation_cleanup_only_removes_local_redundant_states(
    tmp_path: Path,
) -> None:
    trainer = object.__new__(NoiseGapTrainer)
    trainer.output_directory = tmp_path
    for name in ("_initial", "_best", "_test"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "marker").touch()

    trainer._remove_redundant_evaluation_states()

    assert not (tmp_path / "_initial").exists()
    assert not (tmp_path / "_best").exists()
    assert (tmp_path / "_test" / "marker").is_file()
