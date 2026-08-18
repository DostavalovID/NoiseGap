from pathlib import Path

import yaml
from autrainer.augmentations import AugmentationManager
from autrainer.transforms import TransformManager

from noisegap.experiments.matrix import Domain, RunPhase
from noisegap.experiments.speechcommands_cli import MODELS
from noisegap.experiments.waveform_matrix import (
    WaveformSweepSpec,
    build_waveform_matrix,
)
from noisegap.experiments.waveform_render import render_waveform_config


def _spec() -> WaveformSweepSpec:
    return WaveformSweepSpec(
        models=tuple(MODELS.values()),
        domains=(
            Domain("G", "GaussianWaveform", "synthetic"),
            Domain(
                "B",
                "BackgroundWaveform",
                "recorded",
                root=Path("noise"),
                train_manifest=Path("noise/train.csv"),
                test_manifest=Path("noise/test.csv"),
            ),
        ),
    )


def test_headline_matrix_has_12_train_and_132_evaluate_runs() -> None:
    runs = build_waveform_matrix(_spec())
    train = [run for run in runs if run.phase is RunPhase.TRAIN]
    evaluate = [run for run in runs if run.phase is RunPhase.EVALUATE]

    assert len(runs) == 144
    assert len(train) == 12
    assert len(evaluate) == 132
    assert len({run.experiment_id for run in runs}) == 144


def test_both_models_receive_the_same_deterministic_test_corruption() -> None:
    runs = build_waveform_matrix(_spec())
    selected = [
        run
        for run in runs
        if run.phase is RunPhase.EVALUATE
        and run.seed == 0
        and run.train_domain.code == "G"
        and run.test_domain.code == "B"
        and run.test_snr_db == 10
    ]
    assert len(selected) == 2

    configs = [
        render_waveform_config(run, results_dir=Path("results"))
        for run in selected
    ]
    first_test = configs[0]["augmentation"]["test"]
    second_test = configs[1]["augmentation"]["test"]

    assert first_test == second_test
    parameters = first_test["pipeline"][0][
        "noisegap.augmentations.RecordedWaveformNoise"
    ]
    assert parameters["deterministic_per_item"] is True
    assert parameters["generator_seed"] == 0
    assert parameters["order"] == -97


def test_model_seed_and_waveform_contract_are_recorded() -> None:
    run = next(
        run
        for run in build_waveform_matrix(_spec())
        if run.phase is RunPhase.TRAIN and run.model.code == "ast" and run.seed == 2
    )

    config = render_waveform_config(run, results_dir=Path("results"))
    protocol = config["noisegap_protocol"]

    assert {"override /model": "ASTModel-T-waveform"} in config["defaults"]
    assert {"override /optimizer": "AdamW"} in config["defaults"]
    assert protocol["model"] == "AST-online-HF"
    assert protocol["seed"] == 2
    assert protocol["corruption_space"] == "waveform_amplitude"
    assert protocol["frontend_position"] == "after_corruption"
    assert protocol["clipping_policy"] == "no_clipping"


def test_cnn10_composed_transform_order_keeps_noise_before_frontend() -> None:
    repository = Path(__file__).resolve().parents[1]
    dataset = yaml.safe_load(
        (repository / "conf/dataset/SpeechCommands-waveform-16k.yaml").read_text()
    )["transform"]
    model = yaml.safe_load(
        (repository / "conf/model/Cnn10-32k-T-waveform.yaml").read_text()
    )["transform"]
    run = next(
        run
        for run in build_waveform_matrix(_spec())
        if run.phase is RunPhase.TRAIN and run.model.code == "cnn10"
    )
    config = render_waveform_config(run, results_dir=Path("results"))
    augmentations = AugmentationManager(
        config["augmentation"]["train"],
        config["augmentation"]["dev"],
        config["augmentation"]["test"],
    ).get_augmentations()

    train, _, _ = TransformManager(model, dataset, *augmentations).get_transforms()

    actual = [
        (type(transform).__name__, transform.order)
        for transform in train.transforms
    ]
    assert actual == [
        ("Expand", -100),
        ("WaveformGaussianNoise", -97),
        ("Resample", -95),
        ("PannMel", -90),
    ]
