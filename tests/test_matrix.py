from pathlib import Path

from noisegap.experiments.matrix import (
    Domain,
    RunPhase,
    SweepSpec,
    build_matrix,
)
from noisegap.experiments.render import render_config


def _spec() -> SweepSpec:
    return SweepSpec(
        domains=(
            Domain("S", "Synthetic", "synthetic"),
            Domain(
                "R",
                "Recorded",
                "recorded",
                Path("noise"),
                Path("noise/train.csv"),
                Path("noise/test.csv"),
            ),
        )
    )


def test_matrix_has_12_train_and_132_evaluate_runs() -> None:
    runs = build_matrix(_spec())
    train = [run for run in runs if run.phase is RunPhase.TRAIN]
    evaluate = [run for run in runs if run.phase is RunPhase.EVALUATE]
    assert len(runs) == 144
    assert len(train) == 12
    assert len(evaluate) == 132
    assert len({run.experiment_id for run in runs}) == 144
    assert all(run.phase is RunPhase.TRAIN for run in runs[:12])
    assert all(run.phase is RunPhase.EVALUATE for run in runs[12:])


def test_rendered_protocol_is_explicit() -> None:
    run = next(run for run in build_matrix(_spec()) if run.phase is RunPhase.EVALUATE)
    config = render_config(
        run,
        results_dir=Path("results"),
    )
    protocol = config["noisegap_protocol"]
    assert config["defaults"][0] == "noisegap_base"
    assert config["hydra"]["searchpath"] == [
        "file://${oc.env:NOISEGAP_CONFIG_DIR,conf}"
    ]
    assert protocol["feature_layout"] == "channel,time,mel"
    assert protocol["corruption_space"] == "log_mel_power"
    assert protocol["snr_definition"] == (
        "mean_signal_power_over_mean_noise_power_on_nonzero_frames"
    )
    assert protocol["padding_policy"] == "preserve_zero_frames"
    assert config["augmentation"]["train"] is None
    assert config["augmentation"]["dev"] is None
    assert config["model"]["model_checkpoint"] == (
        "${results_dir}/SS_train-5_test-5/_best/model.pt"
    )


def test_training_dev_noise_is_stable_but_train_noise_is_not() -> None:
    run = next(run for run in build_matrix(_spec()) if run.phase is RunPhase.TRAIN)
    config = render_config(run, results_dir=Path("results"))

    assert (
        config["augmentation"]["train"]["pipeline"][0][
            "noisegap.augmentations.SyntheticLogMelNoise"
        ]["deterministic_per_item"]
        is False
    )
    assert (
        config["augmentation"]["dev"]["pipeline"][0][
            "noisegap.augmentations.SyntheticLogMelNoise"
        ]["deterministic_per_item"]
        is True
    )


def test_recorded_noise_uses_speech_pann_parameters() -> None:
    run = next(
        run
        for run in build_matrix(_spec())
        if run.phase is RunPhase.TRAIN and run.train_domain.kind == "recorded"
    )
    config = render_config(run, results_dir=Path("results"))
    parameters = config["augmentation"]["train"]["pipeline"][0][
        "noisegap.augmentations.RecordedLogMelNoise"
    ]

    assert parameters["sample_rate"] == 16000
    assert parameters["window_size"] == 512
    assert parameters["hop_size"] == 160
    assert parameters["mel_bins"] == 64
    assert parameters["fmin"] == 50
    assert parameters["fmax"] == 8000
    assert parameters["ref"] == 1.0
    assert parameters["amin"] == 1e-10
    assert parameters["top_db"] is None
