from pathlib import Path

import pytest

from noisegap.experiments.matrix import (
    Domain,
    RunPhase,
    SweepSpec,
    build_matrix,
    validate_recorded_manifest_split,
)
from noisegap.experiments.render import render_config
from noisegap.seeding import training_augmentation_seed


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
                Path("noise/dev.csv"),
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
    assert protocol["runtime"] == {
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "loader_workers": 0,
    }
    assert config["torch_num_threads"] == 1
    assert config["torch_num_interop_threads"] == 1
    assert config["augmentation"]["train"] is None
    assert config["augmentation"]["dev"] is None
    assert config["model"]["model_checkpoint"] == (
        "${results_dir}/SS_train-5_test-5/_best/model.pt"
    )


def test_rendered_config_can_select_article_dataset_base() -> None:
    run = next(run for run in build_matrix(_spec()) if run.phase is RunPhase.TRAIN)
    config = render_config(
        run,
        results_dir=Path("results"),
        base_config="noisegap_article_speechcommands",
    )

    assert config["defaults"] == ["noisegap_article_speechcommands", "_self_"]


def test_rendered_config_rejects_a_path_as_base_name() -> None:
    run = next(run for run in build_matrix(_spec()) if run.phase is RunPhase.TRAIN)

    with pytest.raises(ValueError, match="Hydra config name"):
        render_config(
            run,
            results_dir=Path("results"),
            base_config="../outside",
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


def test_training_seed_changes_train_but_not_evaluation_corruption() -> None:
    run = next(run for run in build_matrix(_spec()) if run.phase is RunPhase.TRAIN)
    config = render_config(run, results_dir=Path("results"), seed=2)
    train_parameters = config["augmentation"]["train"]["pipeline"][0][
        "noisegap.augmentations.SyntheticLogMelNoise"
    ]
    dev_parameters = config["augmentation"]["dev"]["pipeline"][0][
        "noisegap.augmentations.SyntheticLogMelNoise"
    ]
    test_parameters = config["augmentation"]["test"]["pipeline"][0][
        "noisegap.augmentations.SyntheticLogMelNoise"
    ]

    assert config["seed"] == 2
    assert config["noisegap_protocol"]["seed"] == 2
    assert train_parameters["generator_seed"] == training_augmentation_seed(2)
    assert dev_parameters["generator_seed"] == 1_000_000
    assert test_parameters["generator_seed"] == 2_000_000


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


def test_matched_feature_config_uses_waveform_frontend_and_padding() -> None:
    run = next(
        run
        for run in build_matrix(_spec())
        if run.phase is RunPhase.TRAIN and run.train_domain.kind == "recorded"
    )
    config = render_config(
        run,
        results_dir=Path("results"),
        base_config="noisegap_article_timit_feature_matched",
        seed=2,
        feature_implementation="matched_32k",
        loader_workers=4,
        feature_manifest=Path("features.json"),
    )
    parameters = config["augmentation"]["train"]["pipeline"][0][
        "noisegap.augmentations.RecordedLogMelNoise"
    ]

    assert parameters["sample_rate"] == 32000
    assert parameters["window_size"] == 1024
    assert parameters["hop_size"] == 320
    assert parameters["fmax"] == 14000
    assert parameters["padding_value_db"] == -100.0
    assert parameters["padding_tolerance_db"] == 0.01
    assert parameters["generator_seed"] == training_augmentation_seed(2)
    assert config["noisegap_protocol"]["frontend_cache"] == (
        "precomputed_clean_features"
    )
    assert config["noisegap_protocol"]["runtime"]["loader_workers"] == 4
    assert config["dataset"]["train_loader_kwargs"]["num_workers"] == 4
    assert config["noisegap_feature_manifest"].endswith("features.json")


def test_article_legacy_config_records_exact_historical_behavior() -> None:
    run = next(
        run
        for run in build_matrix(_spec())
        if run.phase is RunPhase.TRAIN and run.train_domain.kind == "recorded"
    )
    config = render_config(
        run,
        results_dir=Path("results"),
        feature_implementation="article_legacy",
    )

    train_pipeline = config["augmentation"]["train"]["pipeline"][0]
    dev_pipeline = config["augmentation"]["dev"]["pipeline"][0]
    assert "noisegap.augmentations.LegacyArticleRecordedLogMelNoise" in train_pipeline
    assert dev_pipeline[
        "noisegap.augmentations.LegacyArticleRecordedLogMelNoise"
    ]["noise_csv"] == "noise/train.csv"
    assert config["noisegap_protocol"]["recorded_noise_axis_behavior"] == (
        "legacy_64_frame_resize"
    )


def test_article_legacy_gaussian_test_is_static_per_item() -> None:
    run = next(run for run in build_matrix(_spec()) if run.phase is RunPhase.EVALUATE)
    config = render_config(
        run,
        results_dir=Path("results"),
        feature_implementation="article_legacy",
    )
    parameters = config["augmentation"]["test"]["pipeline"][0][
        "noisegap.augmentations.LegacyArticleSyntheticLogMelNoise"
    ]
    assert parameters["noise_type"] == "StaticGaussian"
    assert parameters["generator_seed"] == 0


def test_recorded_manifest_split_rejects_dev_test_overlap(tmp_path: Path) -> None:
    (tmp_path / "a.wav").touch()
    (tmp_path / "b.wav").touch()
    (tmp_path / "c.wav").touch()
    train = tmp_path / "train.csv"
    dev = tmp_path / "dev.csv"
    test = tmp_path / "test.csv"
    train.write_text("path\na.wav\n", encoding="utf-8")
    dev.write_text("path\nb.wav\n", encoding="utf-8")
    test.write_text("path\nb.wav\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Noise leakage: dev/test"):
        validate_recorded_manifest_split(tmp_path, train, dev, test)
