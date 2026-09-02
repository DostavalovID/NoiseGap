"""Render common-waveform CNN10/AST SpeechCommands configs."""

from pathlib import Path
from typing import Any

from ..seeding import (
    TRAIN_AUGMENTATION_SEED_STRIDE,
    training_augmentation_seed,
)
from .matrix import Domain, RunPhase
from .waveform_matrix import WaveformRunSpec


def _augmentation(
    domain: Domain,
    snr_db: int,
    *,
    phase: str,
    training_seed: int,
    order: int,
    min_recorded_crop_rms_ratio: float,
) -> dict:
    if phase not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown augmentation phase: {phase}")
    evaluation = phase != "train"
    evaluation_seed = {"dev": 1_000_000, "test": 2_000_000}
    parameters: dict[str, Any] = {
        "snr_db": float(snr_db),
        "deterministic_per_item": evaluation,
        "generator_seed": (
            evaluation_seed[phase]
            if evaluation
            else training_augmentation_seed(training_seed)
        ),
        "order": order,
        "p": 1.0,
    }
    if domain.kind == "synthetic":
        target = "noisegap.augmentations.WaveformGaussianNoise"
    else:
        target = "noisegap.augmentations.RecordedWaveformNoise"
        manifest = {
            "train": domain.train_manifest,
            "dev": domain.dev_manifest,
            "test": domain.test_manifest,
        }[phase]
        parameters.update(
            {
                "noise_root": str(domain.root),
                "manifest_csv": str(manifest),
                "sample_rate": 16000,
                "max_crop_attempts": 32,
                "min_crop_rms_ratio": min_recorded_crop_rms_ratio,
            }
        )
    return {
        "_target_": "autrainer.augmentations.AugmentationPipeline",
        "id": f"{domain.label}({snr_db}dB)",
        "pipeline": [{target: parameters}],
    }


def render_waveform_config(
    run: WaveformRunSpec,
    *,
    results_dir: Path,
    base_config: str = "noisegap_speechcommands",
    dataset_label: str = "SpeechCommands-v0.02",
    noise_order: int = -97,
    min_recorded_crop_rms_ratio: float = 0.1,
    tracking_metric: str = "autrainer.metrics.UAR",
) -> dict[str, Any]:
    """Render one run with corruption before either model frontend."""
    if not base_config or "/" in base_config or "\\" in base_config:
        raise ValueError("base_config must be a non-empty Hydra config name.")
    if not dataset_label:
        raise ValueError("dataset_label must be non-empty.")
    if not 0 <= min_recorded_crop_rms_ratio <= 1:
        raise ValueError("min_recorded_crop_rms_ratio must be in [0, 1].")
    if tracking_metric not in {
        "autrainer.metrics.Accuracy",
        "autrainer.metrics.UAR",
        "autrainer.metrics.F1",
    }:
        raise ValueError(f"Unsupported tracking_metric: {tracking_metric}")
    train = _augmentation(
        run.train_domain,
        run.train_snr_db,
        phase="train",
        training_seed=run.seed,
        order=noise_order,
        min_recorded_crop_rms_ratio=min_recorded_crop_rms_ratio,
    )
    dev = _augmentation(
        run.train_domain,
        run.train_snr_db,
        phase="dev",
        training_seed=run.seed,
        order=noise_order,
        min_recorded_crop_rms_ratio=min_recorded_crop_rms_ratio,
    )
    test = _augmentation(
        run.test_domain,
        run.test_snr_db,
        phase="test",
        training_seed=run.seed,
        order=noise_order,
        min_recorded_crop_rms_ratio=min_recorded_crop_rms_ratio,
    )
    config: dict[str, Any] = {
        "defaults": [
            base_config,
            {"override /model": run.model.config},
            {"override /optimizer": run.model.optimizer},
            "_self_",
        ],
        "hydra": {
            "searchpath": ["file://${oc.env:NOISEGAP_CONFIG_DIR,conf}"],
        },
        "results_dir": str(results_dir),
        "experiment_id": run.experiment_id,
        "iterations": run.iterations,
        "seed": run.seed,
        "batch_size": run.model.batch_size,
        "learning_rate": run.model.learning_rate,
        "noisegap_protocol": {
            "phase": run.phase.value,
            "dataset": dataset_label,
            "model": run.model.label,
            "model_config": run.model.config,
            "seed": run.seed,
            "waveform_layout": "channel,samples",
            "input_sample_rate": 16000,
            "corruption_space": "waveform_amplitude",
            "snr_definition": "mean_signal_square_over_mean_added_noise_square",
            "clipping_policy": "no_clipping",
            "frontend_position": "after_corruption",
            "corruption_order": noise_order,
            "train_domain": run.train_domain.label,
            "test_domain": run.test_domain.label,
            "train_snr_db": run.train_snr_db,
            "test_snr_db": run.test_snr_db,
            "checkpoint_selection_domain": run.train_domain.label,
            "checkpoint_selection_snr_db": run.train_snr_db,
            "dev_noise_seed": 1_000_000,
            "test_noise_seed": 2_000_000,
            "train_augmentation_seed": training_augmentation_seed(run.seed),
            "train_augmentation_seed_stride_between_runs": (
                TRAIN_AUGMENTATION_SEED_STRIDE
            ),
            "train_worker_seed_offset_policy": "base_plus_worker_id",
            "checkpoint_selection_metric": tracking_metric,
            "recorded_noise_silence_policy": "source_relative_rms_threshold",
            "recorded_noise_max_crop_attempts": 32,
            "recorded_noise_min_crop_rms_ratio": min_recorded_crop_rms_ratio,
        },
        "dataset": {"tracking_metric": tracking_metric},
        "augmentation": {
            "id": (
                f"{run.train_domain.code}{run.train_snr_db}"
                f"-{run.test_domain.code}{run.test_snr_db}"
            ),
            "train": train if run.phase is RunPhase.TRAIN else None,
            "dev": dev if run.phase is RunPhase.TRAIN else None,
            "test": test,
        },
    }
    if run.phase is RunPhase.EVALUATE:
        config["model"] = {
            "model_checkpoint": (
                f"${{results_dir}}/{run.training_experiment_id}/_best/model.pt"
            ),
            "skip_last_layer": False,
        }
    return config
