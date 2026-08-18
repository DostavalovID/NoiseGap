"""Render common-waveform CNN10/AST SpeechCommands configs."""

from pathlib import Path
from typing import Any

from .matrix import Domain, RunPhase
from .waveform_matrix import WaveformRunSpec


def _augmentation(
    domain: Domain,
    snr_db: int,
    *,
    evaluation: bool,
    training_seed: int,
) -> dict:
    parameters: dict[str, Any] = {
        "snr_db": float(snr_db),
        "deterministic_per_item": evaluation,
        "generator_seed": 0 if evaluation else training_seed,
        "order": -97,
        "p": 1.0,
    }
    if domain.kind == "synthetic":
        target = "noisegap.augmentations.WaveformGaussianNoise"
    else:
        target = "noisegap.augmentations.RecordedWaveformNoise"
        manifest = domain.test_manifest if evaluation else domain.train_manifest
        parameters.update(
            {
                "noise_root": str(domain.root),
                "manifest_csv": str(manifest),
                "sample_rate": 16000,
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
) -> dict[str, Any]:
    """Render one run with corruption before either model frontend."""
    train = _augmentation(
        run.train_domain,
        run.train_snr_db,
        evaluation=False,
        training_seed=run.seed,
    )
    dev = _augmentation(
        run.train_domain,
        run.train_snr_db,
        evaluation=True,
        training_seed=run.seed,
    )
    test = _augmentation(
        run.test_domain,
        run.test_snr_db,
        evaluation=True,
        training_seed=run.seed,
    )
    config: dict[str, Any] = {
        "defaults": [
            "noisegap_speechcommands",
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
            "dataset": "SpeechCommands-v0.02",
            "model": run.model.label,
            "model_config": run.model.config,
            "seed": run.seed,
            "waveform_layout": "channel,samples",
            "input_sample_rate": 16000,
            "corruption_space": "waveform_amplitude",
            "snr_definition": "mean_signal_square_over_mean_added_noise_square",
            "clipping_policy": "no_clipping",
            "frontend_position": "after_corruption",
            "train_domain": run.train_domain.label,
            "test_domain": run.test_domain.label,
            "train_snr_db": run.train_snr_db,
            "test_snr_db": run.test_snr_db,
            "checkpoint_selection_domain": run.train_domain.label,
            "checkpoint_selection_snr_db": run.train_snr_db,
            "evaluation_noise_seed": 0,
        },
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
