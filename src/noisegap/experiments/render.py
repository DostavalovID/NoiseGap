"""Render auditable Hydra configs from matrix entries."""

from pathlib import Path
from typing import Any

import yaml

from .matrix import Domain, RunPhase, RunSpec


def _augmentation(
    domain: Domain,
    snr_db: int,
    *,
    phase: str,
    training_seed: int,
    feature_implementation: str,
) -> dict:
    if phase not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown augmentation phase: {phase}")
    if feature_implementation not in {"corrected", "article_legacy"}:
        raise ValueError(
            f"Unknown feature implementation: {feature_implementation}"
        )
    evaluation = phase != "train"
    evaluation_seed = {"dev": 1_000_000, "test": 2_000_000}
    if feature_implementation == "article_legacy" and domain.kind == "synthetic":
        target = "noisegap.augmentations.LegacyArticleSyntheticLogMelNoise"
        parameters = {
            "snr": float(snr_db),
            "noise_type": "StaticGaussian" if phase == "test" else "Gaussian",
            "generator_seed": training_seed,
            "p": 1.0,
        }
    elif feature_implementation == "article_legacy":
        target = "noisegap.augmentations.LegacyArticleRecordedLogMelNoise"
        parameters = {
            "noise_dir": str(domain.root),
            "noise_csv": str(
                domain.test_manifest if phase == "test" else domain.train_manifest
            ),
            "snr_db": float(snr_db),
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 160,
            "n_mels": 64,
            "noise_type": None,
            "generator_seed": training_seed,
            "p": 1.0,
        }
    elif domain.kind == "synthetic":
        target = "noisegap.augmentations.SyntheticLogMelNoise"
        parameters: dict[str, Any] = {
            "snr_db": float(snr_db),
            "deterministic_per_item": evaluation,
            "generator_seed": (evaluation_seed[phase] if evaluation else training_seed),
            "p": 1.0,
        }
    else:
        manifest = {
            "train": domain.train_manifest,
            "dev": domain.dev_manifest,
            "test": domain.test_manifest,
        }[phase]
        target = "noisegap.augmentations.RecordedLogMelNoise"
        parameters = {
            "noise_root": str(domain.root),
            "manifest_csv": str(manifest),
            "snr_db": float(snr_db),
            "sample_rate": 16000,
            "window_size": 512,
            "hop_size": 160,
            "mel_bins": 64,
            "fmin": 50,
            "fmax": 8000,
            "ref": 1.0,
            "amin": 1e-10,
            "top_db": None,
            "deterministic_per_item": evaluation,
            "generator_seed": (evaluation_seed[phase] if evaluation else training_seed),
            "p": 1.0,
        }
    return {
        "_target_": "autrainer.augmentations.AugmentationPipeline",
        "id": f"{domain.label}({snr_db}dB)",
        "pipeline": [{target: parameters}],
    }


def render_config(
    run: RunSpec,
    *,
    results_dir: Path,
    base_config: str = "noisegap_base",
    seed: int = 0,
    feature_implementation: str = "corrected",
) -> dict:
    """Render one config with explicit train/dev/test semantics."""
    if not base_config or "/" in base_config or "\\" in base_config:
        raise ValueError("base_config must be a non-empty Hydra config name.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    train = _augmentation(
        run.train_domain,
        run.train_snr_db,
        phase="train",
        training_seed=seed,
        feature_implementation=feature_implementation,
    )
    test = _augmentation(
        run.test_domain,
        run.test_snr_db,
        phase="test",
        training_seed=seed,
        feature_implementation=feature_implementation,
    )
    dev = _augmentation(
        run.train_domain,
        run.train_snr_db,
        phase="dev",
        training_seed=seed,
        feature_implementation=feature_implementation,
    )
    config: dict[str, Any] = {
        "defaults": [base_config, "_self_"],
        "hydra": {
            "searchpath": [
                "file://${oc.env:NOISEGAP_CONFIG_DIR,conf}",
            ]
        },
        "results_dir": str(results_dir),
        "experiment_id": run.experiment_id,
        "iterations": run.iterations,
        "seed": seed,
        "batch_size": 64,
        "learning_rate": 0.001,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "noisegap_protocol": {
            "phase": run.phase.value,
            "seed": seed,
            "feature_layout": "channel,time,mel",
            "corruption_space": "log_mel_power",
            "feature_implementation": feature_implementation,
            "snr_definition": (
                "mean_signal_power_over_mean_noise_power_on_nonzero_frames"
            ),
            "padding_policy": "preserve_zero_frames",
            "speech_feature_extractor": "autrainer:PannMel(log_mel_16k)",
            "train_domain": run.train_domain.label,
            "test_domain": run.test_domain.label,
            "train_snr_db": run.train_snr_db,
            "test_snr_db": run.test_snr_db,
            "checkpoint_selection_domain": run.train_domain.label,
            "checkpoint_selection_snr_db": run.train_snr_db,
            "dev_noise_seed": 1_000_000,
            "test_noise_seed": 2_000_000,
            "runtime": {
                "torch_num_threads": 1,
                "torch_num_interop_threads": 1,
                "loader_workers": 0,
            },
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
    if feature_implementation == "article_legacy":
        config["noisegap_protocol"].update(
            {
                "historical_source_revision": (
                    "f46af323af907c9e7425364974a7975474d17e2f"
                ),
                "gaussian_power_field": "abs_randn_scaled_by_mean",
                "recorded_noise_frontend": "torchaudio_MelSpectrogram_defaults",
                "recorded_noise_decoder": "audiofile_pcm_compatibility_adapter",
                "recorded_noise_axis_behavior": "legacy_64_frame_resize",
                "dev_recorded_noise_manifest": "train_manifest",
                "dev_noise_seed": seed,
                "test_noise_seed": seed,
            }
        )
    if run.phase is RunPhase.EVALUATE:
        config["model"] = {
            "model_checkpoint": (
                f"${{results_dir}}/{run.training_experiment_id}/_best/model.pt"
            ),
            "skip_last_layer": False,
        }
    return config


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
