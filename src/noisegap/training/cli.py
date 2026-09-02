"""Hydra entrypoint using the NoiseGap trainer adapter."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import autrainer
import hydra
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf

from noisegap import __version__

from .trainer import NoiseGapTrainer


def configure_torch_runtime(cfg: DictConfig) -> dict[str, int]:
    """Apply an explicit CPU thread budget before dataset construction."""
    num_threads = int(cfg.get("torch_num_threads", torch.get_num_threads()))
    num_interop_threads = int(
        cfg.get("torch_num_interop_threads", torch.get_num_interop_threads())
    )
    if num_threads <= 0 or num_interop_threads <= 0:
        raise ValueError("Torch thread counts must be positive.")
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_interop_threads)
    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def sha256_file(path: Path) -> str:
    """Hash an artifact without loading the full file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return None
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _input_metadata(cfg: DictConfig) -> dict:
    """Bind split metadata and phase-specific noise manifests to a run."""
    artifacts: dict[str, object] = {"dataset_splits": {}, "noise_manifests": {}}
    dataset = cfg.get("dataset", {})
    dataset_path = dataset.get("path")
    if dataset_path:
        for split in ("train", "dev", "test"):
            path = Path(str(dataset_path), f"{split}.csv").resolve()
            if path.is_file():
                artifacts["dataset_splits"][split] = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
        split_manifest = Path(str(dataset_path), "split_manifest.json").resolve()
        if split_manifest.is_file():
            artifacts["dataset_split_manifest"] = {
                "path": str(split_manifest),
                "sha256": sha256_file(split_manifest),
            }

    feature_manifest = cfg.get("noisegap_feature_manifest")
    if feature_manifest:
        path = Path(str(feature_manifest)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured feature manifest is missing: {path}")
        artifacts["feature_manifest"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    augmentation_config = cfg.get("augmentation", {})
    augmentation = (
        OmegaConf.to_container(augmentation_config, resolve=True)
        if OmegaConf.is_config(augmentation_config)
        else augmentation_config
    )
    if isinstance(augmentation, dict):

        def collect_manifests(value: object, output: list[str]) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    if key in {"manifest_csv", "noise_csv"} and nested is not None:
                        output.append(str(nested))
                    else:
                        collect_manifests(nested, output)
            elif isinstance(value, list):
                for nested in value:
                    collect_manifests(nested, output)

        for phase in ("train", "dev", "test"):
            phase_config = augmentation.get(phase)
            if not isinstance(phase_config, dict):
                continue
            manifests: list[str] = []
            collect_manifests(phase_config, manifests)
            if manifests:
                phase_artifacts = []
                for manifest in manifests:
                    path = Path(manifest).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"Configured {phase} noise manifest is missing: {path}"
                        )
                    phase_artifacts.append(
                        {
                            "role": "noise_paths",
                            "path": str(path),
                            "sha256": sha256_file(path),
                        }
                    )
                    split_manifest = path.parent / "split_manifest.json"
                    if split_manifest.is_file():
                        phase_artifacts.append(
                            {
                                "role": "noise_split_policy",
                                "path": str(split_manifest),
                                "sha256": sha256_file(split_manifest),
                            }
                        )
                artifacts["noise_manifests"][phase] = phase_artifacts
    return artifacts


def build_provenance(cfg: DictConfig) -> dict:
    """Build non-secret run provenance from the resolved experiment state."""
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    revision = _git_value("rev-parse", "HEAD")
    status = _git_value("status", "--porcelain")
    provenance = {
        "noisegap_version": __version__,
        "autrainer_version": autrainer.__version__,
        "torch_version": torch.__version__,
        "torchaudio_version": torchaudio.__version__,
        "python_version": sys.version.split()[0],
        "torch_runtime": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "git_revision": revision,
        "git_dirty": None if revision is None or status is None else bool(status),
        "resolved_config_sha256": hashlib.sha256(
            resolved_yaml.encode("utf-8")
        ).hexdigest(),
        "protocol": OmegaConf.to_container(
            cfg.noisegap_protocol,
            resolve=True,
        ),
        "input_metadata": _input_metadata(cfg),
    }
    checkpoint = cfg.get("model", {}).get("model_checkpoint")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        provenance["input_checkpoint"] = {
            "path": str(checkpoint_path),
            "sha256": (
                sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
            ),
        }
    return provenance


def finalize_provenance(
    provenance_path: Path,
    output_dir: Path,
    cfg: DictConfig,
) -> None:
    """Bind successful run artifacts to the pre-run provenance record."""
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    artifacts = {}
    test_artifacts = {
        "test_holistic": "test_holistic.yaml",
        "test_results": "test_results.csv",
        "test_indices": "test_indices.npy",
        "test_targets": "test_targets.npy",
        "test_outputs": "test_outputs.npy",
        "test_losses": "test_losses.npy",
    }
    for artifact_name, filename in test_artifacts.items():
        artifact_path = output_dir / "_test" / filename
        if artifact_path.is_file():
            artifacts[artifact_name] = {
                "path": str(artifact_path),
                "sha256": sha256_file(artifact_path),
            }
    if cfg.iterations > 0:
        checkpoint = output_dir / "_best" / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Successful training did not produce checkpoint: {checkpoint}"
            )
        artifacts["output_checkpoint"] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        }
    provenance["artifacts"] = artifacts
    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    @autrainer.main("config")
    def run(cfg: DictConfig) -> float:
        OmegaConf.set_struct(cfg, False)
        OmegaConf.resolve(cfg)
        configure_torch_runtime(cfg)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        completed = os.path.join(output_dir, "_test", "test_holistic.yaml")
        if os.path.isfile(completed):
            raise RuntimeError(
                f"Run already has a completed test artifact: {completed}"
            )
        cfg_path = os.path.join(output_dir, ".hydra", "config.yaml")
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        OmegaConf.save(cfg, cfg_path)
        provenance_path = Path(output_dir, "noisegap_provenance.json")
        provenance_path.write_text(
            json.dumps(build_provenance(cfg), indent=2),
            encoding="utf-8",
        )
        trainer = NoiseGapTrainer(cfg=cfg, output_directory=output_dir)
        result = trainer.train()
        finalize_provenance(
            provenance_path,
            Path(output_dir),
            cfg,
        )
        return result

    run()


if __name__ == "__main__":
    main()
