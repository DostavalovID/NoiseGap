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
        "git_revision": revision,
        "git_dirty": None if revision is None or status is None else bool(status),
        "resolved_config_sha256": hashlib.sha256(
            resolved_yaml.encode("utf-8")
        ).hexdigest(),
        "protocol": OmegaConf.to_container(
            cfg.noisegap_protocol,
            resolve=True,
        ),
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
    test_artifact = output_dir / "_test" / "test_holistic.yaml"
    if test_artifact.is_file():
        artifacts["test_holistic"] = {
            "path": str(test_artifact),
            "sha256": sha256_file(test_artifact),
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
