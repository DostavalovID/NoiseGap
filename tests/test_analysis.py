import csv
import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from noisegap.analysis import summarize_manifest
from noisegap.training.cli import sha256_file


def _write_completed_run(tmp_path: Path) -> Path:
    result_dir = tmp_path / "results" / "SS_train-5_test-5"
    test_artifact = result_dir / "_test" / "test_holistic.yaml"
    test_artifact.parent.mkdir(parents=True)
    test_artifact.write_text(
        "accuracy:\n  all: 0.75\nloss:\n  all: 1.25\n",
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "iterations": 1,
            "noisegap_protocol": {
                "phase": "train",
                "train_domain": "Synthetic",
                "test_domain": "Synthetic",
                "train_snr_db": -5,
                "test_snr_db": -5,
            },
        }
    )
    resolved_config = result_dir / ".hydra" / "config.yaml"
    resolved_config.parent.mkdir()
    OmegaConf.save(cfg, resolved_config)
    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    config_hash = hashlib.sha256(resolved_yaml.encode()).hexdigest()
    provenance = {
        "git_revision": "abc123",
        "git_dirty": False,
        "resolved_config_sha256": config_hash,
        "protocol": OmegaConf.to_container(cfg.noisegap_protocol),
        "artifacts": {
            "test_holistic": {
                "path": str(test_artifact),
                "sha256": sha256_file(test_artifact),
            }
        },
    }
    (result_dir / "noisegap_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    manifest = [
        {
            "phase": "train",
            "experiment_id": "SS_train-5_test-5",
            "train_domain": "Synthetic",
            "test_domain": "Synthetic",
            "train_snr_db": -5,
            "test_snr_db": -5,
            "config": "configs/train.yaml",
            "result_dir": str(result_dir),
            "checkpoint": str(result_dir / "_best/model.pt"),
            "depends_on": None,
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_summarizer_writes_only_provenance_checked_metrics(
    tmp_path: Path,
) -> None:
    manifest = _write_completed_run(tmp_path)
    output = tmp_path / "summary.csv"

    completed, missing = summarize_manifest(manifest, output)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert (completed, missing) == (1, 0)
    assert rows[0]["accuracy"] == "0.75"
    assert rows[0]["git_revision"] == "abc123"


def test_summarizer_rejects_tampered_metric_artifact(tmp_path: Path) -> None:
    manifest = _write_completed_run(tmp_path)
    test_artifact = (
        tmp_path / "results" / "SS_train-5_test-5" / "_test" / "test_holistic.yaml"
    )
    test_artifact.write_text("accuracy:\n  all: 1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash"):
        summarize_manifest(manifest, tmp_path / "summary.csv")


def test_summarizer_rejects_uncommitted_run_by_default(tmp_path: Path) -> None:
    manifest = _write_completed_run(tmp_path)
    provenance_path = (
        tmp_path / "results" / "SS_train-5_test-5" / "noisegap_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["git_revision"] = None
    provenance["git_dirty"] = None
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="Committed clean Git provenance"):
        summarize_manifest(manifest, tmp_path / "summary.csv")

    completed, missing = summarize_manifest(
        manifest,
        tmp_path / "diagnostic.csv",
        allow_uncommitted=True,
    )
    assert (completed, missing) == (1, 0)
