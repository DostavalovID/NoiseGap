import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from noisegap.diagnostics import diagnose_manifests
from noisegap.training.cli import sha256_file


def _write_diagnostic_run(
    tmp_path: Path,
    *,
    hash_predictions: bool = True,
    hash_test_split: bool = True,
) -> Path:
    result_dir = tmp_path / "results" / "cell"
    test_dir = result_dir / "_test"
    test_dir.mkdir(parents=True)
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    test_split = dataset_dir / "test.csv"
    with test_split.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(
            [
                {"path": "0.wav", "label": "A"},
                {"path": "1.wav", "label": "B"},
                {"path": "2.wav", "label": "C"},
                {"path": "3.wav", "label": "C"},
            ]
        )
    for split in ("train", "dev"):
        (dataset_dir / f"{split}.csv").write_text(
            test_split.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    predictions = test_dir / "test_results.csv"
    with predictions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["index", "predictions", "A", "B", "C"],
        )
        writer.writeheader()
        for index in range(4):
            writer.writerow(
                {
                    "index": index,
                    "predictions": "C",
                    "A": 0.0,
                    "B": 0.0,
                    "C": 1.0,
                }
            )

    holistic = test_dir / "test_holistic.yaml"
    holistic.write_text(
        yaml.safe_dump(
            {
                "accuracy": {"all": 0.5},
                "uar": {"all": 1 / 3},
                "f1": {"all": 2 / 9},
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "dataset": {"target_column": "label"},
            "noisegap_protocol": {
                "phase": "evaluate",
                "train_domain": "Gaussian",
                "test_domain": "Recorded",
                "train_snr_db": -5,
                "test_snr_db": -5,
                "seed": 0,
            },
        }
    )
    config_path = result_dir / ".hydra" / "config.yaml"
    config_path.parent.mkdir()
    OmegaConf.save(cfg, config_path)
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    artifacts = {
        "test_holistic": {
            "path": str(holistic),
            "sha256": sha256_file(holistic),
        }
    }
    if hash_predictions:
        artifacts["test_results"] = {
            "path": str(predictions),
            "sha256": sha256_file(predictions),
        }
    provenance = {
        "git_revision": "abc123",
        "git_dirty": False,
        "resolved_config_sha256": hashlib.sha256(resolved.encode()).hexdigest(),
        "protocol": OmegaConf.to_container(cfg.noisegap_protocol),
        "input_metadata": {"dataset_splits": {}},
        "artifacts": artifacts,
    }
    if hash_test_split:
        provenance["input_metadata"]["dataset_splits"]["test"] = {
            "path": str(test_split),
            "sha256": sha256_file(test_split),
        }
    (result_dir / "noisegap_provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    manifest = [
        {
            "phase": "evaluate",
            "experiment_id": "cell",
            "seed": 0,
            "train_domain": "Gaussian",
            "test_domain": "Recorded",
            "train_snr_db": -5,
            "test_snr_db": -5,
            "config": "configs/cell.yaml",
            "result_dir": str(result_dir),
            "checkpoint": str(tmp_path / "model.pt"),
            "depends_on": "train.yaml",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_diagnostics_recompute_metrics_and_detect_collapse(tmp_path: Path) -> None:
    manifest = _write_diagnostic_run(tmp_path)
    output = tmp_path / "diagnostics.csv"

    count = diagnose_manifests(
        [manifest],
        output,
        require_hashed_predictions=True,
    )

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert count == 1
    assert float(row["accuracy"]) == pytest.approx(0.5)
    assert float(row["uar"]) == pytest.approx(1 / 3)
    assert float(row["f1"]) == pytest.approx(2 / 9)
    assert row["majority_prediction"] == "C"
    assert float(row["majority_prediction_share"]) == 1.0
    assert row["collapsed"] == "True"
    assert row["test_results_provenance_verified"] == "True"
    assert row["test_split_provenance_verified"] == "True"
    assert row["confusion_A_as_C"] == "1"
    assert row["confusion_C_as_C"] == "2"


def test_diagnostics_can_label_legacy_unhashed_predictions(tmp_path: Path) -> None:
    manifest = _write_diagnostic_run(tmp_path, hash_predictions=False)
    output = tmp_path / "diagnostics.csv"

    diagnose_manifests([manifest], output)

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["test_results_provenance_verified"] == "False"
    with pytest.raises(ValueError, match="Missing test_results provenance"):
        diagnose_manifests(
            [manifest],
            tmp_path / "strict.csv",
            require_hashed_predictions=True,
        )


def test_diagnostics_reject_metric_mismatch(tmp_path: Path) -> None:
    manifest = _write_diagnostic_run(tmp_path)
    result_dir = tmp_path / "results" / "cell"
    holistic = result_dir / "_test" / "test_holistic.yaml"
    holistic.write_text(
        "accuracy:\n  all: 0.75\n"
        "uar:\n  all: 0.3333333333333333\n"
        "f1:\n  all: 0.2222222222222222\n",
        encoding="utf-8",
    )
    provenance_path = result_dir / "noisegap_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["artifacts"]["test_holistic"]["sha256"] = sha256_file(holistic)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="Recomputed accuracy mismatch"):
        diagnose_manifests([manifest], tmp_path / "diagnostics.csv")


def test_diagnostics_requires_explicit_unhashed_test_split(tmp_path: Path) -> None:
    manifest = _write_diagnostic_run(tmp_path, hash_test_split=False)
    with pytest.raises(ValueError, match="Missing test split provenance"):
        diagnose_manifests([manifest], tmp_path / "strict.csv")

    output = tmp_path / "legacy.csv"
    diagnose_manifests(
        [manifest],
        output,
        unhashed_test_split=tmp_path / "dataset" / "test.csv",
    )
    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["test_split_provenance_verified"] == "False"
