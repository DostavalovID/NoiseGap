import json
from pathlib import Path

from omegaconf import OmegaConf

from noisegap.training.cli import (
    build_provenance,
    finalize_provenance,
    sha256_file,
)


def test_provenance_binds_resolved_config_and_protocol(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for split in ("train", "dev", "test"):
        (dataset / f"{split}.csv").write_text(
            f"path,target\n{split}.wav,label\n",
            encoding="utf-8",
        )
    noise = tmp_path / "noise.csv"
    noise.write_text("path\nnoise.wav\n", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "value": "${answer}",
            "answer": 42,
            "dataset": {"path": str(dataset)},
            "augmentation": {
                "train": {
                    "pipeline": [{"RecordedNoise": {"manifest_csv": str(noise)}}]
                },
                "dev": {
                    "pipeline": [{"LegacyRecordedNoise": {"noise_csv": str(noise)}}]
                },
                "test": None,
            },
            "noisegap_protocol": {
                "phase": "train",
                "feature_layout": "channel,time,mel",
            },
        }
    )

    provenance = build_provenance(cfg)

    assert provenance["protocol"]["phase"] == "train"
    assert provenance["protocol"]["feature_layout"] == "channel,time,mel"
    assert len(provenance["resolved_config_sha256"]) == 64
    assert provenance["python_version"]
    assert provenance["autrainer_version"] == "0.8.1"
    assert set(provenance["input_metadata"]["dataset_splits"]) == {
        "train",
        "dev",
        "test",
    }
    assert provenance["input_metadata"]["noise_manifests"]["train"][0][
        "sha256"
    ] == sha256_file(noise)
    assert provenance["input_metadata"]["noise_manifests"]["dev"][0][
        "sha256"
    ] == sha256_file(noise)


def test_provenance_binds_successful_artifacts(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "iterations": 1,
            "model": {},
            "noisegap_protocol": {"phase": "train"},
        }
    )
    provenance_path = tmp_path / "noisegap_provenance.json"
    provenance_path.write_text(
        json.dumps(build_provenance(cfg)),
        encoding="utf-8",
    )
    test_artifact = tmp_path / "_test" / "test_holistic.yaml"
    test_artifact.parent.mkdir()
    test_artifact.write_text("accuracy:\n  all: 0.5\n", encoding="utf-8")
    test_results = tmp_path / "_test" / "test_results.csv"
    test_results.write_text(
        "index,predictions,A,B\n0,A,1.0,0.0\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "_best" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    finalize_provenance(provenance_path, tmp_path, cfg)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert provenance["artifacts"]["test_holistic"]["sha256"] == sha256_file(
        test_artifact
    )
    assert provenance["artifacts"]["test_results"]["sha256"] == sha256_file(
        test_results
    )
    assert provenance["artifacts"]["output_checkpoint"]["sha256"] == sha256_file(
        checkpoint
    )
