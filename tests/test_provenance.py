import json
from pathlib import Path

from omegaconf import OmegaConf

from noisegap.training.cli import (
    build_provenance,
    finalize_provenance,
    sha256_file,
)


def test_provenance_binds_resolved_config_and_protocol() -> None:
    cfg = OmegaConf.create(
        {
            "value": "${answer}",
            "answer": 42,
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
    checkpoint = tmp_path / "_best" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    finalize_provenance(provenance_path, tmp_path, cfg)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert provenance["artifacts"]["test_holistic"]["sha256"] == sha256_file(
        test_artifact
    )
    assert provenance["artifacts"]["output_checkpoint"]["sha256"] == sha256_file(
        checkpoint
    )
