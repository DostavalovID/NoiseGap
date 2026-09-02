import csv
from pathlib import Path

import numpy as np
import pytest

from noisegap.features import verify_feature_manifest


def _manifest(tmp_path: Path) -> Path:
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    feature = feature_root / "sample.npy"
    np.save(feature, np.zeros((1, 2, 3), dtype=np.float32))
    import hashlib
    import json

    digest = hashlib.sha256(feature.read_bytes()).hexdigest()
    aggregate = hashlib.sha256(b"sample.npy\0" + digest.encode() + b"\n").hexdigest()
    split_artifacts = {}
    for split in ("train", "dev", "test"):
        split_path = tmp_path / f"{split}.csv"
        with split_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "sentence_type"])
            writer.writerow(["sample.WAV", "SA"])
        split_artifacts[split] = {
            "path": str(split_path),
            "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "count": 1,
        }
    preprocessing = tmp_path / "preprocessing.yaml"
    preprocessing.write_text("pipeline: []\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_root": str(feature_root),
                "feature_set_sha256": aggregate,
                "split_artifacts": split_artifacts,
                "preprocessing_config": {
                    "path": str(preprocessing),
                    "sha256": hashlib.sha256(preprocessing.read_bytes()).hexdigest(),
                },
                "items": [{"feature_path": "sample.npy", "sha256": digest}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_feature_manifest_verifies_indexed_artifacts(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    verified = verify_feature_manifest(manifest)

    assert verified["schema_version"] == 1


def test_feature_manifest_rejects_changed_feature(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    np.save(tmp_path / "features" / "sample.npy", np.ones((1, 2, 3)))

    with pytest.raises(ValueError, match="Feature hash mismatch"):
        verify_feature_manifest(manifest)
