import json
from pathlib import Path

import pytest

from noisegap.experiments.run import (
    command_for,
    load_manifest,
    select_runs,
)


def _records() -> list[dict]:
    return [
        {
            "phase": "train",
            "config": "configs/train.yaml",
            "checkpoint": "/tmp/train/_best/model.pt",
            "depends_on": None,
        },
        {
            "phase": "evaluate",
            "config": "configs/evaluate.yaml",
            "checkpoint": "/tmp/train/_best/model.pt",
            "depends_on": "configs/train.yaml",
        },
    ]


def test_runner_selects_index_within_phase() -> None:
    records = _records()

    assert select_runs(records, "train", 0) == [records[0]]
    assert select_runs(records, "evaluate", 0) == [records[1]]
    with pytest.raises(IndexError, match="outside"):
        select_runs(records, "train", 1)


def test_manifest_schema_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"phase": "train"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain"):
        load_manifest(manifest)


def test_runner_command_uses_config_directory_and_name(tmp_path: Path) -> None:
    config = tmp_path / "configs" / "train_SS.yaml"

    command = command_for(config, ["device=cpu"])

    assert command[-5:] == [
        "--config-dir",
        str(config.parent),
        "--config-name",
        "train_SS",
        "device=cpu",
    ]
