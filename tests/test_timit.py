import csv
from pathlib import Path

import pytest

from noisegap.datasets.timit import prepare_timit, validate_timit_metadata


def _touch_utterance(root: Path, split: str, speaker: str, name: str) -> None:
    path = root / split / "DR1" / speaker / f"{name}.WAV"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _read_paths(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["path"] for row in csv.DictReader(stream)}


def test_official_test_is_preserved_and_train_dev_are_disjoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "TIMIT"
    for speaker in ("MABC0", "FDEF0", "MGHI0", "FJKL0"):
        _touch_utterance(source, "TRAIN", speaker, "SA1")
        _touch_utterance(source, "TRAIN", speaker, "SX1")
    _touch_utterance(source, "TEST", "MTEST0", "SI1")
    output = tmp_path / "prepared"

    summary = prepare_timit(source, output, dev_fraction=0.25, seed=7)
    train = _read_paths(output / "train.csv")
    dev = _read_paths(output / "dev.csv")
    test = _read_paths(output / "test.csv")

    assert summary == {"train": 6, "dev": 2, "test": 1}
    assert train.isdisjoint(dev)
    assert test == {"TEST/DR1/MTEST0/SI1.WAV"}
    assert (output / "default").resolve() == source.resolve()
    assert validate_timit_metadata(output) == summary


def test_existing_wrong_data_link_fails_before_metadata_overwrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "TIMIT"
    source.mkdir()
    output = tmp_path / "prepared"
    (output / "default").mkdir(parents=True)
    sentinel = output / "train.csv"
    sentinel.write_text("do-not-overwrite\n", encoding="utf-8")

    with pytest.raises(ValueError, match="another dataset"):
        prepare_timit(source, output)

    assert sentinel.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_timit_metadata_rejects_speaker_overlap(tmp_path: Path) -> None:
    root = tmp_path / "prepared"
    audio = root / "default" / "TRAIN" / "DR1" / "MSAME0"
    audio.mkdir(parents=True)
    for name in ("SA1", "SX1", "SI1"):
        (audio / f"{name}.WAV").touch()
    (root / "train.csv").write_text(
        "path,sentence_type\nTRAIN/DR1/MSAME0/SA1.WAV,SA\n",
        encoding="utf-8",
    )
    (root / "dev.csv").write_text(
        "path,sentence_type\nTRAIN/DR1/MSAME0/SX1.WAV,SX\n",
        encoding="utf-8",
    )
    (root / "test.csv").write_text(
        "path,sentence_type\nTRAIN/DR1/MSAME0/SI1.WAV,SI\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="speaker leakage"):
        validate_timit_metadata(root)
