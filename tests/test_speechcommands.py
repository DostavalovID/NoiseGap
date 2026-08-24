import csv
from pathlib import Path

from noisegap.datasets.speechcommands import prepare_background_noise_manifests


def _read_paths(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["path"] for row in csv.DictReader(stream)}


def test_background_noise_split_is_file_disjoint_and_reproducible(
    tmp_path: Path,
) -> None:
    root = tmp_path / "SpeechCommands"
    noise_root = root / "default" / "_background_noise_"
    noise_root.mkdir(parents=True)
    for index in range(6):
        (noise_root / f"noise_{index}.wav").touch()

    first = tmp_path / "first"
    second = tmp_path / "second"
    summary = prepare_background_noise_manifests(root, first, seed=17)
    prepare_background_noise_manifests(root, second, seed=17)
    train = _read_paths(first / "train.csv")
    dev = _read_paths(first / "dev.csv")
    test = _read_paths(first / "test.csv")

    assert summary["train_files"] == 2
    assert summary["dev_files"] == 2
    assert summary["test_files"] == 2
    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)
    assert train | dev | test == {f"noise_{index}.wav" for index in range(6)}
    assert (first / "train.csv").read_bytes() == (second / "train.csv").read_bytes()
    assert (first / "dev.csv").read_bytes() == (second / "dev.csv").read_bytes()
    assert (first / "test.csv").read_bytes() == (second / "test.csv").read_bytes()
