"""Prepare sentence-type classification metadata from licensed TIMIT."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from autrainer.datasets import BaseClassificationDataset


class TimitSentenceType(BaseClassificationDataset):
    """autrainer dataset whose files are prepared by `noisegap-prepare-timit`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        path = kwargs.get("path", args[0] if args else None)
        if path is None:
            raise TypeError("TimitSentenceType requires path.")
        validate_timit_metadata(Path(path))
        super().__init__(*args, **kwargs)

    @staticmethod
    def download(path: str) -> None:
        root = Path(path)
        required = [root / f"{split}.csv" for split in ("train", "dev", "test")]
        required.append(root / "default")
        if not all(item.exists() for item in required):
            raise RuntimeError(
                "NoiseGap does not download TIMIT. Obtain licensed LDC93S1 data "
                "and run noisegap-prepare-timit before autrainer fetch."
            )


def _read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        required = {"path", "sentence_type"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}.")
        return list(rows)


def validate_timit_metadata(
    root: Path,
    *,
    audio_subdir: str = "default",
) -> dict[str, int]:
    """Verify path, speaker, label, and file separation for all TIMIT splits."""
    split_paths: dict[str, set[Path]] = {}
    split_speakers: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    audio_root = root / audio_subdir
    for split in ("train", "dev", "test"):
        metadata = root / f"{split}.csv"
        if not metadata.is_file():
            raise FileNotFoundError(f"Missing TIMIT metadata: {metadata}")
        rows = _read_metadata(metadata)
        if not rows:
            raise ValueError(f"TIMIT {split} split must not be empty.")
        paths = [Path(row["path"]) for row in rows]
        unsafe = [path for path in paths if path.is_absolute() or ".." in path.parts]
        if unsafe:
            raise ValueError(f"TIMIT {split} path escapes dataset root: {unsafe[0]}")
        if len(paths) != len(set(paths)):
            raise ValueError(f"TIMIT {split} contains duplicate utterance paths.")
        for row, path in zip(rows, paths, strict=True):
            label = row["sentence_type"].upper()
            if label not in {"SA", "SI", "SX"}:
                raise ValueError(f"Unexpected TIMIT label '{label}' in {metadata}.")
            if path.stem[:2].upper() != label:
                raise ValueError(f"TIMIT label/path mismatch for {path}: {label}.")
            if len(path.parts) < 2:
                raise ValueError(f"Cannot derive TIMIT speaker from {path}.")
            if not (audio_root / path).is_file():
                raise FileNotFoundError(f"Missing TIMIT utterance: {audio_root / path}")
        split_paths[split] = set(paths)
        split_speakers[split] = {path.parts[-2] for path in paths}
        counts[split] = len(paths)

    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        path_overlap = split_paths[first] & split_paths[second]
        if path_overlap:
            raise ValueError(
                f"TIMIT utterance leakage between {first}/{second}: "
                f"{sorted(path_overlap)[0]}"
            )
        speaker_overlap = split_speakers[first] & split_speakers[second]
        if speaker_overlap:
            raise ValueError(
                f"TIMIT speaker leakage between {first}/{second}: "
                f"{sorted(speaker_overlap)[0]}"
            )
    return counts


def _collect(root: Path, split: str) -> list[dict[str, str]]:
    split_root = root / split
    if not split_root.is_dir():
        split_root = root / split.lower()
    if not split_root.is_dir():
        raise ValueError(f"Missing official TIMIT {split} directory in {root}.")

    rows = []
    for path in sorted(split_root.rglob("*")):
        if path.suffix.lower() != ".wav":
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) < 4:
            raise ValueError(f"Unexpected TIMIT path layout: {relative}.")
        speaker = parts[-2]
        sentence_type = path.stem[:2].upper()
        if sentence_type not in {"SA", "SI", "SX"}:
            raise ValueError(f"Unexpected sentence type in {relative}.")
        rows.append(
            {
                "path": relative.as_posix(),
                "speaker_id": speaker,
                "sentence_type": sentence_type,
            }
        )
    if not rows:
        raise ValueError(f"No WAV files found in {split_root}.")
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "sentence_type"])
        writer.writeheader()
        writer.writerows(
            {"path": row["path"], "sentence_type": row["sentence_type"]} for row in rows
        )


def prepare_timit(
    source: Path,
    output: Path,
    *,
    dev_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Preserve official TEST and split official TRAIN by speaker."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1.")
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Prepared output must be outside the TIMIT source tree.")

    default = output / "default"
    if (default.exists() or default.is_symlink()) and default.resolve() != source:
        raise ValueError(f"{default} already points to another dataset.")

    official_train = _collect(source, "TRAIN")
    official_test = _collect(source, "TEST")
    speakers = sorted({row["speaker_id"] for row in official_train})
    if len(speakers) < 2:
        raise ValueError(
            "At least two official TRAIN speakers are required for a "
            "speaker-disjoint train/dev split."
        )
    random.Random(seed).shuffle(speakers)
    dev_count = min(
        len(speakers) - 1,
        max(1, round(len(speakers) * dev_fraction)),
    )
    dev_speakers = set(speakers[:dev_count])

    train = [row for row in official_train if row["speaker_id"] not in dev_speakers]
    dev = [row for row in official_train if row["speaker_id"] in dev_speakers]
    test = official_test
    _write_rows(output / "train.csv", train)
    _write_rows(output / "dev.csv", dev)
    _write_rows(output / "test.csv", test)

    if not default.exists() and not default.is_symlink():
        default.symlink_to(source, target_is_directory=True)

    summary = {"train": len(train), "dev": len(dev), "test": len(test)}
    (output / "split_manifest.json").write_text(
        json.dumps(
            {
                "catalog": "LDC93S1",
                "policy": "official_test_train_speaker_dev",
                "seed": seed,
                "dev_fraction": dev_fraction,
                "counts": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_timit(
        args.source,
        args.output,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    print(
        "Prepared licensed TIMIT metadata: "
        + ", ".join(f"{split}={count}" for split, count in summary.items())
    )


if __name__ == "__main__":
    main()
