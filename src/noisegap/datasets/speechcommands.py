"""Prepare file-disjoint SpeechCommands background-noise manifests."""

import argparse
import csv
import json
import random
from pathlib import Path


def _write_manifest(path: Path, files: list[Path], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path"])
        writer.writeheader()
        writer.writerows({"path": file.relative_to(root).as_posix()} for file in files)


def prepare_background_noise_manifests(
    speechcommands_root: Path,
    output_dir: Path,
    *,
    test_fraction: float = 1 / 3,
    seed: int = 0,
) -> dict[str, object]:
    """Split SpeechCommands background WAV files by source file."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    noise_root = speechcommands_root / "default" / "_background_noise_"
    files = sorted(noise_root.glob("*.wav"))
    if len(files) < 2:
        raise ValueError(
            "Expected at least two SpeechCommands background-noise WAV files "
            f"under {noise_root}."
        )

    random.Random(seed).shuffle(files)
    test_count = min(len(files) - 1, max(1, round(len(files) * test_fraction)))
    test_files = sorted(files[:test_count])
    train_files = sorted(files[test_count:])
    train_manifest = output_dir / "train.csv"
    test_manifest = output_dir / "test.csv"
    _write_manifest(train_manifest, train_files, noise_root)
    _write_manifest(test_manifest, test_files, noise_root)

    summary: dict[str, object] = {
        "source": "SpeechCommands-v0.02/_background_noise_",
        "split_unit": "source_file",
        "seed": seed,
        "test_fraction": test_fraction,
        "noise_root": str(noise_root.resolve()),
        "train_files": len(train_files),
        "test_files": len(test_files),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speechcommands-root",
        type=Path,
        default=Path("data/SpeechCommands"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/SpeechCommands-background-noise"),
    )
    parser.add_argument("--test-fraction", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = prepare_background_noise_manifests(
        args.speechcommands_root,
        args.output,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(
        "Prepared file-disjoint SpeechCommands background noise: "
        f"train={summary['train_files']}, test={summary['test_files']}"
    )


if __name__ == "__main__":
    main()
