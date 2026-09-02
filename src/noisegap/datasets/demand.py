"""Download and prepare environment-disjoint DEMAND noise manifests."""

import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from noisegap.experiments.matrix import validate_recorded_manifest_split

ZENODO_RECORD = "1227121"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"
DOWNLOAD_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{{filename}}/content"
)

# Official Zenodo MD5 values for the available 16 kHz environment archives.
DEMAND_ARCHIVES = {
    "DKITCHEN": "7ffbf52d7f4699f96927846103dc8788",
    "DLIVING": "46741384d9e434a0bd8b3ec1830b6052",
    "DWASHING": "7e5ee9437ce9409c5f9a779b6212a240",
    "NFIELD": "a740046c6f4e174e16f5d568aaec5024",
    "NPARK": "80f1385a34d7f1705758926b57f138ce",
    "NRIVER": "54264db61d3fe073fb81f2e40e0d19b5",
    "OHALLWAY": "fe918bbb0e63e73d09ba7f4843ef33f1",
    "OMEETING": "62f7cfe7fe6d30b7d8a215fe37c2dfd2",
    "OOFFICE": "7b61cc2d182d5a654cb9c3101ddd4041",
    "PCAFETER": "99927d148128254141a9417d051510bb",
    "PRESTO": "b98d2e6854eeebb397f29a8ad7457092",
    "PSTATION": "d7448009f6c2aeb6ba570375df1750a3",
    "SPSQUARE": "205d0e7b8fe74504a2f8d252fc414b9e",
    "STRAFFIC": "2efa87262f272bbf9ba578088e81939c",
    "TBUS": "706b11b0d8504f9f3b3f3211e91b3863",
    "TCAR": "4d930012796bd298932245a26189f973",
    "TMETRO": "95daf4df678e13b120e14211e6d89571",
}

DEMAND_SPLITS = {
    "train": ("DKITCHEN", "NFIELD", "OOFFICE", "PCAFETER", "SPSQUARE", "TBUS"),
    "dev": ("DLIVING", "NPARK", "OMEETING", "PRESTO", "TCAR"),
    "test": ("DWASHING", "NRIVER", "OHALLWAY", "PSTATION", "STRAFFIC", "TMETRO"),
}

CATEGORIES = {
    "D": "domestic",
    "N": "nature",
    "O": "office",
    "P": "public",
    "S": "street",
    "T": "transportation",
}


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(raw_root: Path, environment: str, expected_md5: str) -> Path:
    filename = f"{environment}_16k.zip"
    output = raw_root / filename
    if output.is_file():
        actual_md5 = _digest(output, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(
                f"Existing DEMAND archive checksum mismatch for {output}: {actual_md5}"
            )
        return output

    partial = output.with_suffix(".zip.part")
    request = urllib.request.Request(DOWNLOAD_URL.format(filename=filename))
    existing_size = partial.stat().st_size if partial.is_file() else 0
    if existing_size:
        request.add_header("Range", f"bytes={existing_size}-")
    with urllib.request.urlopen(request) as response:
        append = existing_size > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
    actual_md5 = _digest(partial, "md5")
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Downloaded DEMAND archive checksum mismatch for {filename}: {actual_md5}"
        )
    partial.replace(output)
    return output


def _validate_wav(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as stream:
        metadata = {
            "sample_rate": stream.getframerate(),
            "channels": stream.getnchannels(),
            "sample_width_bytes": stream.getsampwidth(),
            "frames": stream.getnframes(),
        }
    if metadata["sample_rate"] != 16000 or metadata["channels"] != 1:
        raise ValueError(f"Unexpected DEMAND WAV format for {path}: {metadata}")
    return metadata


def _extract_channel_one(archive: Path, output_root: Path, environment: str) -> Path:
    member = f"{environment}/ch01.wav"
    output = output_root / member
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(".wav.part")
        with zipfile.ZipFile(archive) as bundle:
            if member not in bundle.namelist():
                raise FileNotFoundError(f"Missing {member} in {archive}")
            with bundle.open(member) as source, partial.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        partial.replace(output)
    _validate_wav(output)
    return output


def _write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("path", "environment", "category", "sha256"),
        )
        writer.writeheader()
        writer.writerows(rows)


def prepare_demand_noise(
    raw_root: Path,
    output: Path,
    *,
    download: bool = False,
    workers: int = 4,
    archives: dict[str, str] | None = None,
    splits: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Use one channel per environment and split only by environment."""
    if workers < 1:
        raise ValueError("workers must be positive.")
    archives = DEMAND_ARCHIVES if archives is None else archives
    splits = DEMAND_SPLITS if splits is None else splits
    assigned = [environment for values in splits.values() for environment in values]
    if len(assigned) != len(set(assigned)):
        raise ValueError("DEMAND environment appears in more than one split.")
    if set(assigned) != set(archives):
        raise ValueError("DEMAND split assignment must cover every configured archive.")

    raw_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if download:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                environment: executor.submit(
                    _download_archive,
                    raw_root,
                    environment,
                    expected_md5,
                )
                for environment, expected_md5 in archives.items()
            }
            archive_paths = {
                environment: future.result() for environment, future in futures.items()
            }
    else:
        archive_paths = {}
        for environment, expected_md5 in archives.items():
            archive = raw_root / f"{environment}_16k.zip"
            if not archive.is_file():
                raise FileNotFoundError(
                    f"Missing {archive}; rerun with --download to fetch it."
                )
            if _digest(archive, "md5") != expected_md5:
                raise ValueError(f"DEMAND archive checksum mismatch: {archive}")
            archive_paths[environment] = archive

    audio_root = output / "default"
    extracted = {
        environment: _extract_channel_one(
            archive_paths[environment], audio_root, environment
        )
        for environment in sorted(archives)
    }
    split_rows = {}
    for split in ("train", "dev", "test"):
        rows = []
        for environment in splits[split]:
            path = extracted[environment]
            rows.append(
                {
                    "path": path.relative_to(audio_root).as_posix(),
                    "environment": environment,
                    "category": CATEGORIES.get(environment[0], environment[0]),
                    "sha256": _digest(path, "sha256"),
                }
            )
        split_rows[split] = rows
        _write_split(output / f"{split}.csv", rows)

    validate_recorded_manifest_split(
        audio_root,
        output / "train.csv",
        output / "dev.csv",
        output / "test.csv",
    )
    wav_metadata = {
        environment: _validate_wav(path) for environment, path in extracted.items()
    }
    manifest = {
        "schema_version": 1,
        "dataset": "DEMAND",
        "source_record": ZENODO_RECORD_URL,
        "zenodo_record_id": ZENODO_RECORD,
        "split_unit": "environment",
        "channel_policy": "ch01_only",
        "sample_rate": 16000,
        "archive_hash_algorithm": "md5_from_official_record",
        "archives": archives,
        "splits": {key: list(value) for key, value in splits.items()},
        "counts": {key: len(value) for key, value in split_rows.items()},
        "wav_metadata": wav_metadata,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    manifest = prepare_demand_noise(
        args.raw_root,
        args.output,
        download=args.download,
        workers=args.workers,
    )
    print(
        "Prepared environment-disjoint DEMAND noise: "
        + ", ".join(f"{split}={count}" for split, count in manifest["counts"].items())
    )


if __name__ == "__main__":
    main()
