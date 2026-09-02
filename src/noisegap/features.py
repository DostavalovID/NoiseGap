"""Index and verify precomputed TIMIT feature artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from autrainer import instantiate_shorthand
from autrainer.core.structs import DataItem
from autrainer.transforms import SmartCompose

from noisegap.datasets.timit import validate_timit_metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_rows(dataset_root: Path) -> dict[str, list[dict[str, str]]]:
    result = {}
    for split in ("train", "dev", "test"):
        path = dataset_root / f"{split}.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            result[split] = list(csv.DictReader(stream))
    return result


def _feature_path(relative_audio_path: str) -> Path:
    return Path(relative_audio_path).with_suffix(".npy")


def _load_preprocessing(config_path: Path) -> tuple[Any, SmartCompose]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"file_handler", "pipeline"}.issubset(
        config
    ):
        raise ValueError("Preprocessing config must contain file_handler and pipeline.")
    handler = instantiate_shorthand(config["file_handler"])
    pipeline = SmartCompose(
        [instantiate_shorthand(transform) for transform in config["pipeline"]]
    )
    return handler, pipeline


def _parity_paths(
    rows: dict[str, list[dict[str, str]]],
    samples_per_split: int,
) -> list[tuple[str, str]]:
    selected = []
    for split, split_rows in rows.items():
        count = min(samples_per_split, len(split_rows))
        if count == 0:
            continue
        indices = np.linspace(0, len(split_rows) - 1, count, dtype=int)
        selected.extend((split, split_rows[index]["path"]) for index in indices)
    return selected


def build_feature_manifest(
    dataset_root: Path,
    feature_subdir: str,
    preprocessing_config: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: str = "float32",
    parity_samples_per_split: int = 3,
) -> dict[str, Any]:
    """Validate features and return a content-addressed artifact manifest."""
    dataset_root = dataset_root.resolve()
    preprocessing_config = preprocessing_config.resolve()
    validate_timit_metadata(dataset_root)
    rows = _split_rows(dataset_root)
    feature_root = dataset_root / feature_subdir
    if not feature_root.is_dir():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_root}")

    expected = {
        _feature_path(row["path"]) for split_rows in rows.values() for row in split_rows
    }
    actual = {
        path.relative_to(feature_root)
        for path in feature_root.rglob("*.npy")
        if path.is_file()
    }
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise FileNotFoundError(
            f"Missing precomputed feature: {feature_root / sorted(missing)[0]}"
        )
    if extra:
        raise ValueError(
            f"Unexpected precomputed feature: {feature_root / sorted(extra)[0]}"
        )

    items = []
    aggregate = hashlib.sha256()
    for split, split_rows in rows.items():
        for row in split_rows:
            relative = _feature_path(row["path"])
            path = feature_root / relative
            array = np.load(path, allow_pickle=False)
            if array.shape != expected_shape:
                raise ValueError(
                    f"Unexpected shape for {path}: {array.shape}, "
                    f"expected {expected_shape}"
                )
            if str(array.dtype) != expected_dtype:
                raise ValueError(
                    f"Unexpected dtype for {path}: {array.dtype}, "
                    f"expected {expected_dtype}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite values in precomputed feature: {path}")
            digest = _sha256_file(path)
            aggregate.update(relative.as_posix().encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
            items.append(
                {
                    "split": split,
                    "source_path": row["path"],
                    "feature_path": relative.as_posix(),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "pann_floor_frame_count": int(
                        np.all(np.isclose(array, -100.0, atol=0.01), axis=(0, 2)).sum()
                    ),
                }
            )

    handler, pipeline = _load_preprocessing(preprocessing_config)
    parity = []
    for split, relative_audio in _parity_paths(rows, parity_samples_per_split):
        audio_path = dataset_root / "default" / relative_audio
        cached_path = feature_root / _feature_path(relative_audio)
        cached = np.load(cached_path, allow_pickle=False)
        waveform = handler.load(str(audio_path))
        online = pipeline(DataItem(waveform, 0, 0)).features.detach().cpu().numpy()
        if not np.array_equal(cached, online):
            difference = float(np.max(np.abs(cached - online)))
            raise ValueError(
                f"Cached/online frontend mismatch for {audio_path}: "
                f"max_abs={difference}"
            )
        parity.append(
            {
                "split": split,
                "source_path": relative_audio,
                "feature_path": _feature_path(relative_audio).as_posix(),
                "exact_equal": True,
                "max_abs_error": 0.0,
            }
        )

    split_artifacts = {}
    for split in ("train", "dev", "test"):
        path = (dataset_root / f"{split}.csv").resolve()
        split_artifacts[split] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "count": len(rows[split]),
        }
    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "feature_root": str(feature_root.resolve()),
        "feature_subdir": feature_subdir,
        "preprocessing_config": {
            "path": str(preprocessing_config),
            "sha256": _sha256_file(preprocessing_config),
        },
        "expected_shape": list(expected_shape),
        "expected_dtype": expected_dtype,
        "split_artifacts": split_artifacts,
        "feature_count": len(items),
        "feature_set_sha256": aggregate.hexdigest(),
        "parity_samples_per_split": parity_samples_per_split,
        "parity": parity,
        "items": items,
    }


def verify_feature_manifest(path: Path) -> dict[str, Any]:
    """Fail closed if a feature manifest or any indexed artifact changed."""
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported feature manifest schema: {path}")
    feature_root = Path(manifest["feature_root"])
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Feature manifest has no items: {path}")

    relative_paths = [Path(item["feature_path"]) for item in items]
    unsafe = [
        item
        for item in relative_paths
        if item.is_absolute() or ".." in item.parts
    ]
    if unsafe:
        raise ValueError(f"Feature manifest path escapes its root: {unsafe[0]}")
    expected_paths = set(relative_paths)
    if len(expected_paths) != len(relative_paths):
        raise ValueError("Feature manifest contains duplicate feature paths.")
    if manifest.get("feature_count") not in {None, len(items)}:
        raise ValueError("Feature manifest count does not match its items.")
    actual_paths = {
        item.relative_to(feature_root)
        for item in feature_root.rglob("*.npy")
        if item.is_file()
    }
    if expected_paths != actual_paths:
        raise ValueError("Feature file set no longer matches its manifest.")

    aggregate = hashlib.sha256()
    for item in items:
        relative = Path(item["feature_path"])
        feature = feature_root / relative
        digest = _sha256_file(feature)
        if digest != item["sha256"]:
            raise ValueError(f"Feature hash mismatch: {feature}")
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if aggregate.hexdigest() != manifest["feature_set_sha256"]:
        raise ValueError("Feature set aggregate hash mismatch.")

    for artifact in manifest["split_artifacts"].values():
        artifact_path = Path(artifact["path"])
        if _sha256_file(artifact_path) != artifact["sha256"]:
            raise ValueError(f"Dataset split hash mismatch: {artifact_path}")
    preprocessing = manifest["preprocessing_config"]
    preprocessing_path = Path(preprocessing["path"])
    if _sha256_file(preprocessing_path) != preprocessing["sha256"]:
        raise ValueError(f"Preprocessing config hash mismatch: {preprocessing_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--feature-subdir", required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shape", type=int, nargs="+", required=True)
    parser.add_argument("--expected-dtype", default="float32")
    parser.add_argument("--parity-samples-per-split", type=int, default=3)
    args = parser.parse_args()
    if args.parity_samples_per_split < 1:
        raise ValueError("At least one parity sample per split is required.")
    manifest = build_feature_manifest(
        args.dataset_root,
        args.feature_subdir,
        args.preprocessing_config,
        expected_shape=tuple(args.expected_shape),
        expected_dtype=args.expected_dtype,
        parity_samples_per_split=args.parity_samples_per_split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Verified {manifest['feature_count']} features; "
        f"set_sha256={manifest['feature_set_sha256']}"
    )


if __name__ == "__main__":
    main()
