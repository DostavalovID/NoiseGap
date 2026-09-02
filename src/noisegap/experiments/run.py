"""Execute a generated experiment manifest without duplicating sweep logic."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from noisegap.features import verify_feature_manifest


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate a NoiseGap manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Manifest must be a non-empty JSON list.")
    required = {"phase", "config", "checkpoint", "depends_on"}
    for index, record in enumerate(payload):
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError(
                f"Manifest record {index} must contain {sorted(required)}."
            )
        if record["phase"] not in {"train", "evaluate"}:
            raise ValueError(
                f"Manifest record {index} has invalid phase: {record['phase']}"
            )
    return payload


def select_runs(
    records: list[dict[str, Any]],
    phase: str,
    index: int | None,
) -> list[dict[str, Any]]:
    """Filter manifest records; index is relative to the selected phase."""
    selected = (
        records
        if phase == "all"
        else [record for record in records if record["phase"] == phase]
    )
    if index is None:
        return selected
    if index < 0 or index >= len(selected):
        raise IndexError(
            f"Run index {index} is outside selected range 0..{len(selected) - 1}."
        )
    return [selected[index]]


def command_for(config: Path, overrides: list[str]) -> list[str]:
    """Build one interpreter-stable training command."""
    return [
        sys.executable,
        "-m",
        "noisegap.training.cli",
        "--config-dir",
        str(config.parent),
        "--config-name",
        config.stem,
        *overrides,
    ]


def run_manifest(
    manifest_path: Path,
    *,
    phase: str,
    index: int | None,
    dry_run: bool,
    overrides: list[str],
) -> None:
    """Execute selected records and verify their checkpoint dependency."""
    manifest_path = manifest_path.resolve()
    records = select_runs(load_manifest(manifest_path), phase, index)
    feature_manifests = {
        Path(record["feature_manifest"]).resolve()
        for record in records
        if record.get("feature_manifest")
    }
    for feature_manifest in sorted(feature_manifests):
        verify_feature_manifest(feature_manifest)
    environment = os.environ.copy()
    repository_conf = Path(__file__).resolve().parents[3] / "conf"
    if repository_conf.is_dir():
        environment.setdefault("NOISEGAP_CONFIG_DIR", str(repository_conf))

    for record in records:
        config = manifest_path.parent / record["config"]
        checkpoint = Path(record["checkpoint"])
        if not config.is_file():
            raise FileNotFoundError(f"Generated config does not exist: {config}")
        if record["phase"] == "evaluate" and not checkpoint.is_file() and not dry_run:
            raise FileNotFoundError(
                f"Evaluation dependency is incomplete; checkpoint missing: {checkpoint}"
            )

        command = command_for(config, overrides)
        if dry_run:
            print(shlex.join(command))
            continue
        subprocess.run(command, check=True, env=environment)
        if record["phase"] == "train" and not checkpoint.is_file():
            raise RuntimeError(
                f"Training command succeeded but checkpoint is missing: {checkpoint}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("all", "train", "evaluate"),
        default="all",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Zero-based index within the selected phase, suitable for job arrays.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides forwarded to noisegap-train.",
    )
    args = parser.parse_args()
    run_manifest(
        args.manifest,
        phase=args.phase,
        index=args.index,
        dry_run=args.dry_run,
        overrides=args.overrides,
    )


if __name__ == "__main__":
    main()
