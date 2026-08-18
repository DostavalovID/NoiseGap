"""Provenance-checked tabular summary of a completed experiment matrix."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from .experiments.run import load_manifest
from .training.cli import sha256_file


def _resolved_config_hash(path: Path) -> str:
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def _validate_protocol(record: dict[str, Any], provenance: dict[str, Any]) -> None:
    protocol = provenance.get("protocol", {})
    expected = {
        "phase": record["phase"],
        "train_domain": record["train_domain"],
        "test_domain": record["test_domain"],
        "train_snr_db": record["train_snr_db"],
        "test_snr_db": record["test_snr_db"],
    }
    for key in ("model", "model_config", "seed"):
        if key in record:
            expected[key] = record[key]
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Protocol mismatch for {record['experiment_id']}: {mismatches}"
        )


def summarize_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    allow_incomplete: bool = False,
    allow_uncommitted: bool = False,
) -> tuple[int, int]:
    """Validate completed records and write one CSV row per result."""
    manifest_path = manifest_path.resolve()
    records = load_manifest(manifest_path)
    rows = []
    missing = []
    metric_names: set[str] = set()

    for record in records:
        result_dir = Path(record["result_dir"])
        test_artifact = result_dir / "_test" / "test_holistic.yaml"
        provenance_path = result_dir / "noisegap_provenance.json"
        resolved_config = result_dir / ".hydra" / "config.yaml"
        required_paths = (test_artifact, provenance_path, resolved_config)
        absent = [path for path in required_paths if not path.is_file()]
        if absent:
            missing.append(
                {
                    "experiment_id": record["experiment_id"],
                    "missing": [str(path) for path in absent],
                }
            )
            continue

        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_protocol(record, provenance)
        if not allow_uncommitted and (
            not provenance.get("git_revision")
            or provenance.get("git_dirty") is not False
        ):
            raise ValueError(
                "Committed clean Git provenance is required for "
                f"{record['experiment_id']}; use --allow-uncommitted only "
                "for local diagnostics."
            )
        if provenance["resolved_config_sha256"] != _resolved_config_hash(
            resolved_config
        ):
            raise ValueError(
                f"Resolved config hash mismatch: {record['experiment_id']}"
            )
        artifact_record = provenance.get("artifacts", {}).get("test_holistic")
        if not artifact_record or artifact_record.get("sha256") != sha256_file(
            test_artifact
        ):
            raise ValueError(f"Test artifact hash mismatch: {record['experiment_id']}")

        metrics = yaml.safe_load(test_artifact.read_text(encoding="utf-8"))
        flattened = {
            name: values["all"]
            for name, values in metrics.items()
            if isinstance(values, dict) and "all" in values
        }
        metric_names.update(flattened)
        input_checkpoint = provenance.get("input_checkpoint", {})
        output_checkpoint = provenance.get("artifacts", {}).get(
            "output_checkpoint",
            {},
        )
        rows.append(
            {
                "experiment_id": record["experiment_id"],
                "phase": record["phase"],
                **{
                    key: record[key]
                    for key in ("model", "model_config", "seed")
                    if key in record
                },
                "train_domain": record["train_domain"],
                "test_domain": record["test_domain"],
                "train_snr_db": record["train_snr_db"],
                "test_snr_db": record["test_snr_db"],
                "git_revision": provenance.get("git_revision"),
                "git_dirty": provenance.get("git_dirty"),
                "resolved_config_sha256": provenance["resolved_config_sha256"],
                "input_checkpoint_sha256": input_checkpoint.get("sha256"),
                "output_checkpoint_sha256": output_checkpoint.get("sha256"),
                **flattened,
            }
        )

    if missing and not allow_incomplete:
        raise RuntimeError(
            f"Matrix is incomplete: {len(missing)} of {len(records)} result(s) "
            f"missing; first: {missing[0]}"
        )
    if not rows:
        raise RuntimeError("No completed, provenance-valid results found.")

    fixed_fields = [
        "experiment_id",
        "phase",
    ]
    fixed_fields.extend(
        key
        for key in ("model", "model_config", "seed")
        if any(key in row for row in rows)
    )
    fixed_fields.extend(
        [
            "train_domain",
            "test_domain",
            "train_snr_db",
            "test_snr_db",
            "git_revision",
            "git_dirty",
            "resolved_config_sha256",
            "input_checkpoint_sha256",
            "output_checkpoint_sha256",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fixed_fields + sorted(metric_names),
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), len(missing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--allow-uncommitted", action="store_true")
    args = parser.parse_args()
    completed, missing = summarize_manifest(
        args.manifest,
        args.output,
        allow_incomplete=args.allow_incomplete,
        allow_uncommitted=args.allow_uncommitted,
    )
    print(f"Wrote {completed} verified result(s); missing={missing}.")


if __name__ == "__main__":
    main()
