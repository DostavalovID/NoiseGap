"""Per-cell class-balance and collapse diagnostics for completed matrices."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from .analysis import _resolved_config_hash, _validate_protocol
from .experiments.run import load_manifest
from .training.cli import sha256_file

_PREDICTION_METADATA_FIELDS = {"index", "predictions"}


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ValueError(f"CSV is empty or has no header: {path}")
    return fields, rows


def _verify_artifact(
    experiment_id: str,
    path: Path,
    artifact: dict[str, Any] | None,
) -> bool:
    """Verify an artifact when the run provenance contains its hash."""
    if not artifact:
        return False
    recorded_path = Path(str(artifact.get("path", ""))).resolve()
    expected_hash = artifact.get("sha256")
    if recorded_path != path.resolve() or not expected_hash:
        raise ValueError(
            f"Artifact provenance mismatch for {experiment_id}: {path}"
        )
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Artifact hash mismatch for {experiment_id}: {path}")
    return True


def _metric_value(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name, {}).get("all")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Missing or invalid metric '{name}': {value}")
    return float(value)


def _classification_metrics(
    classes: list[str],
    confusion: dict[str, dict[str, int]],
) -> tuple[float, float, float]:
    total = sum(sum(row.values()) for row in confusion.values())
    if total <= 0:
        raise ValueError("Cannot calculate metrics for an empty confusion matrix.")

    accuracy = sum(confusion[label][label] for label in classes) / total
    recalls = []
    f1_scores = []
    for label in classes:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = (
            sum(confusion[actual][label] for actual in classes) - true_positive
        )
        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        recall = (
            true_positive / recall_denominator if recall_denominator > 0 else 0.0
        )
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)
    return accuracy, sum(recalls) / len(recalls), sum(f1_scores) / len(f1_scores)


def _diagnose_record(
    record: dict[str, Any],
    *,
    collapse_threshold: float,
    require_hashed_predictions: bool,
) -> dict[str, object]:
    experiment_id = str(record["experiment_id"])
    result_dir = Path(record["result_dir"])
    provenance_path = result_dir / "noisegap_provenance.json"
    config_path = result_dir / ".hydra" / "config.yaml"
    holistic_path = result_dir / "_test" / "test_holistic.yaml"
    predictions_path = result_dir / "_test" / "test_results.csv"
    for path in (provenance_path, config_path, holistic_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing artifact for {experiment_id}: {path}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _validate_protocol(record, provenance)
    if provenance.get("resolved_config_sha256") != _resolved_config_hash(config_path):
        raise ValueError(f"Resolved config hash mismatch: {experiment_id}")

    artifacts = provenance.get("artifacts", {})
    if not _verify_artifact(
        experiment_id,
        holistic_path,
        artifacts.get("test_holistic"),
    ):
        raise ValueError(f"Missing test_holistic provenance: {experiment_id}")
    predictions_verified = _verify_artifact(
        experiment_id,
        predictions_path,
        artifacts.get("test_results"),
    )
    if require_hashed_predictions and not predictions_verified:
        raise ValueError(f"Missing test_results provenance: {experiment_id}")

    split_artifact = provenance.get("input_metadata", {}).get(
        "dataset_splits", {}
    ).get("test")
    if not split_artifact:
        raise ValueError(f"Missing test split provenance: {experiment_id}")
    split_path = Path(str(split_artifact.get("path", "")))
    if not split_path.is_file() or sha256_file(split_path) != split_artifact.get(
        "sha256"
    ):
        raise ValueError(f"Test split hash mismatch: {experiment_id}")

    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    target_column = str(config.dataset.target_column)
    prediction_fields, prediction_rows = _read_csv(predictions_path)
    _, target_rows = _read_csv(split_path)
    if target_column not in target_rows[0]:
        raise ValueError(
            f"Target column '{target_column}' is absent from {split_path}."
        )
    classes = [
        field
        for field in prediction_fields
        if field not in _PREDICTION_METADATA_FIELDS
    ]
    if not classes:
        raise ValueError(f"No class columns in {predictions_path}")

    confusion = {actual: dict.fromkeys(classes, 0) for actual in classes}
    prediction_counts = dict.fromkeys(classes, 0)
    target_counts = dict.fromkeys(classes, 0)
    seen_indices: set[int] = set()
    for prediction_row in prediction_rows:
        try:
            index = int(prediction_row["index"])
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"Invalid prediction index for {experiment_id}: {prediction_row}"
            ) from error
        if index in seen_indices or not 0 <= index < len(target_rows):
            raise ValueError(
                f"Invalid or duplicate test index {index}: {experiment_id}"
            )
        seen_indices.add(index)
        actual = str(target_rows[index][target_column])
        predicted = str(prediction_row.get("predictions", ""))
        if actual not in confusion or predicted not in prediction_counts:
            raise ValueError(
                f"Unknown target/prediction for {experiment_id}: "
                f"{actual!r}/{predicted!r}"
            )
        confusion[actual][predicted] += 1
        prediction_counts[predicted] += 1
        target_counts[actual] += 1
    if seen_indices != set(range(len(target_rows))):
        raise ValueError(
            f"Predictions do not cover the complete test split: {experiment_id}"
        )

    accuracy, uar, f1 = _classification_metrics(classes, confusion)
    holistic = yaml.safe_load(holistic_path.read_text(encoding="utf-8"))
    recomputed = {"accuracy": accuracy, "uar": uar, "f1": f1}
    for name, value in recomputed.items():
        reported = _metric_value(holistic, name)
        if not math.isclose(value, reported, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"Recomputed {name} mismatch for {experiment_id}: "
                f"{value} != {reported}"
            )

    total = len(prediction_rows)
    majority_label = max(classes, key=prediction_counts.__getitem__)
    majority_share = prediction_counts[majority_label] / total
    output: dict[str, object] = {
        key: record[key]
        for key in (
            "experiment_id",
            "phase",
            "model",
            "model_config",
            "seed",
            "train_domain",
            "test_domain",
            "train_snr_db",
            "test_snr_db",
        )
        if key in record
    }
    output.update(
        {
            "git_revision": provenance.get("git_revision"),
            "num_examples": total,
            "num_classes": len(classes),
            "accuracy": accuracy,
            "uar": uar,
            "f1": f1,
            "majority_prediction": majority_label,
            "majority_prediction_share": majority_share,
            "collapse_threshold": collapse_threshold,
            "collapsed": majority_share >= collapse_threshold,
            "test_results_sha256": sha256_file(predictions_path),
            "test_results_provenance_verified": predictions_verified,
        }
    )
    for label in classes:
        output[f"target_share_{label}"] = target_counts[label] / total
        output[f"prediction_share_{label}"] = prediction_counts[label] / total
        actual_total = sum(confusion[label].values())
        output[f"recall_{label}"] = confusion[label][label] / actual_total
        for predicted in classes:
            output[f"confusion_{label}_as_{predicted}"] = confusion[label][predicted]
    return output


def diagnose_manifests(
    manifest_paths: list[Path],
    output_path: Path,
    *,
    collapse_threshold: float = 0.9,
    require_hashed_predictions: bool = False,
) -> int:
    """Write one verified class-balance diagnostic row per completed cell."""
    if not 0.0 < collapse_threshold <= 1.0:
        raise ValueError("collapse_threshold must be in (0, 1].")
    if not manifest_paths:
        raise ValueError("At least one manifest is required.")

    rows = []
    identities: set[tuple[object, ...]] = set()
    for manifest_path in manifest_paths:
        for record in load_manifest(manifest_path.resolve()):
            row = _diagnose_record(
                record,
                collapse_threshold=collapse_threshold,
                require_hashed_predictions=require_hashed_predictions,
            )
            identity = (
                row.get("model"),
                row.get("seed"),
                row["train_domain"],
                row["test_domain"],
                row["train_snr_db"],
                row["test_snr_db"],
            )
            if identity in identities:
                raise ValueError(f"Duplicate result cell: {identity}")
            identities.add(identity)
            rows.append(row)

    if not rows:
        raise RuntimeError("No completed result cells found.")
    preferred_fields = [
        "experiment_id",
        "phase",
        "model",
        "model_config",
        "seed",
        "train_domain",
        "test_domain",
        "train_snr_db",
        "test_snr_db",
        "git_revision",
        "num_examples",
        "num_classes",
        "accuracy",
        "uar",
        "f1",
        "majority_prediction",
        "majority_prediction_share",
        "collapse_threshold",
        "collapsed",
        "test_results_sha256",
        "test_results_provenance_verified",
    ]
    present_preferred = [
        field for field in preferred_fields if any(field in row for row in rows)
    ]
    extra_fields = sorted(
        {field for row in rows for field in row} - set(present_preferred)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=present_preferred + extra_fields,
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collapse-threshold", type=float, default=0.9)
    parser.add_argument("--require-hashed-predictions", action="store_true")
    args = parser.parse_args()
    count = diagnose_manifests(
        args.manifest,
        args.output,
        collapse_threshold=args.collapse_threshold,
        require_hashed_predictions=args.require_hashed_predictions,
    )
    print(f"Wrote {count} class-balance diagnostic cell(s).")


if __name__ == "__main__":
    main()
