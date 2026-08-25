"""Aggregate provenance-checked result rows across independent seeds."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

CORE_GROUP_FIELDS = (
    "phase",
    "train_domain",
    "test_domain",
    "train_snr_db",
    "test_snr_db",
)

MODEL_GROUP_FIELDS = ("model", "model_config")

NON_METRIC_FIELDS = {
    "experiment_id",
    "seed",
    "git_revision",
    "git_dirty",
    "resolved_config_sha256",
    "input_checkpoint_sha256",
    "output_checkpoint_sha256",
    *CORE_GROUP_FIELDS,
    *MODEL_GROUP_FIELDS,
}


def _read_inputs(input_paths: Sequence[Path]) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    expected_fields: set[str] | None = None
    for input_path in input_paths:
        with input_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or [])
            current_rows = list(reader)
        if not current_rows:
            raise ValueError(f"Seed summary input is empty: {input_path}")
        if expected_fields is None:
            expected_fields = fields
        elif fields != expected_fields:
            raise ValueError(
                "Seed summary inputs have different columns; refusing to merge "
                f"{input_path}."
            )
        rows.extend(current_rows)
    return rows, expected_fields or set()


def aggregate_seed_summary(
    input_path: Path | Sequence[Path],
    output_path: Path,
) -> int:
    """Write mean, sample SD, range, and raw values for every metric."""
    input_paths = [input_path] if isinstance(input_path, Path) else list(input_path)
    if not input_paths:
        raise ValueError("At least one seed summary input is required.")
    rows, fields = _read_inputs(input_paths)
    missing = [field for field in (*CORE_GROUP_FIELDS, "seed") if field not in fields]
    if missing:
        raise ValueError(f"Seed summary is missing fields: {missing}")

    model_fields = [field for field in MODEL_GROUP_FIELDS if field in fields]
    if model_fields and len(model_fields) != len(MODEL_GROUP_FIELDS):
        raise ValueError(
            "Seed summary must contain both model and model_config, or neither."
        )
    group_fields = CORE_GROUP_FIELDS + (
        MODEL_GROUP_FIELDS if model_fields else ()
    )

    metric_names = sorted(fields - NON_METRIC_FIELDS)
    if not metric_names:
        raise ValueError("Seed summary contains no metric columns.")
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    output_rows = []
    for key, group in sorted(grouped.items()):
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in group {key}: {seeds}")
        output: dict[str, object] = dict(zip(group_fields, key, strict=True))
        output["n_seeds"] = len(seeds)
        output["seeds"] = json.dumps(sorted(seeds))
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row.get(metric)]
            if not values:
                continue
            output[f"{metric}_mean"] = statistics.fmean(values)
            output[f"{metric}_sd"] = (
                statistics.stdev(values) if len(values) > 1 else ""
            )
            output[f"{metric}_min"] = min(values)
            output[f"{metric}_max"] = max(values)
            output[f"{metric}_values"] = json.dumps(values)
        output_rows.append(output)

    output_fields = list(group_fields) + ["n_seeds", "seeds"]
    metric_fields = sorted(
        {field for row in output_rows for field in row if field not in output_fields}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields + metric_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = aggregate_seed_summary(args.input, args.output)
    print(f"Wrote {groups} seed-aggregated condition(s).")


if __name__ == "__main__":
    main()
