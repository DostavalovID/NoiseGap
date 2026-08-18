"""Aggregate provenance-checked result rows across independent seeds."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

GROUP_FIELDS = (
    "phase",
    "model",
    "model_config",
    "train_domain",
    "test_domain",
    "train_snr_db",
    "test_snr_db",
)

NON_METRIC_FIELDS = {
    "experiment_id",
    "seed",
    "git_revision",
    "git_dirty",
    "resolved_config_sha256",
    "input_checkpoint_sha256",
    "output_checkpoint_sha256",
    *GROUP_FIELDS,
}


def aggregate_seed_summary(input_path: Path, output_path: Path) -> int:
    """Write mean, sample SD, range, and raw values for every metric."""
    with input_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Seed summary input is empty.")
    missing = [field for field in (*GROUP_FIELDS, "seed") if field not in rows[0]]
    if missing:
        raise ValueError(f"Seed summary is missing fields: {missing}")

    metric_names = sorted(set(rows[0]) - NON_METRIC_FIELDS)
    if not metric_names:
        raise ValueError("Seed summary contains no metric columns.")
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in GROUP_FIELDS)].append(row)

    output_rows = []
    for key, group in sorted(grouped.items()):
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seeds in group {key}: {seeds}")
        output: dict[str, object] = dict(zip(GROUP_FIELDS, key, strict=True))
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

    fields = list(GROUP_FIELDS) + ["n_seeds", "seeds"]
    metric_fields = sorted(
        {field for row in output_rows for field in row if field not in fields}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + metric_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = aggregate_seed_summary(args.input, args.output)
    print(f"Wrote {groups} seed-aggregated condition(s).")


if __name__ == "__main__":
    main()
