"""Seed-level directional contrasts from per-cell diagnostic tables."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from scipy import stats

_METRICS = ("accuracy", "uar", "f1", "collapse_rate")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Diagnostic input is empty: {path}")
    required = {
        "seed",
        "train_domain",
        "test_domain",
        "train_snr_db",
        "test_snr_db",
        "accuracy",
        "uar",
        "f1",
        "collapsed",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Diagnostic input is missing fields: {sorted(missing)}")
    return rows


def _bool_value(value: str) -> float:
    normalized = value.strip().lower()
    if normalized == "true":
        return 1.0
    if normalized == "false":
        return 0.0
    raise ValueError(f"Invalid boolean value: {value!r}")


def _metric_value(row: dict[str, str], metric: str) -> float:
    value = _bool_value(row["collapsed"]) if metric == "collapse_rate" else float(
        row[metric]
    )
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {metric}: {value}")
    return value


def _aggregate(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) == 1:
        return mean, math.nan, math.nan, math.nan
    sd = statistics.stdev(values)
    critical = float(stats.t.ppf(0.975, df=len(values) - 1))
    margin = critical * sd / math.sqrt(len(values))
    return mean, sd, mean - margin, mean + margin


def _paired_test(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return math.nan, math.nan
    sd = statistics.stdev(values)
    mean = statistics.fmean(values)
    if sd == 0.0:
        if mean == 0.0:
            return 0.0, 1.0
        return math.copysign(math.inf, mean), 0.0
    statistic = mean / (sd / math.sqrt(len(values)))
    p_value = float(2.0 * stats.t.sf(abs(statistic), df=len(values) - 1))
    return statistic, p_value


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original input order."""
    if not p_values:
        return []
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("Holm correction requires finite p-values in [0, 1].")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_maximum = 0.0
    for rank, (index, p_value) in enumerate(ordered):
        candidate = min(1.0, (len(p_values) - rank) * p_value)
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    return adjusted


def _profile_specs(
    rows: list[dict[str, str]],
) -> list[tuple[str, str, Callable[[dict[str, str]], bool]]]:
    specs: list[tuple[str, str, Callable[[dict[str, str]], bool]]] = [
        ("overall", "", lambda row: True),
        (
            "matched_snr",
            "",
            lambda row: row["train_snr_db"] == row["test_snr_db"],
        ),
    ]
    train_snrs = sorted({int(row["train_snr_db"]) for row in rows})
    test_snrs = sorted({int(row["test_snr_db"]) for row in rows})
    specs.extend(
        (
            "train_snr",
            str(snr),
            lambda row, snr=snr: int(row["train_snr_db"]) == snr,
        )
        for snr in train_snrs
    )
    specs.extend(
        (
            "test_snr",
            str(snr),
            lambda row, snr=snr: int(row["test_snr_db"]) == snr,
        )
        for snr in test_snrs
    )
    return specs


def summarize_directional_contrasts(
    input_path: Path,
    output_path: Path,
    *,
    pipeline: str,
    gaussian_domain: str,
    recorded_domain: str,
) -> int:
    """Aggregate paired G-to-R minus R-to-G differences at seed level."""
    if not pipeline:
        raise ValueError("pipeline must be non-empty.")
    if gaussian_domain == recorded_domain:
        raise ValueError("Gaussian and recorded domains must differ.")
    all_rows = _read_rows(input_path)
    rows = [
        row
        for row in all_rows
        if (row["train_domain"], row["test_domain"])
        in {
            (gaussian_domain, recorded_domain),
            (recorded_domain, gaussian_domain),
        }
    ]
    if not rows:
        raise ValueError("No requested cross-domain cells were found.")

    model_fields = tuple(
        field for field in ("model", "model_config") if field in rows[0]
    )
    model_groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        model_groups[tuple(row[field] for field in model_fields)].append(row)

    outputs = []
    for model_key, model_rows in sorted(model_groups.items()):
        for profile, snr_db, selector in _profile_specs(model_rows):
            selected = [row for row in model_rows if selector(row)]
            if not selected:
                continue
            for metric in _METRICS:
                by_seed_direction: dict[
                    tuple[int, str, str], list[dict[str, str]]
                ] = defaultdict(list)
                for row in selected:
                    by_seed_direction[
                        (
                            int(row["seed"]),
                            row["train_domain"],
                            row["test_domain"],
                        )
                    ].append(row)
                seeds = sorted({int(row["seed"]) for row in selected})
                g_to_r_values = []
                r_to_g_values = []
                cells_per_seed = []
                for seed in seeds:
                    g_to_r = by_seed_direction[
                        (seed, gaussian_domain, recorded_domain)
                    ]
                    r_to_g = by_seed_direction[
                        (seed, recorded_domain, gaussian_domain)
                    ]
                    g_coordinates = {
                        (row["train_snr_db"], row["test_snr_db"])
                        for row in g_to_r
                    }
                    r_coordinates = {
                        (row["train_snr_db"], row["test_snr_db"])
                        for row in r_to_g
                    }
                    if not g_to_r or g_coordinates != r_coordinates:
                        raise ValueError(
                            f"Unpaired direction cells for seed {seed}, "
                            f"profile {profile}:{snr_db}."
                        )
                    g_to_r_values.append(
                        statistics.fmean(
                            _metric_value(row, metric) for row in g_to_r
                        )
                    )
                    r_to_g_values.append(
                        statistics.fmean(
                            _metric_value(row, metric) for row in r_to_g
                        )
                    )
                    cells_per_seed.append(len(g_to_r))
                if len(set(cells_per_seed)) != 1:
                    raise ValueError(
                        f"Unequal cell counts across seeds for {profile}:{snr_db}."
                    )
                differences = [
                    g_to_r - r_to_g
                    for g_to_r, r_to_g in zip(
                        g_to_r_values,
                        r_to_g_values,
                        strict=True,
                    )
                ]
                g_mean, g_sd, _, _ = _aggregate(g_to_r_values)
                r_mean, r_sd, _, _ = _aggregate(r_to_g_values)
                difference_mean, difference_sd, ci_low, ci_high = _aggregate(
                    differences
                )
                t_statistic, p_value = _paired_test(differences)
                output: dict[str, object] = {
                    "pipeline": pipeline,
                    **dict(zip(model_fields, model_key, strict=True)),
                    "profile": profile,
                    "snr_db": snr_db,
                    "metric": metric,
                    "n_seeds": len(seeds),
                    "seeds": json.dumps(seeds),
                    "cells_per_direction_per_seed": cells_per_seed[0],
                    "gaussian_to_recorded_mean": g_mean,
                    "gaussian_to_recorded_sd": g_sd,
                    "recorded_to_gaussian_mean": r_mean,
                    "recorded_to_gaussian_sd": r_sd,
                    "difference_mean": difference_mean,
                    "difference_sd": difference_sd,
                    "difference_ci95_low": ci_low,
                    "difference_ci95_high": ci_high,
                    "paired_t_statistic": t_statistic,
                    "paired_t_p_value_uncorrected": p_value,
                    "gaussian_to_recorded_values": json.dumps(g_to_r_values),
                    "recorded_to_gaussian_values": json.dumps(r_to_g_values),
                    "difference_values": json.dumps(differences),
                }
                outputs.append(output)

    family_fields = (*model_fields, "profile", "metric")
    families: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for output in outputs:
        families[
            tuple(str(output[field]) for field in family_fields)
        ].append(output)
    for family in families.values():
        finite_rows = [
            row
            for row in family
            if math.isfinite(float(row["paired_t_p_value_uncorrected"]))
        ]
        adjusted = _holm_adjust(
            [float(row["paired_t_p_value_uncorrected"]) for row in finite_rows]
        )
        for row in family:
            row["paired_t_holm_family_size"] = len(finite_rows)
            row["paired_t_p_value_holm_within_profile"] = math.nan
        for row, adjusted_p_value in zip(finite_rows, adjusted, strict=True):
            row["paired_t_p_value_holm_within_profile"] = adjusted_p_value

    fixed_fields = ["pipeline", *model_fields, "profile", "snr_db", "metric"]
    remaining_fields = [
        field for field in outputs[0] if field not in fixed_fields
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fixed_fields + remaining_fields,
        )
        writer.writeheader()
        writer.writerows(outputs)
    return len(outputs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--gaussian-domain", required=True)
    parser.add_argument("--recorded-domain", required=True)
    args = parser.parse_args()
    count = summarize_directional_contrasts(
        args.input,
        args.output,
        pipeline=args.pipeline,
        gaussian_domain=args.gaussian_domain,
        recorded_domain=args.recorded_domain,
    )
    print(f"Wrote {count} seed-level directional contrast row(s).")


if __name__ == "__main__":
    main()
