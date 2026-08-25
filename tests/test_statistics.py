import csv
import json
from pathlib import Path

import pytest

from noisegap.statistics import aggregate_seed_summary


def _write_summary(path: Path) -> None:
    fields = [
        "experiment_id",
        "phase",
        "model",
        "model_config",
        "seed",
        "train_domain",
        "test_domain",
        "train_snr_db",
        "test_snr_db",
        "accuracy",
        "uar",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed, accuracy, uar in ((0, 0.7, 0.6), (1, 0.8, 0.7), (2, 0.9, 0.8)):
            writer.writerow(
                {
                    "experiment_id": f"cnn10_s{seed}_GG_train20_test20",
                    "phase": "train",
                    "model": "CNN10-online-PANN",
                    "model_config": "Cnn10-32k-T-waveform",
                    "seed": seed,
                    "train_domain": "GaussianWaveform",
                    "test_domain": "GaussianWaveform",
                    "train_snr_db": 20,
                    "test_snr_db": 20,
                    "accuracy": accuracy,
                    "uar": uar,
                }
            )


def test_seed_aggregation_retains_mean_sd_and_raw_values(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    output = tmp_path / "aggregate.csv"
    _write_summary(source)

    count = aggregate_seed_summary(source, output)

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert count == 1
    assert row["n_seeds"] == "3"
    assert float(row["accuracy_mean"]) == pytest.approx(0.8)
    assert float(row["accuracy_sd"]) == pytest.approx(0.1)
    assert json.loads(row["accuracy_values"]) == [0.7, 0.8, 0.9]


def test_seed_aggregation_accepts_legacy_feature_summaries_without_model_fields(
    tmp_path: Path,
) -> None:
    inputs = []
    for seed, accuracy in ((0, 0.7), (1, 0.9)):
        source = tmp_path / f"seed-{seed}.csv"
        with source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "experiment_id",
                    "phase",
                    "seed",
                    "train_domain",
                    "test_domain",
                    "train_snr_db",
                    "test_snr_db",
                    "accuracy",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "experiment_id": f"SS_seed_{seed}",
                    "phase": "train",
                    "seed": seed,
                    "train_domain": "SyntheticLogMel",
                    "test_domain": "SyntheticLogMel",
                    "train_snr_db": 20,
                    "test_snr_db": 20,
                    "accuracy": accuracy,
                }
            )
        inputs.append(source)

    output = tmp_path / "aggregate.csv"
    count = aggregate_seed_summary(inputs, output)

    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert count == 1
    assert row["n_seeds"] == "2"
    assert json.loads(row["seeds"]) == [0, 1]
    assert float(row["accuracy_mean"]) == pytest.approx(0.8)
    assert "model" not in row


def test_seed_aggregation_rejects_partial_model_identity(tmp_path: Path) -> None:
    source = tmp_path / "summary.csv"
    _write_summary(source)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    with source.open("w", newline="", encoding="utf-8") as stream:
        fields = [field for field in rows[0] if field != "model_config"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="both model and model_config"):
        aggregate_seed_summary(source, tmp_path / "aggregate.csv")
