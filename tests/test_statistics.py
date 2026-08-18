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
