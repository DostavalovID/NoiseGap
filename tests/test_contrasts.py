import csv
import json
from pathlib import Path

import pytest

from noisegap.contrasts import summarize_directional_contrasts


def _write_diagnostics(path: Path) -> None:
    fields = [
        "experiment_id",
        "seed",
        "train_domain",
        "test_domain",
        "train_snr_db",
        "test_snr_db",
        "accuracy",
        "uar",
        "f1",
        "collapsed",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed, offset in ((0, 0.0), (1, 0.1), (2, 0.2)):
            for train_snr in (0, 10):
                for test_snr in (0, 10):
                    for train_domain, test_domain, base in (
                        ("G", "R", 0.7),
                        ("R", "G", 0.6),
                    ):
                        writer.writerow(
                            {
                                "experiment_id": (
                                    f"{seed}-{train_domain}{test_domain}-"
                                    f"{train_snr}-{test_snr}"
                                ),
                                "seed": seed,
                                "train_domain": train_domain,
                                "test_domain": test_domain,
                                "train_snr_db": train_snr,
                                "test_snr_db": test_snr,
                                "accuracy": base + offset,
                                "uar": base - 0.1 + offset,
                                "f1": base - 0.2 + offset,
                                "collapsed": train_snr == 0,
                            }
                        )


def test_directional_contrasts_use_seed_level_pairs(tmp_path: Path) -> None:
    source = tmp_path / "diagnostics.csv"
    output = tmp_path / "contrasts.csv"
    _write_diagnostics(source)

    count = summarize_directional_contrasts(
        source,
        output,
        pipeline="feature",
        gaussian_domain="G",
        recorded_domain="R",
    )

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert count == 24
    overall_accuracy = next(
        row
        for row in rows
        if row["profile"] == "overall" and row["metric"] == "accuracy"
    )
    assert overall_accuracy["n_seeds"] == "3"
    assert overall_accuracy["cells_per_direction_per_seed"] == "4"
    assert float(overall_accuracy["difference_mean"]) == pytest.approx(0.1)
    assert json.loads(overall_accuracy["difference_values"]) == pytest.approx(
        [0.1, 0.1, 0.1]
    )
    assert float(overall_accuracy["paired_t_p_value_uncorrected"]) < 1e-20
    assert overall_accuracy["paired_t_holm_family_size"] == "1"
    assert float(
        overall_accuracy["paired_t_p_value_holm_within_profile"]
    ) == pytest.approx(float(overall_accuracy["paired_t_p_value_uncorrected"]))

    test_snr_accuracy = [
        row
        for row in rows
        if row["profile"] == "test_snr" and row["metric"] == "accuracy"
    ]
    assert len(test_snr_accuracy) == 2
    assert {row["paired_t_holm_family_size"] for row in test_snr_accuracy} == {"2"}
    assert all(
        float(row["paired_t_p_value_holm_within_profile"])
        >= float(row["paired_t_p_value_uncorrected"])
        for row in test_snr_accuracy
    )

    train_zero_collapse = next(
        row
        for row in rows
        if row["profile"] == "train_snr"
        and row["snr_db"] == "0"
        and row["metric"] == "collapse_rate"
    )
    assert float(train_zero_collapse["gaussian_to_recorded_mean"]) == 1.0
    assert float(train_zero_collapse["recorded_to_gaussian_mean"]) == 1.0
    assert float(train_zero_collapse["difference_mean"]) == 0.0


def test_directional_contrasts_reject_unpaired_cells(tmp_path: Path) -> None:
    source = tmp_path / "diagnostics.csv"
    _write_diagnostics(source)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
        fields = list(rows[0])
    rows.pop()
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Unpaired direction cells"):
        summarize_directional_contrasts(
            source,
            tmp_path / "contrasts.csv",
            pipeline="feature",
            gaussian_domain="G",
            recorded_domain="R",
        )
