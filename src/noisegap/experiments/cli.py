"""Generate configs and a machine-readable experiment manifest."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .matrix import Domain, SweepSpec, build_matrix, validate_recorded_manifest_split
from .render import render_config, write_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--recorded-train-csv", type=Path, required=True)
    parser.add_argument("--recorded-dev-csv", type=Path, required=True)
    parser.add_argument("--recorded-test-csv", type=Path, required=True)
    parser.add_argument(
        "--snr",
        type=int,
        nargs="+",
        default=[-5, 0, 10, 20, 30, 40],
    )
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base-config",
        default="noisegap_base",
        help=(
            "Hydra base config used by every generated run. Use an article-specific "
            "base to keep dataset and frontend provenance explicit."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.resolve()
    recorded_root = args.recorded_root.resolve()
    train_manifest = args.recorded_train_csv.resolve()
    dev_manifest = args.recorded_dev_csv.resolve()
    test_manifest = args.recorded_test_csv.resolve()
    if not recorded_root.is_dir():
        raise FileNotFoundError(f"Recorded-noise root does not exist: {recorded_root}")
    for manifest in (train_manifest, dev_manifest, test_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Recorded-noise manifest does not exist: {manifest}"
            )
    validate_recorded_manifest_split(
        recorded_root,
        train_manifest,
        dev_manifest,
        test_manifest,
    )
    spec = SweepSpec(
        domains=(
            Domain("S", "SyntheticLogMel", "synthetic"),
            Domain(
                "R",
                "RecordedNoise",
                "recorded",
                root=recorded_root,
                train_manifest=train_manifest,
                dev_manifest=dev_manifest,
                test_manifest=test_manifest,
            ),
        ),
        snr_levels=tuple(args.snr),
        iterations=args.iterations,
    )
    runs = build_matrix(spec)
    configs_dir = output / "configs"
    results_dir = output / "results"

    manifest = []
    for run in runs:
        config_name = f"{run.phase.value}_{run.experiment_id}.yaml"
        write_config(
            configs_dir / config_name,
            render_config(
                run,
                results_dir=results_dir,
                base_config=args.base_config,
                seed=args.seed,
            ),
        )
        record = asdict(run)
        record["phase"] = run.phase.value
        record["experiment_id"] = run.experiment_id
        record["base_config"] = args.base_config
        record["seed"] = args.seed
        record["train_domain"] = run.train_domain.label
        record["test_domain"] = run.test_domain.label
        record["config"] = str(Path("configs") / config_name)
        record["result_dir"] = str(results_dir / run.experiment_id)
        record["checkpoint"] = str(
            results_dir / run.training_experiment_id / "_best" / "model.pt"
        )
        record["depends_on"] = (
            None
            if run.phase.value == "train"
            else str(Path("configs") / f"train_{run.training_experiment_id}.yaml")
        )
        manifest.append(record)

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    train_count = sum(run.phase.value == "train" for run in runs)
    print(
        f"Generated {len(runs)} configs: {train_count} train, "
        f"{len(runs) - train_count} evaluate."
    )


if __name__ == "__main__":
    main()
