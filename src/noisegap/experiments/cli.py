"""Generate configs and a machine-readable experiment manifest."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from noisegap.features import verify_feature_manifest

from .matrix import Domain, SweepSpec, build_matrix, validate_recorded_manifest_split
from .render import render_config, snapshot_config_tree, write_config


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
        help="Legacy shorthand that sets both --train-snr and --test-snr.",
    )
    parser.add_argument(
        "--train-snr",
        type=int,
        nargs="+",
        help="Training SNR levels; defaults to -5 0 10 20 30 40.",
    )
    parser.add_argument(
        "--test-snr",
        type=int,
        nargs="+",
        help="Evaluation SNR levels; defaults to -5 0 10 20 30 40.",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        help="Required content manifest for matched_32k cached clean features.",
    )
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--feature-implementation",
        choices=("corrected", "article_legacy", "matched_32k"),
        default="corrected",
        help=(
            "Use corrected feature mixing or reproduce the historical article "
            "augmentation implementation, including its recorded-noise axis behavior."
        ),
    )
    parser.add_argument(
        "--base-config",
        default="noisegap_base",
        help=(
            "Hydra base config used by every generated run. Use an article-specific "
            "base to keep dataset and frontend provenance explicit."
        ),
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        help=(
            "DataLoader workers. Defaults to 4 for matched_32k and 0 for "
            "the other feature implementations."
        ),
    )
    parser.add_argument(
        "--tracking-metric",
        choices=("accuracy", "uar", "f1"),
        help=(
            "Checkpoint-selection metric. Defaults to UAR for matched_32k "
            "and Accuracy for reproduction modes."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.snr is not None and (
        args.train_snr is not None or args.test_snr is not None
    ):
        raise ValueError("Use either --snr or --train-snr/--test-snr, not both.")
    default_snr = (-5, 0, 10, 20, 30, 40)
    train_snr_levels = tuple(args.snr or args.train_snr or default_snr)
    test_snr_levels = tuple(args.snr or args.test_snr or default_snr)
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
    loader_workers = args.loader_workers
    if loader_workers is None:
        loader_workers = 4 if args.feature_implementation == "matched_32k" else 0
    if loader_workers < 0:
        raise ValueError("loader_workers must be non-negative.")
    metric_names = {
        "accuracy": "autrainer.metrics.Accuracy",
        "uar": "autrainer.metrics.UAR",
        "f1": "autrainer.metrics.F1",
    }
    tracking_metric_key = args.tracking_metric
    if tracking_metric_key is None:
        tracking_metric_key = (
            "uar" if args.feature_implementation == "matched_32k" else "accuracy"
        )
    tracking_metric = metric_names[tracking_metric_key]
    feature_manifest = args.feature_manifest
    if args.feature_implementation == "matched_32k":
        if feature_manifest is None:
            raise ValueError("matched_32k requires --feature-manifest.")
        feature_manifest = feature_manifest.resolve()
        verify_feature_manifest(feature_manifest)
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
        train_snr_levels=train_snr_levels,
        test_snr_levels=test_snr_levels,
        iterations=args.iterations,
    )
    runs = build_matrix(spec)
    configs_dir = output / "configs"
    results_dir = output / "results"
    config_snapshot_count = snapshot_config_tree(configs_dir)

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
                feature_implementation=args.feature_implementation,
                loader_workers=loader_workers,
                tracking_metric=tracking_metric,
                feature_manifest=feature_manifest,
            ),
        )
        record = asdict(run)
        record["phase"] = run.phase.value
        record["experiment_id"] = run.experiment_id
        record["base_config"] = args.base_config
        record["seed"] = args.seed
        record["feature_implementation"] = args.feature_implementation
        record["loader_workers"] = loader_workers
        record["tracking_metric"] = tracking_metric
        record["feature_manifest"] = (
            str(feature_manifest) if feature_manifest is not None else None
        )
        record["config_snapshot_count"] = config_snapshot_count
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
