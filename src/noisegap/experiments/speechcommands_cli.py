"""Generate a controlled waveform-level experiment matrix."""

import argparse
import json
from pathlib import Path

from .matrix import Domain, validate_recorded_manifest_split
from .render import write_config
from .waveform_matrix import ModelSpec, WaveformSweepSpec, build_waveform_matrix
from .waveform_render import render_waveform_config

MODELS = {
    "cnn10": ModelSpec(
        code="cnn10",
        label="CNN10-online-PANN",
        config="Cnn10-32k-T-waveform",
        optimizer="Adam",
        batch_size=64,
        learning_rate=0.001,
    ),
    "ast": ModelSpec(
        code="ast",
        label="AST-online-HF",
        config="ASTModel-T-waveform",
        optimizer="AdamW",
        batch_size=16,
        learning_rate=0.0001,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recorded-root",
        type=Path,
        default=Path("data/SpeechCommands/default/_background_noise_"),
    )
    parser.add_argument(
        "--recorded-train-csv",
        type=Path,
        default=Path("data/SpeechCommands-background-noise/train.csv"),
    )
    parser.add_argument(
        "--recorded-test-csv",
        type=Path,
        default=Path("data/SpeechCommands-background-noise/test.csv"),
    )
    parser.add_argument(
        "--recorded-dev-csv",
        type=Path,
        default=Path("data/SpeechCommands-background-noise/dev.csv"),
    )
    parser.add_argument(
        "--models",
        choices=tuple(MODELS),
        nargs="+",
        default=list(MODELS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-snr", type=int, nargs="+", default=[20])
    parser.add_argument(
        "--test-snr",
        type=int,
        nargs="+",
        default=[-5, 0, 10, 20, 30, 40],
    )
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument(
        "--base-config",
        default="noisegap_speechcommands",
        help="Hydra base config that selects the dataset and shared training setup.",
    )
    parser.add_argument(
        "--dataset-label",
        default="SpeechCommands-v0.02",
        help="Dataset provenance label written into every generated config.",
    )
    parser.add_argument(
        "--recorded-label",
        default="SpeechCommandsBackgroundWaveform",
        help="Recorded-noise domain label written into configs and summaries.",
    )
    parser.add_argument(
        "--noise-order",
        type=int,
        default=-97,
        help=(
            "Transform order for waveform corruption. Use a value before the "
            "dataset padding transform for variable-length corpora."
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

    spec = WaveformSweepSpec(
        models=tuple(MODELS[name] for name in args.models),
        domains=(
            Domain("G", "GaussianWaveform", "synthetic"),
            Domain(
                "B",
                args.recorded_label,
                "recorded",
                root=recorded_root,
                train_manifest=train_manifest,
                dev_manifest=dev_manifest,
                test_manifest=test_manifest,
            ),
        ),
        seeds=tuple(args.seeds),
        train_snr_levels=tuple(args.train_snr),
        test_snr_levels=tuple(args.test_snr),
        iterations=args.iterations,
    )
    runs = build_waveform_matrix(spec)
    configs_dir = output / "configs"
    results_dir = output / "results"
    records = []
    for run in runs:
        config_name = f"{run.phase.value}_{run.experiment_id}.yaml"
        write_config(
            configs_dir / config_name,
            render_waveform_config(
                run,
                results_dir=results_dir,
                base_config=args.base_config,
                dataset_label=args.dataset_label,
                noise_order=args.noise_order,
            ),
        )
        records.append(
            {
                "phase": run.phase.value,
                "experiment_id": run.experiment_id,
                "model": run.model.label,
                "model_config": run.model.config,
                "base_config": args.base_config,
                "dataset": args.dataset_label,
                "corruption_order": args.noise_order,
                "seed": run.seed,
                "train_domain": run.train_domain.label,
                "test_domain": run.test_domain.label,
                "train_snr_db": run.train_snr_db,
                "test_snr_db": run.test_snr_db,
                "config": str(Path("configs") / config_name),
                "result_dir": str(results_dir / run.experiment_id),
                "checkpoint": str(
                    results_dir / run.training_experiment_id / "_best" / "model.pt"
                ),
                "depends_on": (
                    None
                    if run.phase.value == "train"
                    else str(
                        Path("configs") / f"train_{run.training_experiment_id}.yaml"
                    )
                ),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    train_count = sum(run.phase.value == "train" for run in runs)
    print(
        f"Generated {len(runs)} waveform configs for {args.dataset_label}: "
        f"{train_count} train, {len(runs) - train_count} evaluate."
    )


if __name__ == "__main__":
    main()
