"""Measure waveform and PANN-component SNR under the exact test policy."""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import audiofile
import torch
from autrainer.core.structs import DataItem

from noisegap.features import load_preprocessing
from noisegap.log_mel import db_to_power, valid_frame_mask
from noisegap.waveform import (
    fit_nonzero_noise_sample_axis_with_metadata,
    scale_waveform_noise_at_snr,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_logmel_snr_db(
    clean_db: torch.Tensor,
    noise_component_db: torch.Tensor,
) -> float:
    """Ratio of separately transformed signal/noise powers on speech frames."""
    frames = valid_frame_mask(
        clean_db,
        padding_value_db=-100.0,
        padding_tolerance_db=0.01,
    )
    mask = frames.unsqueeze(-1).expand_as(clean_db)
    signal_power = db_to_power(clean_db)[mask].mean()
    noise_power = db_to_power(noise_component_db)[mask].mean()
    return float(10.0 * torch.log10(signal_power / noise_power))


def _load_waveform(path: Path) -> torch.Tensor:
    waveform, sample_rate = audiofile.read(str(path), always_2d=True)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz input, got {sample_rate}: {path}")
    tensor = torch.as_tensor(waveform, dtype=torch.float32)
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    return tensor


def _manifest_paths(root: Path, manifest: Path) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None or "path" not in rows.fieldnames:
            raise ValueError("Recorded-noise manifest must contain path.")
        relative = [Path(row["path"]) for row in rows]
    if not relative:
        raise ValueError("Recorded-noise manifest must not be empty.")
    if any(path.is_absolute() or ".." in path.parts for path in relative):
        raise ValueError("Recorded-noise manifest path escapes its root.")
    paths = [root / path for path in relative]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing recorded noise: {missing[0]}")
    return paths


def calibrate_representation_snr(
    dataset_root: Path,
    recorded_root: Path,
    recorded_manifest: Path,
    preprocessing_config: Path,
    *,
    snr_levels: tuple[int, ...],
    samples: int,
    sample_seed: int = 0,
    evaluation_seed: int = 2_000_000,
    min_crop_rms_ratio: float = 0.1,
) -> list[dict[str, Any]]:
    """Return per-utterance physical and frontend-component SNR rows."""
    dataset_root = dataset_root.resolve()
    recorded_root = recorded_root.resolve()
    recorded_manifest = recorded_manifest.resolve()
    preprocessing_config = preprocessing_config.resolve()
    with (dataset_root / "test.csv").open(newline="", encoding="utf-8") as stream:
        test_rows = list(csv.DictReader(stream))
    if samples < 1 or samples > len(test_rows):
        raise ValueError(f"samples must be in 1..{len(test_rows)}")
    indexed_rows = list(enumerate(test_rows))
    random.Random(sample_seed).shuffle(indexed_rows)
    selected = indexed_rows[:samples]
    noise_paths = _manifest_paths(recorded_root, recorded_manifest)
    _, frontend = load_preprocessing(preprocessing_config)
    noise_cache: dict[Path, torch.Tensor] = {}
    output = []

    for dataset_index, row in selected:
        source_path = dataset_root / "default" / row["path"]
        signal = _load_waveform(source_path)
        clean_db = frontend(DataItem(signal.clone(), 0, dataset_index)).features
        for domain in ("GaussianWaveform", "RecordedNoiseWaveform"):
            generator = torch.Generator().manual_seed(evaluation_seed + dataset_index)
            selected_noise_path = None
            crop_start = 0
            crop_attempts = 1
            crop_ratio = 1.0
            if domain == "GaussianWaveform":
                raw_noise = torch.randn(
                    signal.shape,
                    generator=generator,
                    dtype=signal.dtype,
                )
            else:
                file_index = int(
                    torch.randint(len(noise_paths), (1,), generator=generator).item()
                )
                selected_noise_path = noise_paths[file_index]
                raw_source = noise_cache.get(selected_noise_path)
                if raw_source is None:
                    raw_source = _load_waveform(selected_noise_path)
                    noise_cache[selected_noise_path] = raw_source
                raw_noise, crop = fit_nonzero_noise_sample_axis_with_metadata(
                    raw_source,
                    signal.shape[-1],
                    generator,
                    min_crop_rms_ratio=min_crop_rms_ratio,
                )
                crop_start = crop.start_sample
                crop_attempts = crop.attempts
                crop_ratio = crop.crop_rms_ratio

            for snr_db in snr_levels:
                scaled_noise, gain = scale_waveform_noise_at_snr(
                    signal,
                    raw_noise,
                    float(snr_db),
                )
                physical_snr = float(
                    10.0
                    * torch.log10(signal.square().mean() / scaled_noise.square().mean())
                )
                noise_db = frontend(
                    DataItem(scaled_noise.clone(), 0, dataset_index)
                ).features
                output.append(
                    {
                        "dataset_index": dataset_index,
                        "source_path": row["path"],
                        "noise_domain": domain,
                        "noise_path": (
                            str(selected_noise_path) if selected_noise_path else ""
                        ),
                        "nominal_waveform_snr_db": snr_db,
                        "measured_waveform_snr_db": physical_snr,
                        "waveform_snr_error_db": physical_snr - snr_db,
                        "pann_component_snr_db": component_logmel_snr_db(
                            clean_db, noise_db
                        ),
                        "noise_gain": gain,
                        "crop_start_sample": crop_start,
                        "crop_attempts": crop_attempts,
                        "crop_rms_ratio": crop_ratio,
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recorded-root", type=Path, required=True)
    parser.add_argument("--recorded-manifest", type=Path, required=True)
    parser.add_argument("--preprocessing-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr", type=int, nargs="+", default=[-5, 0, 10, 20, 30, 40])
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--evaluation-seed", type=int, default=2_000_000)
    parser.add_argument("--min-crop-rms-ratio", type=float, default=0.1)
    args = parser.parse_args()
    rows = calibrate_representation_snr(
        args.dataset_root,
        args.recorded_root,
        args.recorded_manifest,
        args.preprocessing_config,
        snr_levels=tuple(args.snr),
        samples=args.samples,
        sample_seed=args.sample_seed,
        evaluation_seed=args.evaluation_seed,
        min_crop_rms_ratio=args.min_crop_rms_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "definition": (
            "PANN component SNR is mean clean-component Mel power divided by "
            "mean separately transformed added-noise Mel power on clean speech frames."
        ),
        "cross_term_policy": "components transformed separately",
        "test_split_sha256": _sha256(args.dataset_root / "test.csv"),
        "recorded_manifest_sha256": _sha256(args.recorded_manifest),
        "preprocessing_config_sha256": _sha256(args.preprocessing_config),
        "samples": args.samples,
        "sample_seed": args.sample_seed,
        "evaluation_seed": args.evaluation_seed,
        "snr_levels": args.snr,
        "min_crop_rms_ratio": args.min_crop_rms_ratio,
        "rows": len(rows),
        "output_sha256": _sha256(args.output),
    }
    provenance_path = args.output.with_suffix(args.output.suffix + ".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    maximum_error = max(abs(row["waveform_snr_error_db"]) for row in rows)
    if maximum_error > 1e-4 or not math.isfinite(maximum_error):
        raise ValueError(f"Waveform SNR calibration failed: max error {maximum_error}")
    print(f"Wrote {len(rows)} rows; maximum waveform SNR error={maximum_error:.2e} dB")


if __name__ == "__main__":
    main()
