import math
from pathlib import Path

import audiofile
import numpy as np
import pytest
import torch
from autrainer.core.structs import DataItem

from noisegap.augmentations import RecordedWaveformNoise, WaveformGaussianNoise
from noisegap.waveform import (
    fit_noise_sample_axis,
    fit_nonzero_noise_sample_axis,
    fit_nonzero_noise_sample_axis_with_metadata,
    mix_waveform_at_snr,
)


def _snr_db(signal: torch.Tensor, mixed: torch.Tensor) -> float:
    noise = mixed - signal
    return float(10 * torch.log10(signal.square().mean() / noise.square().mean()))


def test_waveform_mix_hits_requested_snr_without_clipping() -> None:
    signal = torch.linspace(-0.95, 0.95, 16000).unsqueeze(0)
    noise = torch.randn(signal.shape, generator=torch.Generator().manual_seed(7))

    mixed = mix_waveform_at_snr(signal, noise, snr_db=-5)

    assert _snr_db(signal, mixed) == pytest.approx(-5, abs=1e-4)
    assert mixed.abs().max() > 1.0


def test_waveform_gaussian_is_deterministic_per_item() -> None:
    signal = torch.ones((1, 4000))
    augmentation = WaveformGaussianNoise(
        snr_db=10,
        deterministic_per_item=True,
        generator_seed=11,
    )

    first = augmentation(DataItem(signal.clone(), 0, 3))
    second = augmentation(DataItem(signal.clone(), 0, 3))

    assert torch.equal(first.features, second.features)
    assert _snr_db(signal, first.features) == pytest.approx(10, abs=1e-4)


def test_deterministic_eval_noise_is_invariant_to_worker_seed_offset() -> None:
    signal = torch.ones((1, 4000))
    augmentation = WaveformGaussianNoise(
        snr_db=10,
        deterministic_per_item=True,
        generator_seed=11,
    )
    expected = augmentation(DataItem(signal.clone(), 0, 3)).features

    augmentation.offset_generator_seed(7)
    actual = augmentation(DataItem(signal.clone(), 0, 3)).features

    assert torch.equal(actual, expected)


def test_recorded_waveform_resamples_and_hits_snr(tmp_path: Path) -> None:
    noise_path = tmp_path / "noise.wav"
    t = np.arange(8000, dtype=np.float32) / 8000
    audiofile.write(noise_path, np.sin(2 * math.pi * 440 * t), 8000)
    augmentation = RecordedWaveformNoise(
        noise_root=str(tmp_path),
        snr_db=5,
        sample_rate=16000,
        deterministic_per_item=True,
        generator_seed=2,
    )
    signal = torch.full((1, 16000), 0.2)

    item = augmentation(DataItem(signal.clone(), 0, 9))

    assert item.features.shape == signal.shape
    assert _snr_db(signal, item.features) == pytest.approx(5, abs=1e-3)


def test_noise_repeats_only_along_sample_axis() -> None:
    noise = torch.tensor([[1.0, 2.0, 3.0]])

    fitted = fit_noise_sample_axis(noise, 8, torch.Generator().manual_seed(0))

    assert fitted.tolist() == [[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]]


def test_waveform_mix_rejects_silence() -> None:
    with pytest.raises(ValueError, match="Signal has zero"):
        mix_waveform_at_snr(torch.zeros((1, 10)), torch.ones((1, 10)), 0)


def test_recorded_noise_crop_retries_silent_regions_deterministically() -> None:
    noise = torch.cat((torch.zeros((1, 100)), torch.ones((1, 20))), dim=-1)

    first = fit_nonzero_noise_sample_axis(
        noise,
        10,
        torch.Generator().manual_seed(0),
        max_attempts=64,
    )
    second = fit_nonzero_noise_sample_axis(
        noise,
        10,
        torch.Generator().manual_seed(0),
        max_attempts=64,
    )

    assert first.square().mean() > 0
    assert torch.equal(first, second)


def test_recorded_noise_crop_fails_closed_if_every_crop_is_silent() -> None:
    with pytest.raises(ValueError, match="zero power"):
        fit_nonzero_noise_sample_axis(
            torch.zeros((1, 100)),
            10,
            torch.Generator().manual_seed(0),
            max_attempts=4,
        )


def test_recorded_noise_crop_rejects_near_silent_regions() -> None:
    noise = torch.cat(
        (
            torch.full((1, 100), 1e-5),
            torch.ones((1, 100)),
        ),
        dim=-1,
    )

    crop = fit_nonzero_noise_sample_axis(
        noise,
        20,
        torch.Generator().manual_seed(0),
        max_attempts=128,
        min_crop_rms_ratio=0.1,
    )

    assert float(crop.square().mean()) >= float(noise.square().mean()) * 0.01


def test_recorded_noise_crop_reports_reproducible_metadata() -> None:
    noise = torch.arange(20, dtype=torch.float32).reshape(1, -1)
    generator = torch.Generator().manual_seed(3)

    fitted, metadata = fit_nonzero_noise_sample_axis_with_metadata(
        noise,
        5,
        generator,
        min_crop_rms_ratio=0.1,
    )

    expected = noise[:, metadata.start_sample : metadata.start_sample + 5]
    assert torch.equal(fitted, expected)
    assert metadata.attempts == 1
    assert metadata.repeated is False
    assert metadata.crop_rms_ratio > 0.1
