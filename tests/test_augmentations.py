import math
import struct
import wave
from pathlib import Path

import pytest
import torch
from autrainer.core.structs import DataItem

from noisegap.augmentations import (
    LegacyArticleRecordedLogMelNoise,
    LegacyArticleSyntheticLogMelNoise,
    RecordedLogMelNoise,
    SyntheticLogMelNoise,
)


def test_synthetic_evaluation_is_stable_per_item() -> None:
    features = torch.full((1, 30, 64), -30.0)
    first = DataItem(features.clone(), 0, 11)
    second = DataItem(features.clone(), 0, 11)
    augmentation = SyntheticLogMelNoise(
        snr_db=5,
        deterministic_per_item=True,
        generator_seed=42,
    )
    augmentation.apply(first)
    augmentation.apply(second)
    assert torch.equal(first.features, second.features)


def test_legacy_article_synthetic_uses_abs_gaussian_power_field() -> None:
    features = torch.full((1, 3, 4), -20.0)
    item = DataItem(features.clone(), 0, 7)
    augmentation = LegacyArticleSyntheticLogMelNoise(
        snr=0,
        noise_type="Gaussian",
        generator_seed=5,
    )

    generator = torch.Generator().manual_seed(5)
    signal_linear = 10 ** (features / 10)
    noise_raw = torch.abs(torch.randn(features.shape, generator=generator))
    noise_linear = noise_raw * (signal_linear.mean() / noise_raw.mean())
    expected = 10 * torch.log10(signal_linear + noise_linear + 1e-9)

    augmentation.apply(item)
    assert torch.equal(item.features, expected)


def test_legacy_article_recorded_preserves_historical_axis_resize(
    tmp_path: Path,
) -> None:
    noise_file = tmp_path / "noise.wav"
    noise_file.touch()
    augmentation = LegacyArticleRecordedLogMelNoise(
        str(tmp_path),
        snr_db=0,
        generator_seed=0,
    )
    augmentation._load_noise_spectrogram = lambda _: torch.full((1, 64, 100), -30.0)
    item = DataItem(torch.full((1, 30, 64), -20.0), 0, 0)

    augmentation.apply(item)

    assert item.features.shape == (1, 30, 64)


def test_legacy_article_recorded_decodes_pcm_without_torchcodec(
    tmp_path: Path,
) -> None:
    noise_file = tmp_path / "noise.wav"
    samples = [
        int(12000 * math.sin(2 * math.pi * 440 * index / 16000))
        for index in range(16000)
    ]
    with wave.open(str(noise_file), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    augmentation = LegacyArticleRecordedLogMelNoise(str(tmp_path), snr_db=0)
    noise_db = augmentation._load_noise_spectrogram(str(noise_file))

    assert noise_db.shape[0] == 1
    assert noise_db.shape[1] == 64
    assert noise_db.shape[2] > 64
    assert torch.isfinite(noise_db).all()


def test_recorded_noise_requires_matching_mel_axis(tmp_path: Path) -> None:
    noise_file = tmp_path / "noise.wav"
    noise_file.touch()
    augmentation = RecordedLogMelNoise(
        str(tmp_path),
        snr_db=5,
        deterministic_per_item=True,
        generator_seed=42,
    )
    augmentation._load_noise_power = lambda _: torch.ones((1, 20, 32))
    item = DataItem(torch.full((1, 30, 64), -30.0), 0, 1)
    with pytest.raises(ValueError, match="mel-bin"):
        augmentation.apply(item)


def test_recorded_noise_preserves_channel_time_mel(tmp_path: Path) -> None:
    noise_file = tmp_path / "noise.wav"
    noise_file.touch()
    augmentation = RecordedLogMelNoise(
        str(tmp_path),
        snr_db=5,
        deterministic_per_item=True,
        generator_seed=42,
    )
    augmentation._load_noise_power = lambda _: torch.ones((1, 20, 64))
    item = DataItem(torch.full((1, 30, 64), -30.0), 0, 1)
    augmentation.apply(item)
    assert item.features.shape == (1, 30, 64)


def test_recorded_noise_decodes_wav_to_channel_time_mel(tmp_path: Path) -> None:
    noise_file = tmp_path / "noise.wav"
    samples = [
        int(12000 * math.sin(2 * math.pi * 440 * index / 16000))
        for index in range(16000)
    ]
    with wave.open(str(noise_file), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    augmentation = RecordedLogMelNoise(str(tmp_path), snr_db=5)
    noise_power = augmentation._load_noise_power(noise_file)

    assert noise_power.ndim == 3
    assert noise_power.shape[0] == 1
    assert noise_power.shape[1] > noise_power.shape[2]
    assert noise_power.shape[2] == 64


def test_recorded_noise_serializes_portably(tmp_path: Path) -> None:
    (tmp_path / "noise.wav").touch()
    augmentation = RecordedLogMelNoise(str(tmp_path), snr_db=5)

    serialized = augmentation.to_yaml_s()

    assert "pathlib" not in serialized
    assert f"noise_root: {tmp_path}" in serialized


def test_recorded_manifest_fails_closed_on_missing_file(tmp_path: Path) -> None:
    manifest = tmp_path / "noise.csv"
    manifest.write_text("path\nmissing.wav\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing.wav"):
        RecordedLogMelNoise(
            str(tmp_path),
            snr_db=5,
            manifest_csv=str(manifest),
        )


def test_recorded_manifest_rejects_duplicate_weighting(tmp_path: Path) -> None:
    (tmp_path / "noise.wav").touch()
    manifest = tmp_path / "noise.csv"
    manifest.write_text("path\nnoise.wav\nnoise.wav\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        RecordedLogMelNoise(
            str(tmp_path),
            snr_db=5,
            manifest_csv=str(manifest),
        )
