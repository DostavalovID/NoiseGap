"""Pure waveform operations for pre-frontend noise experiments."""

import math
from dataclasses import dataclass

import torch

EPSILON = 1e-12


@dataclass(frozen=True)
class NoiseCropInfo:
    """Reproducibility metadata for one recorded-noise crop."""

    start_sample: int
    attempts: int
    repeated: bool
    source_rms: float
    crop_rms: float

    @property
    def crop_rms_ratio(self) -> float:
        return self.crop_rms / self.source_rms


def assert_channel_samples(waveform: torch.Tensor) -> None:
    """Validate the canonical waveform layout ``[channel, samples]``."""
    if waveform.ndim != 2:
        raise ValueError(
            f"Expected waveform shaped [channel, samples], got {tuple(waveform.shape)}."
        )


def mix_waveform_at_snr(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Add waveform noise at a mean-square SNR without clipping."""
    assert_channel_samples(signal)
    assert_channel_samples(noise)
    if signal.shape != noise.shape:
        raise ValueError(
            "Signal and noise must share [channel, samples] shape, "
            f"got {tuple(signal.shape)} and {tuple(noise.shape)}."
        )
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    scaled_noise, _ = scale_waveform_noise_at_snr(signal, noise, snr_db)
    return signal + scaled_noise


def scale_waveform_noise_at_snr(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> tuple[torch.Tensor, float]:
    """Scale noise to a mean-square waveform SNR and return its gain."""
    assert_channel_samples(signal)
    assert_channel_samples(noise)
    if signal.shape != noise.shape:
        raise ValueError(
            "Signal and noise must share [channel, samples] shape, "
            f"got {tuple(signal.shape)} and {tuple(noise.shape)}."
        )
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")
    signal_power = signal.square().mean()
    noise_power = noise.square().mean()
    if signal_power <= EPSILON:
        raise ValueError("Signal has zero waveform power.")
    if noise_power <= EPSILON:
        raise ValueError("Noise has zero waveform power.")
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    gain = torch.sqrt(target_noise_power / noise_power)
    return noise * gain, float(gain)


def fit_noise_sample_axis(
    noise: torch.Tensor,
    target_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Crop or repeat ``[channel, samples]`` noise along samples only."""
    assert_channel_samples(noise)
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")
    noise_samples = noise.shape[-1]
    if noise_samples <= 0:
        raise ValueError("Noise must contain at least one sample.")
    if noise_samples == target_samples:
        return noise
    if noise_samples > target_samples:
        max_start = noise_samples - target_samples
        start = int(torch.randint(max_start + 1, (1,), generator=generator).item())
        return noise[:, start : start + target_samples]
    repeats = math.ceil(target_samples / noise_samples)
    return noise.repeat(1, repeats)[:, :target_samples]


def fit_nonzero_noise_sample_axis(
    noise: torch.Tensor,
    target_samples: int,
    generator: torch.Generator,
    *,
    max_attempts: int = 32,
    min_crop_rms_ratio: float = 0.0,
) -> torch.Tensor:
    """Fit recorded noise while rejecting silent or inactive crops.

    The source file is kept fixed so files retain uniform sampling weight. Only
    the crop offset is resampled, using the caller's generator so evaluation
    remains deterministic per item.
    """
    fitted, _ = fit_nonzero_noise_sample_axis_with_metadata(
        noise,
        target_samples,
        generator,
        max_attempts=max_attempts,
        min_crop_rms_ratio=min_crop_rms_ratio,
    )
    return fitted


def fit_nonzero_noise_sample_axis_with_metadata(
    noise: torch.Tensor,
    target_samples: int,
    generator: torch.Generator,
    *,
    max_attempts: int = 32,
    min_crop_rms_ratio: float = 0.0,
) -> tuple[torch.Tensor, NoiseCropInfo]:
    """Fit an active crop and expose its offset and relative RMS."""
    assert_channel_samples(noise)
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if not math.isfinite(min_crop_rms_ratio) or not 0 <= min_crop_rms_ratio <= 1:
        raise ValueError("min_crop_rms_ratio must be finite and in [0, 1].")
    noise_samples = noise.shape[-1]
    if noise_samples <= 0:
        raise ValueError("Noise must contain at least one sample.")
    source_power = noise.square().mean()
    if source_power <= EPSILON:
        raise ValueError("Recorded noise source has zero power.")
    minimum_crop_power = max(EPSILON, float(source_power) * min_crop_rms_ratio**2)
    for attempt in range(1, max_attempts + 1):
        if noise_samples > target_samples:
            max_start = noise_samples - target_samples
            start = int(torch.randint(max_start + 1, (1,), generator=generator).item())
            fitted = noise[:, start : start + target_samples]
            repeated = False
        elif noise_samples < target_samples:
            start = 0
            repeats = math.ceil(target_samples / noise_samples)
            fitted = noise.repeat(1, repeats)[:, :target_samples]
            repeated = True
        else:
            start = 0
            fitted = noise
            repeated = False
        crop_power = fitted.square().mean()
        if crop_power >= minimum_crop_power:
            source_rms = math.sqrt(float(source_power))
            crop_rms = math.sqrt(float(crop_power))
            return fitted, NoiseCropInfo(
                start_sample=start,
                attempts=attempt,
                repeated=repeated,
                source_rms=source_rms,
                crop_rms=crop_rms,
            )
    raise ValueError(
        "Could not sample an active recorded-noise crop after "
        f"{max_attempts} attempts (minimum RMS ratio "
        f"{min_crop_rms_ratio})."
    )
