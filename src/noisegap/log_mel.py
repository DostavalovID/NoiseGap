"""Pure log-Mel power operations.

All public functions in this module use tensors shaped ``[channel, time, mel]``.
"""

import math
from typing import Optional

import torch

EPSILON = 1e-12


def assert_channel_time_mel(features: torch.Tensor) -> None:
    """Validate the canonical NoiseGap feature layout."""
    if features.ndim != 3:
        raise ValueError(
            "Expected log-Mel features shaped [channel, time, mel], "
            f"got {tuple(features.shape)}."
        )


def valid_frame_mask(
    features_db: torch.Tensor,
    *,
    padding_value_db: float = 0.0,
    padding_tolerance_db: float = 0.0,
) -> torch.Tensor:
    """Return ``[channel, time]`` mask excluding a known padding value."""
    assert_channel_time_mel(features_db)
    if not math.isfinite(padding_value_db):
        raise ValueError("padding_value_db must be finite.")
    if not math.isfinite(padding_tolerance_db) or padding_tolerance_db < 0:
        raise ValueError("padding_tolerance_db must be finite and non-negative.")
    difference = (features_db - padding_value_db).abs()
    return difference.amax(dim=-1) > padding_tolerance_db


def db_to_power(features_db: torch.Tensor) -> torch.Tensor:
    """Convert decibels to linear power."""
    return torch.pow(10.0, features_db / 10.0)


def power_to_db(power: torch.Tensor) -> torch.Tensor:
    """Convert non-negative linear power to decibels."""
    return 10.0 * torch.log10(power.clamp_min(EPSILON))


def mix_power_at_snr(
    signal_db: torch.Tensor,
    noise_power: torch.Tensor,
    snr_db: float,
    frames: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mix a positive power field into log-Mel features at aggregate frame SNR.

    Zero-padded frames remain exactly unchanged.
    """
    assert_channel_time_mel(signal_db)
    if signal_db.shape != noise_power.shape:
        raise ValueError(
            "Signal and noise must share [channel, time, mel] shape, "
            f"got {tuple(signal_db.shape)} and {tuple(noise_power.shape)}."
        )
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")
    if torch.any(noise_power < 0):
        raise ValueError("noise_power must be non-negative.")

    frame_mask = valid_frame_mask(signal_db) if frames is None else frames
    if frame_mask.shape != signal_db.shape[:-1]:
        raise ValueError(
            f"Frame mask must have shape {tuple(signal_db.shape[:-1])}, "
            f"got {tuple(frame_mask.shape)}."
        )
    if not torch.any(frame_mask):
        return signal_db.clone()

    mask = frame_mask.unsqueeze(-1).expand_as(signal_db)
    signal_power = db_to_power(signal_db)
    signal_mean = signal_power[mask].mean()
    noise_mean = noise_power[mask].mean()
    if noise_mean <= EPSILON:
        raise ValueError("Noise has zero power on valid frames.")

    target_noise_mean = signal_mean / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise_power * (target_noise_mean / noise_mean)

    mixed_power = signal_power.clone()
    mixed_power[mask] = signal_power[mask] + scaled_noise[mask]
    mixed_db = power_to_db(mixed_power)
    mixed_db[~mask] = signal_db[~mask]
    return mixed_db


def fit_noise_time_axis(
    noise_power: torch.Tensor,
    target_time: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Crop or repeat `[channel, time, mel]` noise along time only."""
    assert_channel_time_mel(noise_power)
    if target_time <= 0:
        raise ValueError("target_time must be positive.")
    noise_time = noise_power.shape[1]
    if noise_time <= 0:
        raise ValueError("Noise must contain at least one time frame.")
    if noise_time == target_time:
        return noise_power
    if noise_time > target_time:
        max_start = noise_time - target_time
        start = int(torch.randint(max_start + 1, (1,), generator=generator).item())
        return noise_power[:, start : start + target_time, :]
    repeats = math.ceil(target_time / noise_time)
    return noise_power.repeat(1, repeats, 1)[:, :target_time, :]
