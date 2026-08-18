"""Pure waveform operations for pre-frontend noise experiments."""

import math

import torch

EPSILON = 1e-12


def assert_channel_samples(waveform: torch.Tensor) -> None:
    """Validate the canonical waveform layout ``[channel, samples]``."""
    if waveform.ndim != 2:
        raise ValueError(
            "Expected waveform shaped [channel, samples], "
            f"got {tuple(waveform.shape)}."
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

    signal_power = signal.square().mean()
    noise_power = noise.square().mean()
    if signal_power <= EPSILON:
        raise ValueError("Signal has zero waveform power.")
    if noise_power <= EPSILON:
        raise ValueError("Noise has zero waveform power.")

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scaled_noise = noise * torch.sqrt(target_noise_power / noise_power)
    return signal + scaled_noise


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
