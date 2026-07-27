import pytest
import torch

from noisegap.log_mel import fit_noise_time_axis, mix_power_at_snr


def _measured_snr(
    original_db: torch.Tensor,
    mixed_db: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    signal = torch.pow(10.0, original_db[valid] / 10.0)
    mixed = torch.pow(10.0, mixed_db[valid] / 10.0)
    return float(10.0 * torch.log10(signal.mean() / (mixed - signal).mean()))


def test_mix_hits_snr_and_preserves_padding() -> None:
    signal = torch.full((1, 100, 64), -40.0)
    signal[:, 80:, :] = 0.0
    noise = torch.rand(signal.shape, generator=torch.Generator().manual_seed(7))
    mixed = mix_power_at_snr(signal, noise, 10.0)
    valid = torch.zeros_like(signal, dtype=torch.bool)
    valid[:, :80, :] = True

    assert _measured_snr(signal, mixed, valid) == pytest.approx(10.0, abs=1e-4)
    assert torch.equal(mixed[:, 80:, :], signal[:, 80:, :])


def test_mix_rejects_axis_mismatch() -> None:
    signal = torch.zeros((1, 100, 64))
    noise = torch.zeros((1, 64, 100))
    with pytest.raises(ValueError, match="share"):
        mix_power_at_snr(signal, noise, 0.0)


def test_fit_noise_changes_time_only() -> None:
    noise = torch.ones((1, 20, 64))
    fitted = fit_noise_time_axis(
        noise,
        45,
        torch.Generator().manual_seed(3),
    )
    assert fitted.shape == (1, 45, 64)
