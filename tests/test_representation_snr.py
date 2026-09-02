import torch

from noisegap.representation_snr import component_logmel_snr_db


def test_component_logmel_snr_uses_clean_active_frames() -> None:
    clean = torch.zeros((1, 4, 2))
    noise = torch.full((1, 4, 2), -10.0)
    clean[:, 3, :] = -100.0
    noise[:, 3, :] = 0.0

    measured = component_logmel_snr_db(clean, noise)

    assert measured == 10.0
