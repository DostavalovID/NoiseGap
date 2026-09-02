from pathlib import Path

import pytest
import torch

from noisegap.representation_snr import (
    calibrate_representation_snr,
    component_logmel_snr_db,
)


def test_component_logmel_snr_uses_clean_active_frames() -> None:
    clean = torch.zeros((1, 4, 2))
    noise = torch.full((1, 4, 2), -10.0)
    clean[:, 3, :] = -100.0
    noise[:, 3, :] = 0.0

    measured = component_logmel_snr_db(clean, noise)

    assert measured == 10.0


def test_calibration_rejects_unknown_dataset_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset_split"):
        calibrate_representation_snr(
            tmp_path,
            tmp_path,
            tmp_path / "noise.csv",
            tmp_path / "preprocessing.yaml",
            dataset_split="validation",
            snr_levels=(0,),
            samples=1,
        )
