"""NoiseGap experiment components."""

from importlib.metadata import version

from .augmentations import (
    RecordedLogMelNoise,
    RecordedWaveformNoise,
    SyntheticLogMelNoise,
    WaveformGaussianNoise,
)

__version__ = version("noisegap")

__all__ = [
    "RecordedLogMelNoise",
    "RecordedWaveformNoise",
    "SyntheticLogMelNoise",
    "WaveformGaussianNoise",
    "__version__",
]
