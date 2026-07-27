"""NoiseGap experiment components."""

from importlib.metadata import version

from .augmentations import RecordedLogMelNoise, SyntheticLogMelNoise

__version__ = version("noisegap")

__all__ = ["RecordedLogMelNoise", "SyntheticLogMelNoise", "__version__"]
