"""Experiment matrix planning and config rendering."""

from .matrix import Domain, RunPhase, RunSpec, SweepSpec, build_matrix
from .waveform_matrix import (
    ModelSpec,
    WaveformRunSpec,
    WaveformSweepSpec,
    build_waveform_matrix,
)

__all__ = [
    "Domain",
    "ModelSpec",
    "RunPhase",
    "RunSpec",
    "SweepSpec",
    "WaveformRunSpec",
    "WaveformSweepSpec",
    "build_matrix",
    "build_waveform_matrix",
]
