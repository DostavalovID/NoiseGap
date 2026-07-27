"""Experiment matrix planning and config rendering."""

from .matrix import Domain, RunPhase, RunSpec, SweepSpec, build_matrix

__all__ = ["Domain", "RunPhase", "RunSpec", "SweepSpec", "build_matrix"]
