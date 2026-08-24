"""Declarative experiment matrix."""

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


def validate_recorded_manifest_split(
    root: Path,
    train_manifest: Path,
    dev_manifest: Path,
    test_manifest: Path,
) -> dict[str, int]:
    """Fail closed on missing, unsafe, duplicated, or cross-split noise paths."""
    root = root.resolve()
    split_paths: dict[str, set[Path]] = {}
    for split, manifest in (
        ("train", train_manifest),
        ("dev", dev_manifest),
        ("test", test_manifest),
    ):
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            if rows.fieldnames is None or "path" not in rows.fieldnames:
                raise ValueError(
                    f"Noise {split} manifest must contain a 'path' column."
                )
            relative_paths = [Path(row["path"]) for row in rows]
        if not relative_paths:
            raise ValueError(f"Noise {split} manifest must not be empty.")
        unsafe = [
            path for path in relative_paths if path.is_absolute() or ".." in path.parts
        ]
        if unsafe:
            raise ValueError(f"Noise {split} manifest path escapes root: {unsafe[0]}")
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError(f"Noise {split} manifest contains duplicate paths.")
        missing = [path for path in relative_paths if not (root / path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Noise {split} manifest references a missing file: {missing[0]}"
            )
        split_paths[split] = set(relative_paths)

    for first, second in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = split_paths[first] & split_paths[second]
        if overlap:
            raise ValueError(
                f"Noise leakage: {first}/{second} manifests overlap at "
                f"{sorted(overlap)[0]}."
            )
    return {split: len(paths) for split, paths in split_paths.items()}


class RunPhase(str, Enum):
    TRAIN = "train"
    EVALUATE = "evaluate"


@dataclass(frozen=True)
class Domain:
    code: str
    label: str
    kind: str
    root: Path | None = None
    train_manifest: Path | None = None
    dev_manifest: Path | None = None
    test_manifest: Path | None = None

    def __post_init__(self) -> None:
        if len(self.code) != 1 or not self.code.isalpha():
            raise ValueError("Domain code must be one alphabetic character.")
        if self.kind not in {"synthetic", "recorded"}:
            raise ValueError("Domain kind must be 'synthetic' or 'recorded'.")
        if self.kind == "recorded" and (
            self.root is None
            or self.train_manifest is None
            or self.dev_manifest is None
            or self.test_manifest is None
        ):
            raise ValueError(
                "Recorded domains require root and train/dev/test manifests."
            )


@dataclass(frozen=True)
class SweepSpec:
    domains: tuple[Domain, ...]
    snr_levels: tuple[int, ...] = (-5, 0, 10, 20, 30, 40)
    iterations: int = 15

    def __post_init__(self) -> None:
        if len(self.domains) < 2:
            raise ValueError("At least two domains are required.")
        if len({domain.code for domain in self.domains}) != len(self.domains):
            raise ValueError("Domain codes must be unique.")
        if not self.snr_levels:
            raise ValueError("At least one SNR level is required.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")


@dataclass(frozen=True)
class RunSpec:
    phase: RunPhase
    train_domain: Domain
    test_domain: Domain
    train_snr_db: int
    test_snr_db: int
    iterations: int

    @property
    def experiment_id(self) -> str:
        pair = f"{self.train_domain.code}{self.test_domain.code}"
        return f"{pair}_train{self.train_snr_db}_test{self.test_snr_db}"

    @property
    def training_experiment_id(self) -> str:
        code = self.train_domain.code
        return f"{code}{code}_train{self.train_snr_db}_test{self.train_snr_db}"


def build_matrix(spec: SweepSpec) -> list[RunSpec]:
    """Build a train-first matrix followed by checkpoint-reuse evaluations."""
    runs = []
    for train_domain in spec.domains:
        for test_domain in spec.domains:
            for train_snr in spec.snr_levels:
                for test_snr in spec.snr_levels:
                    diagonal = train_domain == test_domain and train_snr == test_snr
                    runs.append(
                        RunSpec(
                            phase=(RunPhase.TRAIN if diagonal else RunPhase.EVALUATE),
                            train_domain=train_domain,
                            test_domain=test_domain,
                            train_snr_db=train_snr,
                            test_snr_db=test_snr,
                            iterations=spec.iterations if diagonal else 0,
                        )
                    )
    return sorted(runs, key=lambda run: run.phase is RunPhase.EVALUATE)
