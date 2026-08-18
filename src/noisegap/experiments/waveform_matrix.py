"""Controlled SpeechCommands matrix with shared waveform corruption."""

from dataclasses import dataclass

from .matrix import Domain, RunPhase


@dataclass(frozen=True)
class ModelSpec:
    code: str
    label: str
    config: str
    optimizer: str
    batch_size: int
    learning_rate: float

    def __post_init__(self) -> None:
        if not self.code or not self.code.replace("_", "").isalnum():
            raise ValueError("Model code must contain only letters, numbers, or '_'.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")


@dataclass(frozen=True)
class WaveformSweepSpec:
    models: tuple[ModelSpec, ...]
    domains: tuple[Domain, ...]
    seeds: tuple[int, ...] = (0, 1, 2)
    train_snr_levels: tuple[int, ...] = (20,)
    test_snr_levels: tuple[int, ...] = (-5, 0, 10, 20, 30, 40)
    iterations: int = 15

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("At least one model is required.")
        if len({model.code for model in self.models}) != len(self.models):
            raise ValueError("Model codes must be unique.")
        if len(self.domains) < 2:
            raise ValueError("At least two noise domains are required.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be non-empty and unique.")
        if not self.train_snr_levels or not self.test_snr_levels:
            raise ValueError("Train and test SNR levels must be non-empty.")
        if not set(self.train_snr_levels).issubset(self.test_snr_levels):
            raise ValueError("Every train SNR must also be present in test SNRs.")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")


@dataclass(frozen=True)
class WaveformRunSpec:
    phase: RunPhase
    model: ModelSpec
    seed: int
    train_domain: Domain
    test_domain: Domain
    train_snr_db: int
    test_snr_db: int
    iterations: int

    @property
    def experiment_id(self) -> str:
        pair = f"{self.train_domain.code}{self.test_domain.code}"
        return (
            f"{self.model.code}_s{self.seed}_{pair}_"
            f"train{self.train_snr_db}_test{self.test_snr_db}"
        )

    @property
    def training_experiment_id(self) -> str:
        code = self.train_domain.code
        return (
            f"{self.model.code}_s{self.seed}_{code}{code}_"
            f"train{self.train_snr_db}_test{self.train_snr_db}"
        )


def build_waveform_matrix(spec: WaveformSweepSpec) -> list[WaveformRunSpec]:
    """Build train-first model/seed runs and checkpoint-reuse evaluations."""
    runs = []
    for model in spec.models:
        for seed in spec.seeds:
            for train_domain in spec.domains:
                for train_snr in spec.train_snr_levels:
                    for test_domain in spec.domains:
                        for test_snr in spec.test_snr_levels:
                            diagonal = (
                                train_domain == test_domain
                                and train_snr == test_snr
                            )
                            runs.append(
                                WaveformRunSpec(
                                    phase=(
                                        RunPhase.TRAIN
                                        if diagonal
                                        else RunPhase.EVALUATE
                                    ),
                                    model=model,
                                    seed=seed,
                                    train_domain=train_domain,
                                    test_domain=test_domain,
                                    train_snr_db=train_snr,
                                    test_snr_db=test_snr,
                                    iterations=(spec.iterations if diagonal else 0),
                                )
                            )
    return sorted(runs, key=lambda run: run.phase is RunPhase.EVALUATE)
