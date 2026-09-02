"""Deterministic seed namespaces for independent experiment runs."""

TRAIN_AUGMENTATION_SEED_BASE = 10_000_000
TRAIN_AUGMENTATION_SEED_STRIDE = 100_000
MAX_TORCH_SEED = 2**63 - 1


def training_augmentation_seed(training_seed: int) -> int:
    """Reserve a worker-offset range for one training-seed replication."""
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative.")
    seed = (
        TRAIN_AUGMENTATION_SEED_BASE
        + training_seed * TRAIN_AUGMENTATION_SEED_STRIDE
    )
    if seed > MAX_TORCH_SEED:
        raise ValueError("training_seed is too large for torch.Generator.")
    return seed
