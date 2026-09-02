from itertools import pairwise

import pytest

from noisegap.seeding import (
    TRAIN_AUGMENTATION_SEED_STRIDE,
    training_augmentation_seed,
)


def test_training_augmentation_seeds_reserve_worker_ranges() -> None:
    seeds = [training_augmentation_seed(seed) for seed in range(8)]

    assert len(seeds) == len(set(seeds))
    assert all(
        later - earlier == TRAIN_AUGMENTATION_SEED_STRIDE
        for earlier, later in pairwise(seeds)
    )
    assert seeds[0] + 99999 < seeds[1]


def test_training_augmentation_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        training_augmentation_seed(-1)
