"""Model adapters that make cross-domain initialization explicit."""

from typing import Optional

from autrainer.core.utils import set_seed
from autrainer.models import Cnn10


class SeededCnn10(Cnn10):
    """CNN10 whose initialization is independent of dataset construction.

    autrainer normally seeds once before it instantiates augmentations and the
    dataset. Different corruption implementations can consume different global
    RNG streams before the model is created. Resetting here makes the complete
    initial model and the subsequent loader RNG state identical for a given
    experimental seed.
    """

    def __init__(
        self,
        output_dim: int,
        initialization_seed: int,
        segmentwise: bool = False,
        in_channels: int = 1,
        transfer: Optional[str] = None,
    ) -> None:
        if initialization_seed < 0:
            raise ValueError("initialization_seed must be non-negative.")
        self.initialization_seed = initialization_seed
        set_seed(initialization_seed)
        super().__init__(
            output_dim=output_dim,
            segmentwise=segmentwise,
            in_channels=in_channels,
            transfer=transfer,
        )
