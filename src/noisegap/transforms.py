"""Exact frontend adapters used by the experiment pipelines."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from typing import TYPE_CHECKING

from autrainer.transforms import PannMel

if TYPE_CHECKING:
    import torch
    from autrainer.core.structs import AbstractDataItem


DETERMINISTIC_FRONTEND_CACHE_KEY = "_noisegap_deterministic_frontend_cache_key"


class CachedPannMel(PannMel):
    """Cache PANN features only when an upstream transform supplies a safe key.

    Training augmentations deliberately do not attach a key, so their random
    corruption is recomputed on every epoch. Deterministic development and test
    augmentations attach a key that fully describes their waveform realization.
    Persistent DataLoader workers can then reuse the exact first PANN result.
    """

    def __init__(
        self,
        window_size: int,
        hop_size: int,
        sample_rate: int,
        fmin: int,
        fmax: int,
        mel_bins: int,
        ref: float,
        amin: float,
        top_db: int | None,
        cache_size: int = 1024,
        order: int = -90,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")
        super().__init__(
            window_size=window_size,
            hop_size=hop_size,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
            mel_bins=mel_bins,
            ref=ref,
            amin=amin,
            top_db=top_db,
            order=order,
        )
        self.cache_size = cache_size
        self._noisegap_cache: OrderedDict[Hashable, torch.Tensor] = OrderedDict()

    def __call__(self, item: AbstractDataItem) -> AbstractDataItem:
        key = getattr(item, DETERMINISTIC_FRONTEND_CACHE_KEY, None)
        if key is not None and not isinstance(key, Hashable):
            raise TypeError("Deterministic frontend cache keys must be hashable.")
        if key is not None and self.cache_size > 0:
            cached = self._noisegap_cache.get(key)
            if cached is not None:
                self._noisegap_cache.move_to_end(key)
                item.features = cached.clone()
                return item

        item = super().__call__(item)
        if key is not None and self.cache_size > 0:
            self._noisegap_cache[key] = item.features.detach().clone()
            self._noisegap_cache.move_to_end(key)
            while len(self._noisegap_cache) > self.cache_size:
                self._noisegap_cache.popitem(last=False)
        return item
