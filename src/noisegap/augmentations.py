"""autrainer adapters for feature-space noise experiments."""

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import audiofile
import torch
from autrainer.augmentations import AbstractAugmentation
from autrainer.core.structs import AbstractDataItem, DataItem
from autrainer.transforms import PannMel
from torchaudio import functional as audio_functional

from .log_mel import EPSILON, db_to_power, fit_noise_time_axis, mix_power_at_snr
from .waveform import fit_noise_sample_axis, mix_waveform_at_snr


class _SeededNoise(AbstractAugmentation):
    def __init__(
        self,
        *,
        deterministic_per_item: bool,
        order: int,
        p: float,
        generator_seed: Optional[int],
    ) -> None:
        super().__init__(order=order, p=p, generator_seed=generator_seed)
        self.deterministic_per_item = deterministic_per_item
        self._noise_generator = torch.Generator()
        if generator_seed is not None:
            self._noise_generator.manual_seed(generator_seed)

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        if self.generator_seed is not None:
            self._noise_generator.manual_seed(self.generator_seed)

    def _generator_for(self, item: AbstractDataItem) -> torch.Generator:
        if not self.deterministic_per_item:
            return self._noise_generator
        if self.generator_seed is None:
            raise ValueError("deterministic_per_item=True requires generator_seed.")
        generator = torch.Generator()
        generator.manual_seed(self.generator_seed + int(item.index))
        return generator


class SyntheticLogMelNoise(_SeededNoise):
    """Add a squared-Gaussian random power field in log-Mel space.

    This is feature-space corruption, not waveform-level Gaussian noise.
    """

    def __init__(
        self,
        snr_db: float,
        deterministic_per_item: bool = False,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            deterministic_per_item=deterministic_per_item,
            order=order,
            p=p,
            generator_seed=generator_seed,
        )
        self.snr_db = snr_db

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        generator = self._generator_for(item)
        noise_power = torch.randn(
            item.features.shape,
            generator=generator,
            dtype=item.features.dtype,
            device="cpu",
        ).square()
        noise_power = noise_power.to(item.features.device)
        item.features = mix_power_at_snr(
            item.features,
            noise_power,
            self.snr_db,
        )
        return item


class WaveformGaussianNoise(_SeededNoise):
    """Add Gaussian noise to raw ``[channel, samples]`` audio before frontend."""

    def __init__(
        self,
        snr_db: float,
        deterministic_per_item: bool = False,
        order: int = -97,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            deterministic_per_item=deterministic_per_item,
            order=order,
            p=p,
            generator_seed=generator_seed,
        )
        self.snr_db = snr_db

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        generator = self._generator_for(item)
        noise = torch.randn(
            item.features.shape,
            generator=generator,
            dtype=item.features.dtype,
            device="cpu",
        ).to(item.features.device)
        item.features = mix_waveform_at_snr(
            item.features,
            noise,
            self.snr_db,
        )
        return item


class RecordedLogMelNoise(_SeededNoise):
    """Mix waveform-derived recorded noise into `[channel, time, mel]` features."""

    def __init__(
        self,
        noise_root: str,
        snr_db: float,
        sample_rate: int = 16000,
        window_size: int = 512,
        hop_size: int = 160,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 8000,
        ref: float = 1.0,
        amin: float = 1e-10,
        top_db: Optional[int] = None,
        manifest_csv: Optional[str] = None,
        deterministic_per_item: bool = False,
        cache_size: int = 16,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            deterministic_per_item=deterministic_per_item,
            order=order,
            p=p,
            generator_seed=generator_seed,
        )
        self.noise_root = noise_root
        self._noise_root = Path(noise_root)
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.manifest_csv = manifest_csv
        self.cache_size = cache_size
        self.noise_files = self._discover_noise_files()
        self._cache: OrderedDict[Path, torch.Tensor] = OrderedDict()
        self._pann_mel = PannMel(
            window_size=window_size,
            hop_size=hop_size,
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
            mel_bins=mel_bins,
            ref=ref,
            amin=amin,
            top_db=top_db,
        )

    def _discover_noise_files(self) -> list[Path]:
        if self.manifest_csv is None:
            files = sorted(self._noise_root.rglob("*.wav"))
        else:
            manifest = Path(self.manifest_csv)
            with manifest.open(newline="", encoding="utf-8") as stream:
                rows = csv.DictReader(stream)
                if rows.fieldnames is None or "path" not in rows.fieldnames:
                    raise ValueError("Noise manifest must contain a 'path' column.")
                relative_paths = [Path(row["path"]) for row in rows]
            invalid = [
                path
                for path in relative_paths
                if path.is_absolute() or ".." in path.parts
            ]
            if invalid:
                raise ValueError(
                    "Noise manifest paths must stay below noise_root; "
                    f"invalid path: {invalid[0]}"
                )
            files = [self._noise_root / path for path in relative_paths]
            if len(files) != len(set(files)):
                raise ValueError("Noise manifest contains duplicate paths.")
            missing = [path for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Noise manifest references {len(missing)} missing file(s); "
                    f"first: {missing[0]}"
                )
        if not files:
            raise ValueError(f"No WAV noise files found under {self.noise_root}.")
        return files

    def _load_noise_power(self, path: Path) -> torch.Tensor:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached.clone()

        waveform, sample_rate = audiofile.read(str(path), always_2d=True)
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            tensor = audio_functional.resample(
                tensor,
                sample_rate,
                self.sample_rate,
            )
        if tensor.square().mean() <= EPSILON:
            raise ValueError(f"Recorded noise file has zero power: {path}")

        # Use the exact PANN feature extractor configured for the speech signal.
        noise_db = self._pann_mel(DataItem(features=tensor, target=0, index=0)).features
        power = db_to_power(noise_db)
        if power.shape[-1] != self.mel_bins:
            raise ValueError(
                f"Expected {self.mel_bins} mel bins, got {power.shape[-1]}."
            )

        if self.cache_size > 0:
            self._cache[path] = power
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return power.clone()

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        generator = self._generator_for(item)
        file_index = int(
            torch.randint(len(self.noise_files), (1,), generator=generator).item()
        )
        noise_power = self._load_noise_power(self.noise_files[file_index])
        if noise_power.shape[0] != item.features.shape[0]:
            if noise_power.shape[0] == 1:
                noise_power = noise_power.expand(item.features.shape[0], -1, -1)
            else:
                raise ValueError(
                    "Recorded noise channel count does not match signal: "
                    f"{noise_power.shape[0]} vs {item.features.shape[0]}."
                )
        if noise_power.shape[-1] != item.features.shape[-1]:
            raise ValueError(
                "Recorded noise and signal use different mel-bin counts: "
                f"{noise_power.shape[-1]} vs {item.features.shape[-1]}."
            )
        noise_power = fit_noise_time_axis(
            noise_power,
            item.features.shape[1],
            generator,
        ).to(device=item.features.device, dtype=item.features.dtype)
        item.features = mix_power_at_snr(
            item.features,
            noise_power,
            self.snr_db,
        )
        return item


class RecordedWaveformNoise(_SeededNoise):
    """Mix recorded noise with raw audio before either model frontend."""

    def __init__(
        self,
        noise_root: str,
        snr_db: float,
        sample_rate: int = 16000,
        manifest_csv: Optional[str] = None,
        deterministic_per_item: bool = False,
        cache_size: int = 16,
        order: int = -97,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            deterministic_per_item=deterministic_per_item,
            order=order,
            p=p,
            generator_seed=generator_seed,
        )
        self.noise_root = noise_root
        self._noise_root = Path(noise_root)
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.manifest_csv = manifest_csv
        self.cache_size = cache_size
        self.noise_files = self._discover_noise_files()
        self._cache: OrderedDict[Path, torch.Tensor] = OrderedDict()

    def _discover_noise_files(self) -> list[Path]:
        if self.manifest_csv is None:
            files = sorted(self._noise_root.rglob("*.wav"))
        else:
            manifest = Path(self.manifest_csv)
            with manifest.open(newline="", encoding="utf-8") as stream:
                rows = csv.DictReader(stream)
                if rows.fieldnames is None or "path" not in rows.fieldnames:
                    raise ValueError("Noise manifest must contain a 'path' column.")
                relative_paths = [Path(row["path"]) for row in rows]
            invalid = [
                path
                for path in relative_paths
                if path.is_absolute() or ".." in path.parts
            ]
            if invalid:
                raise ValueError(
                    "Noise manifest paths must stay below noise_root; "
                    f"invalid path: {invalid[0]}"
                )
            files = [self._noise_root / path for path in relative_paths]
            if len(files) != len(set(files)):
                raise ValueError("Noise manifest contains duplicate paths.")
            missing = [path for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Noise manifest references {len(missing)} missing file(s); "
                    f"first: {missing[0]}"
                )
        if not files:
            raise ValueError(f"No WAV noise files found under {self.noise_root}.")
        return files

    def _load_waveform(self, path: Path) -> torch.Tensor:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached.clone()

        waveform, sample_rate = audiofile.read(str(path), always_2d=True)
        tensor = torch.as_tensor(waveform, dtype=torch.float32)
        if tensor.shape[0] > 1:
            tensor = tensor.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            tensor = audio_functional.resample(tensor, sample_rate, self.sample_rate)
        if tensor.square().mean() <= EPSILON:
            raise ValueError(f"Recorded noise file has zero power: {path}")

        if self.cache_size > 0:
            self._cache[path] = tensor
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return tensor.clone()

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        generator = self._generator_for(item)
        file_index = int(
            torch.randint(len(self.noise_files), (1,), generator=generator).item()
        )
        noise = self._load_waveform(self.noise_files[file_index])
        if noise.shape[0] != item.features.shape[0]:
            if noise.shape[0] == 1:
                noise = noise.expand(item.features.shape[0], -1)
            else:
                raise ValueError(
                    "Recorded noise channel count does not match signal: "
                    f"{noise.shape[0]} vs {item.features.shape[0]}."
                )
        noise = fit_noise_sample_axis(
            noise,
            item.features.shape[-1],
            generator,
        ).to(device=item.features.device, dtype=item.features.dtype)
        item.features = mix_waveform_at_snr(item.features, noise, self.snr_db)
        return item
