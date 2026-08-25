"""autrainer adapters for feature-space noise experiments."""

import csv
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import audiofile
import torch
from autrainer.augmentations import AbstractAugmentation
from autrainer.core.structs import AbstractDataItem, DataItem
from autrainer.transforms import PannMel
from torchaudio import functional as audio_functional
from torchaudio import transforms as audio_transforms

from .log_mel import EPSILON, db_to_power, fit_noise_time_axis, mix_power_at_snr
from .waveform import fit_nonzero_noise_sample_axis, mix_waveform_at_snr


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
        self._deterministic_seed = generator_seed
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
        if self._deterministic_seed is None:
            raise ValueError("deterministic_per_item=True requires generator_seed.")
        generator = torch.Generator()
        generator.manual_seed(self._deterministic_seed + int(item.index))
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


class LegacyArticleSyntheticLogMelNoise(AbstractAugmentation):
    """Exact Gaussian feature-noise implementation used by the article run.

    This compatibility class intentionally preserves the historical
    ``abs(randn)`` power-field construction.  It must not be used as the
    corrected feature-space implementation.
    """

    _AVAILABLE_NOISE_TYPES = {"Gaussian", "StaticGaussian"}

    def __init__(
        self,
        snr: float = 0.0,
        noise_type: str = "Gaussian",
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        if noise_type not in self._AVAILABLE_NOISE_TYPES:
            raise ValueError(f"Unsupported legacy noise type: {noise_type}")
        super().__init__(order=order, p=p, generator_seed=generator_seed)
        self.snr = snr
        self.noise_type = noise_type
        self._generator = torch.Generator()
        if generator_seed is not None:
            self._generator.manual_seed(generator_seed)

    def offset_generator_seed(self, offset: int) -> None:
        super().offset_generator_seed(offset)
        if self.generator_seed is not None:
            self._generator.manual_seed(self.generator_seed)

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        mask_signal = item.features.abs().sum(dim=-1) > 0
        signal_linear = 10 ** (item.features / 10)
        p_signal = signal_linear[mask_signal].mean()
        p_noise_target = p_signal / (10 ** (self.snr / 10))

        if self.noise_type == "Gaussian":
            generator = self._generator
        else:
            if self.generator_seed is None:
                raise ValueError("StaticGaussian requires generator_seed.")
            generator = torch.Generator()
            generator.manual_seed(self.generator_seed + int(item.index))
        noise_raw = torch.abs(
            torch.randn(
                item.features.size(),
                generator=generator,
                dtype=item.features.dtype,
                device="cpu",
            )
        ).to(item.features.device)
        noise_linear = noise_raw * (
            p_noise_target / (noise_raw.mean() + 1e-9)
        )

        mixed_linear = signal_linear.clone()
        mixed_linear[mask_signal] = (
            mixed_linear[mask_signal] + noise_linear[mask_signal]
        )
        item.features = 10 * torch.log10(mixed_linear + 1e-9)
        return item


class LegacyArticleRecordedLogMelNoise(AbstractAugmentation):
    """Exact AudioSet feature-noise path used by the article run.

    The historical implementation treated the final axis as time before
    resizing a ``[channel, mel, time]`` tensor.  That behavior is intentionally
    retained here solely to make the published experiment reproducible.
    """

    def __init__(
        self,
        noise_dir: str,
        snr_db: float,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 64,
        noise_csv: Optional[str] = None,
        noise_type: Optional[str] = None,
        order: int = 0,
        p: float = 1.0,
        generator_seed: Optional[int] = None,
    ) -> None:
        super().__init__(order=order, p=p, generator_seed=generator_seed)
        self.noise_dir = noise_dir
        self.snr_db = snr_db
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.noise_csv = noise_csv
        self.noise_type = noise_type
        self.noise_files = self._load_noise_files(noise_csv)
        if not self.noise_files:
            raise ValueError(f"No noise files found in {noise_dir}")
        self.mel_transform = audio_transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self._noise_cache: dict[str, torch.Tensor] = {}

    def _load_noise_files(self, noise_csv: Optional[str]) -> list[str]:
        noise_files: list[str] = []
        if noise_csv is not None and os.path.exists(noise_csv):
            with open(noise_csv, newline="", encoding="utf-8") as stream:
                rows = csv.DictReader(stream)
                for row in rows:
                    if (
                        self.noise_type is not None
                        and row.get("label") != self.noise_type
                    ):
                        continue
                    full_path = os.path.join(self.noise_dir, row["path"])
                    if os.path.exists(full_path):
                        noise_files.append(full_path)
        else:
            for root, _, files in os.walk(self.noise_dir):
                for filename in files:
                    if filename.endswith(".wav"):
                        noise_files.append(os.path.join(root, filename))
        return noise_files

    def _load_noise_spectrogram(self, noise_path: str) -> torch.Tensor:
        if noise_path in self._noise_cache:
            return self._noise_cache[noise_path].clone()
        samples, sample_rate = audiofile.read(noise_path, always_2d=True)
        waveform = torch.as_tensor(samples, dtype=torch.float32)
        if sample_rate != self.sample_rate:
            waveform = audio_transforms.Resample(
                sample_rate,
                self.sample_rate,
            )(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = 10 * torch.log10(mel_spec + 1e-9)
        self._noise_cache[noise_path] = log_mel_spec
        return log_mel_spec.clone()

    @staticmethod
    def _match_length(noise: torch.Tensor, target_length: int) -> torch.Tensor:
        noise_length = noise.shape[2]
        if noise_length == target_length:
            return noise
        if noise_length > target_length:
            start = random.randint(0, noise_length - target_length)
            return noise[:, :, start : start + target_length]
        repeat_times = (target_length // noise_length) + 1
        return noise.repeat(1, 1, repeat_times)[:, :, :target_length]

    def apply(self, item: AbstractDataItem) -> AbstractDataItem:
        mask_signal = item.features.abs().sum(dim=-1) > 0
        noise_path = random.choice(self.noise_files)
        noise_spec = self._load_noise_spectrogram(noise_path)

        # Historical axis behavior: item.features.shape[-1] is the mel axis.
        noise_spec = self._match_length(noise_spec, item.features.shape[-1])
        if noise_spec.shape[1] != item.features.shape[1]:
            noise_spec = torch.nn.functional.interpolate(
                noise_spec.unsqueeze(0),
                size=(item.features.shape[1], item.features.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        signal_linear = 10 ** (item.features / 10)
        noise_linear = 10 ** (noise_spec / 10)
        p_signal = signal_linear[mask_signal].mean()
        p_noise_current = noise_linear[mask_signal].mean()
        p_noise_target = p_signal / (10 ** (self.snr_db / 10))
        noise_scale = p_noise_target / (p_noise_current + 1e-9)
        scaled_noise_linear = noise_linear * noise_scale

        mixed_linear = signal_linear.clone()
        mixed_linear[mask_signal] = (
            mixed_linear[mask_signal] + scaled_noise_linear[mask_signal]
        )
        item.features = 10 * torch.log10(mixed_linear + 1e-9)
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
        max_crop_attempts: int = 32,
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
        if max_crop_attempts <= 0:
            raise ValueError("max_crop_attempts must be positive.")
        self.max_crop_attempts = max_crop_attempts
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
        noise = fit_nonzero_noise_sample_axis(
            noise,
            item.features.shape[-1],
            generator,
            max_attempts=self.max_crop_attempts,
        ).to(device=item.features.device, dtype=item.features.dtype)
        item.features = mix_waveform_at_snr(item.features, noise, self.snr_db)
        return item
