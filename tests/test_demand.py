import hashlib
import io
import wave
import zipfile
from pathlib import Path

from noisegap.datasets.demand import prepare_demand_noise


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160)
    return output.getvalue()


def _archive(root: Path, environment: str) -> str:
    path = root / f"{environment}_16k.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(f"{environment}/ch01.wav", _wav_bytes())
        bundle.writestr(f"{environment}/ch02.wav", _wav_bytes())
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


def test_prepare_demand_uses_one_environment_per_split(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    archives = {
        environment: _archive(raw, environment)
        for environment in ("DKITCHEN", "DLIVING", "DWASHING")
    }
    output = tmp_path / "prepared"

    manifest = prepare_demand_noise(
        raw,
        output,
        archives=archives,
        splits={
            "train": ("DKITCHEN",),
            "dev": ("DLIVING",),
            "test": ("DWASHING",),
        },
    )

    assert manifest["counts"] == {"train": 1, "dev": 1, "test": 1}
    assert manifest["channel_policy"] == "ch01_only"
    assert (output / "default" / "DKITCHEN" / "ch01.wav").is_file()
    assert not (output / "default" / "DKITCHEN" / "ch02.wav").exists()
