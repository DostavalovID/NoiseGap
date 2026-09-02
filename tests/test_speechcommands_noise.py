import wave
from pathlib import Path

from noisegap.datasets.speechcommands import prepare_background_noise_manifests


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 160)


def test_speechcommands_real_noise_excludes_white_and_pink(tmp_path: Path) -> None:
    noise = tmp_path / "SpeechCommands" / "default" / "_background_noise_"
    noise.mkdir(parents=True)
    for filename in (
        "doing_the_dishes.wav",
        "dude_miaowing.wav",
        "exercise_bike.wav",
        "running_tap.wav",
        "pink_noise.wav",
        "white_noise.wav",
    ):
        _write_wav(noise / filename)

    output = tmp_path / "manifests"
    summary = prepare_background_noise_manifests(
        tmp_path / "SpeechCommands",
        output,
    )

    combined = "".join(
        (output / f"{split}.csv").read_text(encoding="utf-8")
        for split in ("train", "dev", "test")
    )
    assert "white_noise.wav" not in combined
    assert "pink_noise.wav" not in combined
    assert summary["excluded_files"] == ["pink_noise.wav", "white_noise.wav"]
    assert summary["train_files"] == 2
    assert summary["dev_files"] == 1
    assert summary["test_files"] == 1
