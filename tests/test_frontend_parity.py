from pathlib import Path

import torch
import yaml
from autrainer import instantiate_shorthand
from autrainer.core.structs import DataItem
from autrainer.transforms import SmartCompose, TransformManager

from noisegap.transforms import DETERMINISTIC_FRONTEND_CACHE_KEY, CachedPannMel


def test_cached_matched_frontend_equals_online_waveform_frontend() -> None:
    repository = Path(__file__).resolve().parents[1]
    waveform_dataset = yaml.safe_load(
        (
            repository
            / "conf/dataset/TIMIT-sentencetype-article-waveform-16k.yaml"
        ).read_text(encoding="utf-8")
    )["transform"]
    waveform_model = yaml.safe_load(
        (repository / "conf/model/Cnn10-32k-T-waveform.yaml").read_text(
            encoding="utf-8"
        )
    )["transform"]
    preprocessing = yaml.safe_load(
        (
            repository / "conf/preprocessing/pann_32k_timit_padded_779.yaml"
        ).read_text(encoding="utf-8")
    )["pipeline"]
    online, _, _ = TransformManager(
        waveform_model,
        waveform_dataset,
    ).get_transforms()
    cached = SmartCompose(
        [instantiate_shorthand(transform) for transform in preprocessing]
    )
    waveform = torch.randn(
        (1, 16000),
        generator=torch.Generator().manual_seed(7),
    )

    online_features = online(DataItem(waveform.clone(), 0, 0)).features
    cached_features = cached(DataItem(waveform.clone(), 0, 0)).features

    assert online_features.shape == (1, 779, 64)
    assert torch.equal(cached_features, online_features)


def test_pann_cache_reuses_only_explicitly_keyed_frontend_result() -> None:
    kwargs = {
        "sample_rate": 32000,
        "window_size": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "fmin": 50,
        "fmax": 14000,
        "ref": 1.0,
        "amin": 1.0e-10,
        "top_db": None,
    }
    transform = CachedPannMel(**kwargs, cache_size=2)
    waveform = torch.randn((1, 32000), generator=torch.Generator().manual_seed(7))
    first = DataItem(waveform.clone(), 0, 4)
    setattr(first, DETERMINISTIC_FRONTEND_CACHE_KEY, ("test", 4))
    expected = transform(first).features.clone()

    keyed_again = DataItem(torch.zeros_like(waveform), 0, 4)
    setattr(keyed_again, DETERMINISTIC_FRONTEND_CACHE_KEY, ("test", 4))
    actual_cached = transform(keyed_again).features
    unkeyed = transform(DataItem(torch.zeros_like(waveform), 0, 4)).features

    assert torch.equal(actual_cached, expected)
    assert not torch.equal(unkeyed, expected)
    assert not hasattr(keyed_again, DETERMINISTIC_FRONTEND_CACHE_KEY)
