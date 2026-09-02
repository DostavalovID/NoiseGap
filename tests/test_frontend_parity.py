from pathlib import Path

import torch
import yaml
from autrainer import instantiate_shorthand
from autrainer.core.structs import DataItem
from autrainer.transforms import SmartCompose, TransformManager


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
