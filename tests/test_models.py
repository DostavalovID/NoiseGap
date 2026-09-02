from pathlib import Path

import torch
import yaml

from noisegap.models import SeededCnn10


def _state(model: SeededCnn10) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }


def test_seeded_cnn10_ignores_prior_global_rng_consumption() -> None:
    torch.manual_seed(11)
    torch.randn(37)
    first = _state(SeededCnn10(output_dim=3, initialization_seed=7))

    torch.manual_seed(29)
    torch.randn(91)
    second = _state(SeededCnn10(output_dim=3, initialization_seed=7))

    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_seeded_cnn10_changes_initialization_across_seeds() -> None:
    first = SeededCnn10(output_dim=3, initialization_seed=7)
    second = SeededCnn10(output_dim=3, initialization_seed=8)

    assert not torch.equal(first.out.weight, second.out.weight)


def test_cnn10_experiment_configs_use_seeded_initialization() -> None:
    root = Path(__file__).parents[1]
    for name in ("Cnn10-32k-T-seeded.yaml", "Cnn10-32k-T-waveform.yaml"):
        config = yaml.safe_load((root / "conf" / "model" / name).read_text())
        assert config["_target_"] == "noisegap.models.SeededCnn10"
        assert config["initialization_seed"] == "${seed}"

    feature = yaml.safe_load(
        (root / "conf" / "noisegap_article_timit_feature_matched.yaml").read_text()
    )
    assert {"override model": "Cnn10-32k-T-seeded"} in feature["defaults"]
