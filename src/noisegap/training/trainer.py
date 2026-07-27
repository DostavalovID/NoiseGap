"""Small compatibility boundary around autrainer 0.8.1.

autrainer accepts one augmentation config and applies it only to training.
NoiseGap keeps train/dev/test semantics explicit without copying the trainer.
"""

import shutil
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from autrainer.augmentations import AugmentationManager
from autrainer.training import training as training_module
from autrainer.training.training import ModularTaskTrainer
from autrainer.training.utils import format_results
from omegaconf import DictConfig

_CONSTRUCTION_LOCK = Lock()


def extract_test_tracking_value(
    test_results: dict[str, float],
    tracking_metric_name: str,
) -> float:
    """Read an autrainer test metric without dropping its phase prefix."""
    key = f"test_{tracking_metric_name}"
    if key not in test_results:
        raise KeyError(
            f"Tracking metric '{key}' is absent from test results: "
            f"{sorted(test_results)}"
        )
    return float(test_results[key])


class PhaseAwareAugmentationManager(AugmentationManager):
    """Translate a phase bundle into autrainer's existing manager."""

    def __init__(self, bundle: Optional[DictConfig | dict] = None) -> None:
        if bundle is None:
            super().__init__()
            return
        if not all(key in bundle for key in ("train", "dev", "test")):
            super().__init__(train_augmentation=bundle)
            return
        super().__init__(
            train_augmentation=bundle.get("train"),
            dev_augmentation=bundle.get("dev"),
            test_augmentation=bundle.get("test"),
        )


class NoiseGapTrainer(ModularTaskTrainer):
    """Use explicit phase augmentations and a real checkpoint-evaluation path."""

    def __init__(self, cfg: DictConfig, output_directory: str, **kwargs: Any) -> None:
        checkpoint = cfg.model.get("model_checkpoint")
        if cfg.iterations == 0:
            if not checkpoint:
                raise ValueError("Evaluation-only runs require model.model_checkpoint.")
            if not Path(checkpoint).is_file():
                raise FileNotFoundError(
                    f"Evaluation checkpoint does not exist: {checkpoint}"
                )
        with _CONSTRUCTION_LOCK:
            original = training_module.AugmentationManager
            training_module.AugmentationManager = PhaseAwareAugmentationManager
            try:
                super().__init__(cfg=cfg, output_directory=output_directory, **kwargs)
            finally:
                training_module.AugmentationManager = original

    def train(self) -> float:
        if self.cfg.iterations > 0:
            return super().train()
        return self.evaluate_checkpoint()

    def evaluate_checkpoint(self) -> float:
        """Evaluate the configured checkpoint without inventing training metrics."""
        self._thread_manager.join()
        self._remove_redundant_evaluation_states()
        self.callback_manager.callback(position="cb_on_train_begin", trainer=self)
        self.model.to(self.DEVICE)
        self.model.eval()
        self.bookkeeping.create_folder("_test")
        self.test_timer.start()
        test_results = self.evaluate(
            -1,
            "_test",
            self.test_loader,
            self.data.df_test,
            dev_evaluation=False,
            save_to="test_holistic",
            tracker=self.test_tracker,
        )
        self.test_timer.stop()
        self.callback_manager.callback(
            position="cb_on_test_end",
            trainer=self,
            test_results=test_results,
        )
        self.bookkeeping.log(
            format_results(test_results, "Test", self.cfg.training_type)
        )
        self.test_timer.save()
        self.callback_manager.callback(position="cb_on_train_end", trainer=self)
        return extract_test_tracking_value(
            test_results,
            self.data.tracking_metric.name,
        )

    def _remove_redundant_evaluation_states(self) -> None:
        """Remove states created by upstream construction but unused by eval."""
        for directory_name in ("_initial", "_best"):
            directory = self.output_directory / directory_name
            if directory.is_dir():
                shutil.rmtree(directory)
