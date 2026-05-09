from typing import Any

import numpy as np
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.preprocess import preprocess
from src.preprocessors import thingseeg2
from src.preprocessors.thingseeg2 import mvnn, zscore


class DummyPreprocessor:
    """Small configured preprocessor used to exercise the generic entry point."""

    def __init__(self, value: int) -> None:
        self.value = value

    def run(self) -> dict[str, Any]:
        return {"value": self.value}


def test_preprocess_run(cfg_preprocess: DictConfig) -> None:
    """Run the configured preprocessing entry point.

    :param cfg_preprocess: A DictConfig containing a valid preprocessing configuration.
    """
    HydraConfig().set_config(cfg_preprocess)
    with open_dict(cfg_preprocess):
        cfg_preprocess.preprocess = {
            "_target_": "tests.test_preprocess.DummyPreprocessor",
            "value": 7,
        }

    metric_dict, object_dict = preprocess(cfg_preprocess)

    assert metric_dict == {"value": 7}
    assert isinstance(object_dict["preprocessor"], DummyPreprocessor)


def test_mvnn_releases_input_sessions() -> None:
    """Release consumed epoched sessions after whitening to reduce peak memory."""
    epoched_test = [
        np.arange(36, dtype=np.float32).reshape(2, 3, 2, 3),
    ]
    epoched_train = [
        (np.arange(36, dtype=np.float32).reshape(2, 3, 2, 3) + 1.0),
    ]

    whitened_test, whitened_train = mvnn(
        num_sessions=1,
        mvnn_dim="time",
        epoched_test=epoched_test,
        epoched_train=epoched_train,
    )

    assert epoched_test[0] is None
    assert epoched_train[0] is None
    assert whitened_test[0].shape == (2, 3, 2, 3)
    assert whitened_train[0].shape == (2, 3, 2, 3)


def test_zscore_uses_training_statistics_per_session_and_channel() -> None:
    """Normalize train and test sessions with training-only channel statistics."""
    epoched_test = [
        np.array(
            [
                [
                    [[10.0, 20.0, 30.0], [100.0, 200.0, 300.0]],
                    [[40.0, 50.0, 60.0], [400.0, 500.0, 600.0]],
                ]
            ],
            dtype=np.float32,
        )
    ]
    epoched_train = [
        np.array(
            [
                [
                    [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                    [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
                ]
            ],
            dtype=np.float32,
        )
    ]

    normalized_test, normalized_train = zscore(
        epoched_test=epoched_test,
        epoched_train=epoched_train,
    )

    train_mean = epoched_train[0].mean(axis=(0, 1, 3), keepdims=True)
    train_std = epoched_train[0].std(axis=(0, 1, 3), keepdims=True)
    np.testing.assert_allclose(
        normalized_train[0],
        (epoched_train[0] - train_mean) / train_std,
    )
    np.testing.assert_allclose(
        normalized_test[0],
        (epoched_test[0] - train_mean) / train_std,
    )
    assert normalized_test[0].dtype == np.float32
    assert normalized_train[0].dtype == np.float32


def test_without_mvnn_applies_zscore_before_saving(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply z-score to both partitions before saving when MVNN is disabled."""
    epoched_test = [
        np.array(
            [
                [
                    [[10.0, 20.0], [100.0, 200.0]],
                    [[30.0, 40.0], [300.0, 400.0]],
                ]
            ],
            dtype=np.float32,
        )
    ]
    epoched_train = [
        np.array(
            [
                [
                    [[1.0, 2.0], [10.0, 20.0]],
                    [[3.0, 4.0], [30.0, 40.0]],
                ]
            ],
            dtype=np.float32,
        )
    ]
    img_conditions_train = [np.array([1])]
    times = np.array([0.0, 0.004])
    saved: dict[str, Any] = {}

    def fake_epoching(
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[str], np.ndarray]:
        if kwargs["data_part"] == "test":
            return [epoched_test[0].copy()], [], ["Cz", "Pz"], times
        return [epoched_train[0].copy()], img_conditions_train, ["Cz", "Pz"], times

    def fake_save_prepr(**kwargs: Any) -> dict[str, str]:
        saved.update(kwargs)
        return {"test": "test.pt", "training": "training.pt"}

    monkeypatch.setattr(thingseeg2, "epoching", fake_epoching)
    monkeypatch.setattr(thingseeg2, "save_prepr", fake_save_prepr)

    preprocessor = thingseeg2.Thingseeg2Preprocessor(
        subject_id=1,
        num_sessions=1,
        sfreq=1000,
        tmin=-0.2,
        tmax=1.0,
        mvnn_dim=None,
        raw_data_dir="raw",
        img_data_dir="images",
        save_dir="save",
        seed=0,
    )

    outputs = preprocessor._run_partitioned_without_mvnn()

    train_mean = epoched_train[0].mean(axis=(0, 1, 3), keepdims=True)
    train_std = epoched_train[0].std(axis=(0, 1, 3), keepdims=True)
    np.testing.assert_allclose(
        saved["whitened_test"][0],
        (epoched_test[0] - train_mean) / train_std,
    )
    np.testing.assert_allclose(
        saved["whitened_train"][0],
        (epoched_train[0] - train_mean) / train_std,
    )
    assert saved["img_conditions_train"] == img_conditions_train
    assert outputs == {"test": "test.pt", "training": "training.pt"}


def test_target_time_indices_selects_post_stimulus_window() -> None:
    """Select exactly one second from stimulus onset instead of taking the tail."""
    times = np.arange(-0.2, 1.004, 0.004)

    indices = thingseeg2._target_time_indices(times=times, dsfreq=250)

    selected_times = times[indices]
    assert selected_times.shape == (250,)
    assert selected_times[0] == pytest.approx(0.0)
    assert selected_times[-1] == pytest.approx(0.996)


def test_validate_metadata_lengths_rejects_misaligned_images() -> None:
    """Reject image metadata that cannot align one-to-one with EEG condition rows."""
    with pytest.raises(ValueError, match="training metadata"):
        thingseeg2._validate_metadata_lengths(
            partition="training",
            expected_rows=2,
            imgs=["image_a.jpg"],
            labels=[0],
            texts=["image a"],
        )
