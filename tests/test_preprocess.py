from typing import Any

import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.preprocess import preprocess
from src.preprocessors.thingseeg2 import mvnn


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
