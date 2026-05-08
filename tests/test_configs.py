import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from omegaconf import DictConfig

from src.utils.instantiators import instantiate_callbacks


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_rich_progress_bar_uses_ascii_time_columns(cfg_train: DictConfig) -> None:
    """Tests that progress rendering avoids Unicode-only separators on Windows terminals.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    callbacks = instantiate_callbacks(cfg_train.callbacks)
    progress_bar = next(callback for callback in callbacks if isinstance(callback, RichProgressBar))

    column_names = {type(column).__name__ for column in progress_bar.configure_columns(trainer=None)}

    assert "CustomTimeColumn" not in column_names
