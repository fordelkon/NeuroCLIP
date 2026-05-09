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


def test_preprocess_config(cfg_preprocess: DictConfig) -> None:
    """Tests the preprocessing configuration provided by the `cfg_preprocess` fixture.

    :param cfg_preprocess: A DictConfig containing a valid preprocessing configuration.
    """
    assert cfg_preprocess
    assert cfg_preprocess.preprocess
    assert cfg_preprocess.preprocess._target_.startswith("src.preprocessors.")

    HydraConfig().set_config(cfg_preprocess)

    preprocessor = hydra.utils.instantiate(cfg_preprocess.preprocess)

    assert hasattr(preprocessor, "run")


def test_extract_config(cfg_extract: DictConfig) -> None:
    """Tests the extraction configuration provided by the `cfg_extract` fixture.

    :param cfg_extract: A DictConfig containing a valid extraction configuration.
    """
    assert cfg_extract
    assert cfg_extract.extract
    assert cfg_extract.extract._target_.startswith("src.extractors.")

    HydraConfig().set_config(cfg_extract)

    extractor = hydra.utils.instantiate(cfg_extract.extract)

    assert hasattr(extractor, "run")


def test_preprocess_uses_global_thingseeg2_paths(cfg_preprocess: DictConfig) -> None:
    """Tests that THINGS-EEG2 preprocessing paths are centralized in `paths`.

    :param cfg_preprocess: A DictConfig containing a valid preprocessing configuration.
    """
    assert cfg_preprocess.paths.thingseeg2_raw_dir
    assert cfg_preprocess.paths.thingseeg2_img_dir
    assert cfg_preprocess.paths.thingseeg2_preprocessed_dir
    assert cfg_preprocess.preprocess.raw_data_dir == cfg_preprocess.paths.thingseeg2_raw_dir
    assert cfg_preprocess.preprocess.img_data_dir == cfg_preprocess.paths.thingseeg2_img_dir
    assert cfg_preprocess.preprocess.save_dir == cfg_preprocess.paths.thingseeg2_preprocessed_dir


def test_extract_uses_global_thingseeg2_paths(cfg_extract: DictConfig) -> None:
    """Tests that THINGS-EEG2 extraction paths are centralized in `paths`.

    :param cfg_extract: A DictConfig containing a valid extraction configuration.
    """
    assert cfg_extract.paths.thingseeg2_preprocessed_dir
    assert cfg_extract.paths.thingseeg2_img_dir
    assert cfg_extract.paths.thingseeg2_clip_features_dir
    assert cfg_extract.extract.prep_eeg_data_dir == cfg_extract.paths.thingseeg2_preprocessed_dir
    assert cfg_extract.extract.save_dir == cfg_extract.paths.thingseeg2_clip_features_dir
    assert cfg_extract.extract.model_cache_dir == cfg_extract.paths.clip_model_cache_dir


def test_rich_progress_bar_uses_ascii_time_columns(cfg_train: DictConfig) -> None:
    """Tests that progress rendering avoids Unicode-only separators on Windows terminals.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    callbacks = instantiate_callbacks(cfg_train.callbacks)
    progress_bar = next(
        callback for callback in callbacks if isinstance(callback, RichProgressBar)
    )

    column_names = {
        type(column).__name__ for column in progress_bar.configure_columns(trainer=None)
    }

    assert "CustomTimeColumn" not in column_names
