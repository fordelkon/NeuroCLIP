from collections.abc import Mapping
from typing import Any

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _normalize_metrics(result: Any) -> dict[str, Any]:
    """Convert a preprocessor return value into a metric dictionary."""
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": result}


@task_wrapper
def preprocess(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs a configured preprocessing pipeline.

    The configured object must expose a ``run()`` method. This keeps the entry point
    dataset-agnostic while allowing dataset-specific preprocessors to own heavy logic.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and instantiated objects.
    """
    log.info(f"Instantiating preprocessor <{cfg.preprocess._target_}>")
    preprocessor = hydra.utils.instantiate(cfg.preprocess)

    object_dict = {
        "cfg": cfg,
        "preprocessor": preprocessor,
    }

    log.info("Starting preprocessing!")
    result = preprocessor.run()

    metric_dict = _normalize_metrics(result)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for preprocessing.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    preprocess(cfg)


if __name__ == "__main__":
    main()
