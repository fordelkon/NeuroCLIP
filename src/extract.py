from collections.abc import Mapping
from typing import Any

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _normalize_metrics(result: Any) -> dict[str, Any]:
    """Convert an extractor return value into a metric dictionary."""
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": result}


@task_wrapper
def extract(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs a configured feature extraction pipeline.

    The configured object must expose a ``run()`` method. This keeps the entry point
    dataset-agnostic while allowing extractor-specific classes to own heavy logic.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and instantiated objects.
    """
    log.info(f"Instantiating extractor <{cfg.extract._target_}>")
    extractor = hydra.utils.instantiate(cfg.extract)

    object_dict = {
        "cfg": cfg,
        "extractor": extractor,
    }

    log.info("Starting feature extraction!")
    result = extractor.run()

    metric_dict = _normalize_metrics(result)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="extract.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for feature extraction.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    extract(cfg)


if __name__ == "__main__":
    main()
