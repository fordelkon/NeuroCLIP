from collections.abc import Mapping
from typing import Any

import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _normalize_metrics(result: Any) -> dict[str, Any]:
    """Convert a KG builder return value into a metric dictionary."""
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return dict(result)
    return {"result": result}


@task_wrapper
def build_kg(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Runs a configured knowledge graph building pipeline.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and instantiated objects.
    """
    log.info(f"Instantiating KG builder <{cfg.build_kg._target_}>")
    kg_builder = hydra.utils.instantiate(cfg.build_kg)

    object_dict = {
        "cfg": cfg,
        "kg_builder": kg_builder,
    }

    log.info("Starting KG building!")
    result = kg_builder.run()

    metric_dict = _normalize_metrics(result)
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="build_kg.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for knowledge graph building.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    extras(cfg)
    build_kg(cfg)


if __name__ == "__main__":
    main()
