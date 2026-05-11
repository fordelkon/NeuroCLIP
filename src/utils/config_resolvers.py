"""Shared OmegaConf resolvers and CLIP model naming helpers."""

from omegaconf import OmegaConf

CLIP_MODEL_MAP = {
    "ViT-B-16": "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "ViT-B-32": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "ViT-L-14": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "ViT-g-14": "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    "ViT-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-s39B-b160K",
}

CLIP_DIM_MAP = {
    "ViT-B-16": 512,
    "ViT-B-32": 512,
    "ViT-L-14": 768,
    "ViT-H-14": 1024,
    "ViT-g-14": 1024,
    "ViT-bigG-14": 1280,
}

CLIP_HIDDEN_SIZE_MAP = {
    "ViT-B-16": 768,
    "ViT-B-32": 768,
    "ViT-L-14": 1024,
    "ViT-H-14": 1280,
    "ViT-g-14": 1408,
    "ViT-bigG-14": 1664,
}


def sanitize_clip_model_name(value: str) -> str:
    """Return the filesystem-safe model name used by CLIP feature caches."""
    return value.replace("/", "-").replace("\\", "-").replace(" ", "_")


def resolve_clip_model_id(model_name: str, model_id: str | None = None) -> tuple[str, str]:
    """Resolve a simple CLIP model name to a Hugging Face model id."""
    if model_id:
        return model_name, model_id
    if model_name in CLIP_MODEL_MAP:
        return model_name, CLIP_MODEL_MAP[model_name]
    return sanitize_clip_model_name(model_name), model_name


def resolve_clip_dim(model_name: str) -> int:
    """Return the pooled CLIP embedding dimension for a supported model name."""
    if model_name not in CLIP_DIM_MAP:
        raise ValueError(f"Unknown CLIP model: {model_name}. Available: {list(CLIP_DIM_MAP)}")
    return CLIP_DIM_MAP[model_name]


def resolve_clip_hidden_size(model_name: str) -> int:
    """Return the CLIP hidden size for a supported model name."""
    if model_name not in CLIP_HIDDEN_SIZE_MAP:
        raise ValueError(
            f"Unknown CLIP model: {model_name}. Available: {list(CLIP_HIDDEN_SIZE_MAP)}"
        )
    return CLIP_HIDDEN_SIZE_MAP[model_name]


def is_cross_subject(experiment_setting: str) -> bool:
    """Return whether a THINGS-EEG2 experiment setting is cross-subject."""
    return experiment_setting == "cross-subject"


def register_config_resolvers() -> None:
    """Register OmegaConf resolvers used by CLIP-aligned configs."""
    if not OmegaConf.has_resolver("clip_dim"):
        OmegaConf.register_new_resolver("clip_dim", resolve_clip_dim)
    if not OmegaConf.has_resolver("clip_hidden_size"):
        OmegaConf.register_new_resolver("clip_hidden_size", resolve_clip_hidden_size)
    if not OmegaConf.has_resolver("len"):
        OmegaConf.register_new_resolver("len", len)
    if not OmegaConf.has_resolver("is_cross_subject"):
        OmegaConf.register_new_resolver("is_cross_subject", is_cross_subject)
