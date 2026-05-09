"""Shared CLIP model naming helpers."""

CLIP_MODEL_MAP = {
    "ViT-B-16": "laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
    "ViT-B-32": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "ViT-L-14": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
    "ViT-H-14": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "ViT-g-14": "laion/CLIP-ViT-g-14-laion2B-s34B-b88K",
    "ViT-bigG-14": "laion/CLIP-ViT-bigG-14-laion2B-s39B-b160K",
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
