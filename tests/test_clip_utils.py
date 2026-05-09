from src.utils.clip import resolve_clip_model_id, sanitize_clip_model_name


def test_resolve_clip_model_id_maps_known_short_name() -> None:
    assert resolve_clip_model_id("ViT-B-32") == (
        "ViT-B-32",
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    )


def test_resolve_clip_model_id_keeps_explicit_model_id() -> None:
    assert resolve_clip_model_id("custom", "org/model-a") == ("custom", "org/model-a")


def test_resolve_clip_model_id_uses_direct_model_name_as_fallback() -> None:
    assert resolve_clip_model_id("org/model-a") == ("org-model-a", "org/model-a")


def test_sanitize_clip_model_name_matches_feature_directory_names() -> None:
    assert sanitize_clip_model_name("org/model a") == "org-model_a"
