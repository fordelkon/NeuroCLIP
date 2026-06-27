import pytest
import torch

from src.models.components.simple_nice import (
    NICE,
    EEGEncoder,
    FlattenHead,
    FlattenProjEEG,
    ResidualAdd,
)


def test_flatten_head_preserves_batch_and_flattens_remaining_dims() -> None:
    x = torch.randn(2, 3, 4)
    head = FlattenHead()

    output = head(x)

    assert output.shape == (2, 12)


def test_residual_add_preserves_shape_and_adds_function_output() -> None:
    x = torch.ones(2, 4)
    block = ResidualAdd(torch.nn.Linear(4, 4, bias=False))
    torch.nn.init.zeros_(block.fn.weight)

    output = block(x)

    assert output.shape == x.shape
    assert torch.equal(output, x)


def test_nice_projection_projects_eeg_to_clip_dict() -> None:
    model = NICE(
        emb_size=8,
        num_channels=4,
        emb_dim=112,
        proj_dim=16,
        p=0.0,
        ts_filters=3,
        temporal_kernel=3,
        pool_kernel=3,
        pool_stride=2,
    )
    x = torch.randn(2, 4, 32)

    output = model(x)

    assert output["eeg_clip"].shape == (2, 16)


def test_nice_rejects_wrong_eeg_shape() -> None:
    model = NICE(
        emb_size=8,
        num_channels=4,
        emb_dim=112,
        proj_dim=16,
        p=0.0,
        ts_filters=3,
        temporal_kernel=3,
        pool_kernel=3,
        pool_stride=2,
        sequence_length=32,
    )

    with pytest.raises(ValueError, match="Expected EEG shape"):
        model(torch.randn(2, 3, 32))


def test_eeg_encoder_rejects_non_3d_eeg() -> None:
    encoder = EEGEncoder(
        emb_size=8,
        num_channels=4,
        sequence_length=32,
        p=0.0,
        ts_filters=3,
        temporal_kernel=3,
        pool_kernel=3,
        pool_stride=2,
    )

    with pytest.raises(ValueError, match="Expected EEG shape"):
        encoder(torch.randn(2, 4, 32, 1))


def test_flatten_projection_projects_raw_eeg() -> None:
    model = FlattenProjEEG(num_channels=4, time_points=8, proj_dim=16, p=0.0)
    x = torch.randn(2, 4, 8)

    output = model(x)

    assert output["eeg_clip"].shape == (2, 16)


def test_flatten_projection_rejects_wrong_eeg_shape() -> None:
    model = FlattenProjEEG(num_channels=4, time_points=8, proj_dim=16, p=0.0)

    with pytest.raises(ValueError, match="Expected EEG shape"):
        model(torch.randn(2, 4, 9))


def test_legacy_class_names_are_not_exported() -> None:
    import src.models.components.simple_nice as simple_nice

    legacy_names = [
        "Enc_eeg",
        "Proj_eeg",
        "Proj_img",
        "Proj_text",
        "Flatten_Proj_eeg",
        "NICE_PRO",
        "NICE_IMG_PRO",
    ]

    for name in legacy_names:
        with pytest.raises(AttributeError):
            getattr(simple_nice, name)
