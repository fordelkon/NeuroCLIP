import pytest
import torch

from src.models.components.simple_atms import ATMS


def _small_atms(**kwargs) -> ATMS:
    return ATMS(
        num_channels=4,
        sequence_length=32,
        num_subjects=10,
        nhead=4,
        num_layers=1,
        emb_size=8,
        emb_dim=112,
        proj_dim=16,
        p=0.0,
        ts_filters=3,
        temporal_kernel=3,
        pool_kernel=3,
        pool_stride=2,
        **kwargs,
    )


def test_atms_projects_eeg_to_clip_dict_without_subject_embedding() -> None:
    model = _small_atms(use_subject_embedding=False)
    x = torch.randn(2, 4, 32)

    output = model(x)

    assert output["eeg_clip"].shape == (2, 16)


def test_atms_accepts_one_based_subject_ids_when_subject_embedding_is_enabled() -> None:
    model = _small_atms(use_subject_embedding=True)
    x = torch.randn(2, 4, 32)
    subject_ids = torch.tensor([1, 10])

    output = model(x, subject_ids=subject_ids)

    assert output["eeg_clip"].shape == (2, 16)


def test_atms_requires_subject_ids_when_subject_embedding_is_enabled() -> None:
    model = _small_atms(use_subject_embedding=True)
    x = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="subject_ids"):
        model(x)


def test_atms_rejects_out_of_range_subject_ids() -> None:
    model = _small_atms(use_subject_embedding=True)
    x = torch.randn(2, 4, 32)

    with pytest.raises(ValueError, match="range"):
        model(x, subject_ids=torch.tensor([0, 11]))
