from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.components.thingseeg2_dataset import ThingsEEG2Dataset


def _write_eeg_partition(path: Path, subject_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eeg = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    eeg = eeg + subject_offset
    img = np.array(
        [["img-a.jpg", "img-a.jpg", "img-a.jpg"], ["img-b.jpg", "img-b.jpg", "img-b.jpg"]],
        dtype=object,
    )
    text = np.array([["alpha", "alpha", "alpha"], ["beta", "beta", "beta"]], dtype=object)
    label = np.array([[7, 7, 7], [8, 8, 8]], dtype=np.int64)
    torch.save(
        {
            "eeg": eeg,
            "img": img,
            "text": text,
            "label": label,
            "ch_names": ["P7", "P5", "Oz", "O2"],
            "times": np.arange(5, dtype=np.float32),
        },
        path,
    )


def _write_clip_partition(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_path": ["img-a.jpg", "img-b.jpg"],
            "text": ["alpha", "beta"],
            "label": torch.tensor([7, 8]),
            "image_features": torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
            "text_features": torch.tensor([[3.0, 30.0], [4.0, 40.0]]),
        },
        path,
    )


def _write_multiview_clip_partition(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_path": ["img-a.jpg", "img-b.jpg"],
            "text": ["alpha", "beta"],
            "label": torch.tensor([7, 8]),
            "image_features_mid_blur": torch.tensor([[10.0, 100.0], [20.0, 200.0]]),
            "image_features_no_blur": torch.tensor([[11.0, 110.0], [21.0, 210.0]]),
            "image_features_heavy_blur": torch.tensor([[12.0, 120.0], [22.0, 220.0]]),
            "text_features": torch.tensor([[3.0, 30.0], [4.0, 40.0]]),
        },
        path,
    )


def _write_dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    eeg_dir = tmp_path / "eeg"
    clip_dir = tmp_path / "clip" / "ViT-B-32" / "laion-CLIP-ViT-B-32-laion2B-s34B-b79K" / "pooled"
    _write_eeg_partition(eeg_dir / "sub-01" / "training.pt", subject_offset=0)
    _write_eeg_partition(eeg_dir / "sub-02" / "training.pt", subject_offset=1000)
    _write_clip_partition(clip_dir / "training.pt")
    return eeg_dir, clip_dir


def test_thingseeg2_dataset_does_not_expose_exclude_subject() -> None:
    assert "exclude_subject" not in ThingsEEG2Dataset.__init__.__annotations__


def test_thingseeg2_dataset_flattens_repetitions_and_aligns_features(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_dataset_files(tmp_path)

    dataset = ThingsEEG2Dataset(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects=["sub-01", "sub-02"],
        partition="training",
        average_reps=False,
    )

    assert len(dataset) == 12

    first = dataset[0]
    assert first["idx"].item() == 0
    assert first["subject_id"].item() == 1
    assert first["subject"] == "sub-01"
    assert first["image_index"].item() == 0
    assert first["rep"].item() == 0
    assert first["eeg"].shape == (4, 5)
    assert first["label"].item() == 7
    assert first["img_path"] == "img-a.jpg"
    assert first["text"] == "alpha"
    assert first["image_features"].tolist() == [1.0, 10.0]
    assert first["text_features"].tolist() == [3.0, 30.0]

    repeated = dataset[2]
    assert repeated["image_index"].item() == 0
    assert repeated["rep"].item() == 2
    assert repeated["image_features"].tolist() == [1.0, 10.0]

    second_subject = dataset[6]
    assert second_subject["subject_id"].item() == 2
    assert second_subject["eeg"][0, 0].item() == 1000.0


def test_thingseeg2_dataset_can_average_repetitions_and_select_channels(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_dataset_files(tmp_path)

    dataset = ThingsEEG2Dataset(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        partition="training",
        average_reps=True,
        selected_channels=["Oz", "P7"],
    )

    assert len(dataset) == 2

    sample = dataset[0]
    raw_eeg = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    expected = torch.tensor(raw_eeg[0][:, [2, 0], :].mean(axis=0))
    assert sample["rep"].item() == -1
    assert sample["eeg"].shape == (2, 5)
    assert torch.equal(sample["eeg"], expected)
    assert dataset.ch_names == ["Oz", "P7"]


def test_thingseeg2_dataset_initializes_ubp_match_labels_for_multiview_features(
    tmp_path: Path,
) -> None:
    eeg_dir, clip_dir = _write_dataset_files(tmp_path)
    _write_multiview_clip_partition(clip_dir / "training.pt")

    dataset = ThingsEEG2Dataset(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        partition="training",
        average_reps=False,
    )

    assert dataset.view_names == ["mid_blur", "no_blur", "heavy_blur"]
    assert dataset.match_label.tolist() == [1, 1, 1, 1, 1, 1]
    assert dataset.describe()["image_features_shape"] == {
        "mid_blur": (2, 2),
        "no_blur": (2, 2),
        "heavy_blur": (2, 2),
    }

    sample = dataset[0]
    assert sample["selected_view"] == "no_blur"
    assert sample["view_index"] == 1
    assert sample["image_features"].tolist() == [11.0, 110.0]


def test_thingseeg2_dataset_updates_and_resets_ubp_match_labels(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_dataset_files(tmp_path)
    _write_multiview_clip_partition(clip_dir / "training.pt")
    dataset = ThingsEEG2Dataset(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        partition="training",
        average_reps=False,
    )

    dataset.update_match_labels(
        indices=np.array([0, 3], dtype=np.int64),
        labels=np.array([0, 2], dtype=np.int32),
    )

    low_confidence = dataset[0]
    high_confidence = dataset[3]
    assert low_confidence["selected_view"] == "mid_blur"
    assert low_confidence["view_index"] == 0
    assert low_confidence["image_features"].tolist() == [10.0, 100.0]
    assert high_confidence["selected_view"] == "heavy_blur"
    assert high_confidence["view_index"] == 2
    assert high_confidence["image_features"].tolist() == [22.0, 220.0]

    dataset.reset_match_labels()

    assert dataset.match_label.tolist() == [1, 1, 1, 1, 1, 1]
    reset_sample = dataset[3]
    assert reset_sample["selected_view"] == "no_blur"
    assert reset_sample["view_index"] == 1
    assert reset_sample["image_features"].tolist() == [21.0, 210.0]


def test_thingseeg2_dataset_rejects_ubp_match_label_updates_for_single_view_features(
    tmp_path: Path,
) -> None:
    eeg_dir, clip_dir = _write_dataset_files(tmp_path)
    dataset = ThingsEEG2Dataset(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        partition="training",
    )

    with pytest.raises(ValueError, match="Multi-view features required"):
        dataset.update_match_labels(
            indices=np.array([0], dtype=np.int64),
            labels=np.array([0], dtype=np.int32),
        )


def test_thingseeg2_dataset_describes_loaded_structure(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_dataset_files(tmp_path)

    dataset = ThingsEEG2Dataset(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects=["sub-01", "sub-02"],
        partition="training",
        average_reps=False,
        selected_channels=["Oz", "P7"],
    )

    summary = dataset.describe()

    assert summary == {
        "partition": "training",
        "subjects": ["sub-01", "sub-02"],
        "n_subjects": 2,
        "n_images": 2,
        "n_reps": 3,
        "n_channels": 2,
        "n_timepoints": 5,
        "length": 12,
        "average_reps": False,
        "eeg_shape": (2, 2, 3, 2, 5),
        "label_shape": (2, 2, 3),
        "selected_channels": ["Oz", "P7"],
        "image_features_shape": (2, 2),
        "text_features_shape": (2, 2),
        "image_paths": 2,
        "texts": 2,
        "clip_features_dir": str(clip_dir),
    }


def test_thingseeg2_dataset_rejects_missing_selected_channels(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_dataset_files(tmp_path)

    with pytest.raises(ValueError, match="Invalid channel names"):
        ThingsEEG2Dataset(
            eeg_data_dir=eeg_dir,
            clip_features_dir=clip_dir,
            subjects="sub-01",
            partition="training",
            selected_channels=["NotAChannel"],
        )
