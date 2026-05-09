from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from src.data.thingseeg2_datamodule import ThingsEEG2DataModule


def _write_eeg_partition(path: Path, subject_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eeg = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    eeg = eeg + subject_offset
    img = np.array(
        [
            ["img-a.jpg", "img-a.jpg", "img-a.jpg"],
            ["img-b.jpg", "img-b.jpg", "img-b.jpg"],
        ],
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


def _write_datamodule_files(tmp_path: Path) -> tuple[Path, Path]:
    eeg_dir = tmp_path / "eeg"
    clip_dir = tmp_path / "clip" / "ViT-B-32" / "laion-CLIP-ViT-B-32-laion2B-s34B-b79K" / "pooled"
    for subject_id in range(1, 11):
        subject = "sub-" + str(subject_id).zfill(2)
        offset = subject_id * 1000
        _write_eeg_partition(eeg_dir / subject / "training.pt", subject_offset=offset)
        _write_eeg_partition(eeg_dir / subject / "test.pt", subject_offset=offset + 100)

    _write_clip_partition(clip_dir / "training.pt")
    _write_clip_partition(clip_dir / "test.pt")
    return eeg_dir, clip_dir


def test_thingseeg2_datamodule_builds_intra_subject_loaders(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_datamodule_files(tmp_path)
    dm = ThingsEEG2DataModule(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        train_val_split=[4, 2],
        train_batch_size=2,
        val_batch_size=2,
        test_batch_size=3,
        num_workers=0,
    )

    dm.setup()

    assert len(dm.data_train) == 4
    assert len(dm.data_val) == 2
    assert len(dm.data_test) == 6
    batch = next(iter(dm.train_dataloader()))
    assert batch["eeg"].shape == (2, 4, 5)
    assert batch["image_features"].shape == (2, 2)


def test_thingseeg2_datamodule_uses_k_fold_for_intra_subject_split(
    tmp_path: Path,
) -> None:
    eeg_dir, clip_dir = _write_datamodule_files(tmp_path)
    dm = ThingsEEG2DataModule(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        k_fold=3,
        fold_idx=1,
        train_batch_size=2,
        val_batch_size=2,
        num_workers=0,
    )

    dm.setup("fit")

    assert len(dm.data_train) == 4
    assert len(dm.data_val) == 2


def test_thingseeg2_datamodule_uses_held_out_subject_for_cross_subject(
    tmp_path: Path,
) -> None:
    eeg_dir, clip_dir = _write_datamodule_files(tmp_path)
    dm = ThingsEEG2DataModule(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        experiment_setting="cross-subject",
        train_val_split=[48, 6],
        train_batch_size=3,
        val_batch_size=3,
        test_batch_size=4,
        num_workers=0,
    )

    dm.setup()

    assert dm.data_train.dataset.subjects == [
        "sub-02",
        "sub-03",
        "sub-04",
        "sub-05",
        "sub-06",
        "sub-07",
        "sub-08",
        "sub-09",
        "sub-10",
    ]
    assert len(dm.data_train) == 48
    assert len(dm.data_val) == 6
    assert dm.data_test.subjects == ["sub-01"]
    assert len(dm.data_test) == 6


def test_thingseeg2_data_config_can_instantiate(tmp_path: Path) -> None:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="train.yaml", overrides=["data=thingseeg2"])

        with open_dict(cfg):
            cfg.paths.root_dir = str(tmp_path)
            cfg.paths.data_dir = str(tmp_path / "data")
            cfg.paths.thingseeg2_preprocessed_dir = str(tmp_path / "eeg")
            cfg.paths.thingseeg2_clip_features_dir = str(tmp_path / "clip")

        datamodule = hydra.utils.instantiate(cfg.data)

    GlobalHydra.instance().clear()

    assert isinstance(datamodule, ThingsEEG2DataModule)
    assert datamodule.hparams.eeg_data_dir == str(tmp_path / "eeg")
    assert datamodule.hparams.clip_features_dir == str(tmp_path / "clip")


def test_thingseeg2_datamodule_rejects_invalid_fold(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_datamodule_files(tmp_path)
    dm = ThingsEEG2DataModule(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        k_fold=3,
        fold_idx=3,
        num_workers=0,
    )

    with pytest.raises(ValueError, match="fold_idx"):
        dm.setup("fit")


def test_thingseeg2_datamodule_describes_setup_state(tmp_path: Path) -> None:
    eeg_dir, clip_dir = _write_datamodule_files(tmp_path)
    dm = ThingsEEG2DataModule(
        eeg_data_dir=eeg_dir,
        clip_features_dir=clip_dir,
        subjects="sub-01",
        experiment_setting="cross-subject",
        train_val_split=[48, 6],
        train_batch_size=3,
        val_batch_size=3,
        test_batch_size=4,
        num_workers=0,
    )

    dm.setup()

    description = dm.describe(include_batch=True)

    assert description["experiment_setting"] == "cross-subject"
    assert description["subjects"]["train"] == [
        "sub-02",
        "sub-03",
        "sub-04",
        "sub-05",
        "sub-06",
        "sub-07",
        "sub-08",
        "sub-09",
        "sub-10",
    ]
    assert description["subjects"]["test"] == ["sub-01"]
    assert description["datasets"]["train"]["length"] == 48
    assert description["datasets"]["val"]["length"] == 6
    assert description["datasets"]["test"]["length"] == 6
    assert description["dataloaders"]["train"]["batch_size"] == 3
    assert description["sample_batches"]["train"]["eeg"]["shape"] == [3, 4, 5]
    assert description["sample_batches"]["train"]["image_features"]["shape"] == [3, 2]
