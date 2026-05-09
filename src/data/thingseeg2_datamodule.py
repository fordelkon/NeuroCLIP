from pathlib import Path
from typing import Any, Literal

import torch
from lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.data.components.thingseeg2_dataset import ThingsEEG2Dataset

StrPath = str | Path
Subject = str | int
SubjectChoice = Subject | list[Subject] | tuple[Subject, ...]
ExperimentSetting = Literal["intra-subject", "cross-subject"]

ALL_SUBJECTS = ["sub-" + str(subject_id).zfill(2) for subject_id in range(1, 11)]


class ThingsEEG2DataModule(LightningDataModule):
    """`LightningDataModule` for preprocessed THINGS-EEG2 EEG and CLIP features."""

    def __init__(
        self,
        eeg_data_dir: StrPath,
        clip_features_dir: StrPath,
        subjects: SubjectChoice = "sub-01",
        experiment_setting: ExperimentSetting = "intra-subject",
        train_val_split: list[int]
        | tuple[int, int]
        | list[float]
        | tuple[float, float] = (
            0.95,
            0.05,
        ),
        train_batch_size: int = 256,
        val_batch_size: int = 200,
        test_batch_size: int = 200,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        k_fold: int | None = None,
        fold_idx: int = 0,
        average_reps: bool = False,
        selected_channels: list[str] | tuple[str, ...] | None = None,
        model_name: str = "ViT-B-32",
        model_id: str | None = None,
        feature_mode: str = "pooled",
    ) -> None:
        """Initialize a `ThingsEEG2DataModule`.

        :param eeg_data_dir: Directory storing preprocessed EEG partitions.
        :param clip_features_dir: Directory storing extracted CLIP feature partitions.
        :param subjects: Subject label, id, or sequence. In `cross-subject`, these are held-out
            test subjects and the remaining THINGS-EEG2 subjects are used for train/val.
        :param experiment_setting: Split mode, either `intra-subject` or `cross-subject`.
        :param train_val_split: Train/validation lengths or fractions for the training partition.
        :param train_batch_size: Global training batch size.
        :param val_batch_size: Global validation batch size.
        :param test_batch_size: Global test/predict batch size.
        :param num_workers: Number of dataloader workers.
        :param pin_memory: Whether dataloaders should pin memory.
        :param drop_last: Whether dataloaders should drop incomplete batches.
        :param k_fold: Number of folds for intra-training partition validation.
        :param fold_idx: Fold index to use when `k_fold` is enabled.
        :param average_reps: Average EEG repetitions inside the dataset.
        :param selected_channels: Optional channel names to select from preprocessed EEG.
        :param model_name: CLIP model shorthand used by the feature extractor.
        :param model_id: Optional explicit Hugging Face CLIP model id.
        :param feature_mode: Extracted feature mode directory, e.g. `pooled`.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.train_batch_size_per_device = train_batch_size
        self.val_batch_size_per_device = val_batch_size
        self.test_batch_size_per_device = test_batch_size

    @property
    def num_classes(self) -> int:
        """Return the number of THINGS image concept classes."""
        return 1654

    def prepare_data(self) -> None:
        """THINGS-EEG2 data is expected to be prepared by preprocess/extract commands."""
        pass

    def setup(self, stage: str | None = None) -> None:
        """Load datasets and create train/validation/test splits for the requested stage."""
        self._set_batch_sizes_per_device()

        if stage in ("fit", "validate") or stage is None:
            if self.data_train is None and self.data_val is None:
                dataset = self._build_dataset(
                    subjects=self._train_subjects(),
                    partition="training",
                )
                train_indices, val_indices = self._build_train_val_indices(dataset)
                self.data_train = Subset(dataset, train_indices)
                self.data_val = Subset(dataset, val_indices)

        if stage in ("test", "predict") or stage is None:
            if self.data_test is None:
                test_subjects = self._test_subjects()
                self.data_test = self._build_dataset(
                    subjects=test_subjects,
                    partition="test",
                )

    def describe(self, include_batch: bool = False) -> dict[str, Any]:
        """Return a structured summary of the configured datamodule state.

        :param include_batch: Whether to materialize one batch from each available dataloader.
        :return: A nested dictionary with split, dataset, dataloader, and optional batch details.
        """
        description: dict[str, Any] = {
            "experiment_setting": self.hparams.experiment_setting,
            "subjects": {
                "train": ThingsEEG2Dataset._resolve_subjects(self._train_subjects()),
                "test": (
                    ThingsEEG2Dataset._resolve_subjects(self._test_subjects())
                    if self.data_test is not None
                    else None
                ),
            },
            "paths": {
                "eeg_data_dir": str(self.hparams.eeg_data_dir),
                "clip_features_dir": str(self.hparams.clip_features_dir),
            },
            "features": {
                "model_name": self.hparams.model_name,
                "model_id": self.hparams.model_id,
                "feature_mode": self.hparams.feature_mode,
            },
            "split": {
                "train_val_split": list(self.hparams.train_val_split),
                "k_fold": self.hparams.k_fold,
                "fold_idx": self.hparams.fold_idx,
            },
            "dataset_options": {
                "average_reps": self.hparams.average_reps,
                "selected_channels": self.hparams.selected_channels,
            },
            "dataloaders": {
                "train": self._dataloader_summary(self.train_batch_size_per_device),
                "val": self._dataloader_summary(self.val_batch_size_per_device),
                "test": self._dataloader_summary(self.test_batch_size_per_device),
            },
            "datasets": {
                "train": self._dataset_summary(self.data_train),
                "val": self._dataset_summary(self.data_val),
                "test": self._dataset_summary(self.data_test),
            },
        }

        if include_batch:
            description["sample_batches"] = {
                "train": self._batch_summary(self.train_dataloader())
                if self.data_train is not None
                else None,
                "val": self._batch_summary(self.val_dataloader())
                if self.data_val is not None
                else None,
                "test": self._batch_summary(self.test_dataloader())
                if self.data_test is not None
                else None,
            }

        return description

    def _set_batch_sizes_per_device(self) -> None:
        """Divide configured batch sizes by trainer world size when attached."""
        if self.trainer is None:
            return

        world_size = self.trainer.world_size
        self.train_batch_size_per_device = self._divide_batch_size(
            self.hparams.train_batch_size, world_size
        )
        self.val_batch_size_per_device = self._divide_batch_size(
            self.hparams.val_batch_size, world_size
        )
        self.test_batch_size_per_device = self._divide_batch_size(
            self.hparams.test_batch_size, world_size
        )

    @staticmethod
    def _divide_batch_size(batch_size: int, world_size: int) -> int:
        if batch_size % world_size != 0:
            raise RuntimeError(
                f"Batch size ({batch_size}) is not divisible by the number of devices "
                f"({world_size})."
            )
        return batch_size // world_size

    def _build_dataset(self, subjects: SubjectChoice, partition: str) -> ThingsEEG2Dataset:
        """Create one THINGS-EEG2 dataset partition from shared hyperparameters."""
        return ThingsEEG2Dataset(
            eeg_data_dir=self.hparams.eeg_data_dir,
            clip_features_dir=self.hparams.clip_features_dir,
            subjects=subjects,
            partition=partition,
            average_reps=self.hparams.average_reps,
            selected_channels=self.hparams.selected_channels,
            model_name=self.hparams.model_name,
            model_id=self.hparams.model_id,
            feature_mode=self.hparams.feature_mode,
        )

    def _dataset_summary(self, dataset: Dataset | None) -> dict[str, Any] | None:
        """Describe a loaded dataset or subset without exposing implementation objects."""
        if dataset is None:
            return None

        summary: dict[str, Any] = {
            "type": type(dataset).__name__,
            "length": len(dataset),
        }
        source_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        if isinstance(dataset, Subset):
            summary["subset_indices"] = len(dataset.indices)

        if hasattr(source_dataset, "describe"):
            source_summary = source_dataset.describe()
            summary["source"] = {
                "type": type(source_dataset).__name__,
                "partition": source_summary["partition"],
                "subjects": source_summary["subjects"],
                "n_subjects": source_summary["n_subjects"],
                "n_images": source_summary["n_images"],
                "n_reps": source_summary["n_reps"],
                "n_channels": source_summary["n_channels"],
                "n_timepoints": source_summary["n_timepoints"],
                "eeg_shape": list(source_summary["eeg_shape"]),
                "image_features_shape": (
                    list(source_summary["image_features_shape"])
                    if source_summary["image_features_shape"] is not None
                    else None
                ),
                "text_features_shape": (
                    list(source_summary["text_features_shape"])
                    if source_summary["text_features_shape"] is not None
                    else None
                ),
                "clip_features_dir": source_summary["clip_features_dir"],
            }

        return summary

    def _dataloader_summary(self, batch_size: int) -> dict[str, Any]:
        """Return configured dataloader options for one split."""
        return {
            "batch_size": batch_size,
            "num_workers": self.hparams.num_workers,
            "pin_memory": self.hparams.pin_memory,
            "drop_last": self.hparams.drop_last,
        }

    @classmethod
    def _batch_summary(cls, dataloader: DataLoader[Any]) -> dict[str, Any]:
        """Return field-level type and shape information for the first dataloader batch."""
        batch = next(iter(dataloader))
        if not isinstance(batch, dict):
            return {"type": type(batch).__name__}

        return {key: cls._value_summary(value) for key, value in batch.items()}

    @classmethod
    def _value_summary(cls, value: Any) -> dict[str, Any]:
        """Describe a collated batch field."""
        if torch.is_tensor(value):
            return {
                "type": "Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, dict):
            return {
                "type": "dict",
                "keys": list(value.keys()),
                "items": {key: cls._value_summary(item) for key, item in value.items()},
            }
        if isinstance(value, (list, tuple)):
            return {
                "type": type(value).__name__,
                "length": len(value),
                "sample_type": type(value[0]).__name__ if value else None,
            }
        return {"type": type(value).__name__, "value": value}

    def _build_train_val_indices(self, dataset: Dataset) -> tuple[list[int], list[int]]:
        """Build deterministic train/validation indices."""
        if self.hparams.k_fold is not None:
            return self._build_k_fold_indices(dataset)

        train_subset, val_subset = random_split(
            dataset=dataset,
            lengths=self.hparams.train_val_split,
            generator=torch.Generator().manual_seed(42),
        )
        return list(train_subset.indices), list(val_subset.indices)

    def _build_k_fold_indices(self, dataset: Dataset) -> tuple[list[int], list[int]]:
        """Build deterministic K-Fold train/validation indices."""
        k_fold = self.hparams.k_fold
        fold_idx = self.hparams.fold_idx
        if k_fold is None:
            raise ValueError("k_fold cannot be None when building K-Fold indices.")
        if fold_idx < 0 or fold_idx >= k_fold:
            raise ValueError(f"fold_idx ({fold_idx}) must be in range [0, {k_fold - 1}].")

        indices = list(range(len(dataset)))
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        for current_fold, (train_indices, val_indices) in enumerate(kfold.split(indices)):
            if current_fold == fold_idx:
                return train_indices.tolist(), val_indices.tolist()

        raise RuntimeError("Failed to construct K-Fold split indices.")

    def _test_subjects(self) -> SubjectChoice:
        """Return test subjects for the configured experiment setting."""
        if self.hparams.experiment_setting == "intra-subject":
            return self.hparams.subjects
        if self.hparams.experiment_setting != "cross-subject":
            raise ValueError(
                "experiment_setting must be one of {'intra-subject', 'cross-subject'}."
            )

        return self.hparams.subjects

    def _train_subjects(self) -> SubjectChoice:
        """Return train/validation subjects for the configured experiment setting."""
        if self.hparams.experiment_setting == "intra-subject":
            return self.hparams.subjects
        if self.hparams.experiment_setting != "cross-subject":
            raise ValueError(
                "experiment_setting must be one of {'intra-subject', 'cross-subject'}."
            )

        held_out_subjects = set(ThingsEEG2Dataset._resolve_subjects(self.hparams.subjects))
        train_subjects = [subject for subject in ALL_SUBJECTS if subject not in held_out_subjects]
        if not train_subjects:
            raise ValueError("cross-subject requires at least one non-held-out subject.")
        return train_subjects

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=self.hparams.drop_last,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=self.hparams.drop_last,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=self.hparams.drop_last,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader."""
        return self.test_dataloader()

    def teardown(self, stage: str | None = None) -> None:
        """Lightning teardown hook."""
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Return datamodule checkpoint state."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load datamodule checkpoint state."""
        pass


if __name__ == "__main__":
    _ = ThingsEEG2DataModule(
        eeg_data_dir="data/thingseeg2-eeg2-250hz",
        clip_features_dir="data/thingseeg2-clip-features",
    )
