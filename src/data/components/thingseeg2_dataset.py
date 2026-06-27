from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config_resolvers import resolve_clip_model_id, sanitize_clip_model_name

StrPath = str | Path
Partition = Literal["training", "test"]
Subject = str | int


def _subject_label(subject: Subject) -> str:
    """Return a zero-padded THINGS-EEG2 subject label."""
    if isinstance(subject, int):
        return "sub-" + str(subject).zfill(2)
    if subject.isdigit():
        return "sub-" + str(int(subject)).zfill(2)
    return subject


def _subject_id(subject: str) -> int:
    """Parse the numeric subject id from a THINGS-EEG2 subject label."""
    try:
        return int(subject.split("-")[-1])
    except ValueError as ex:
        raise ValueError(f"Subject label must look like 'sub-01', got {subject!r}.") from ex


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors and array-like values to NumPy arrays."""
    if torch.is_tensor(value):
        return value.cpu().numpy()
    return np.asarray(value)


def _load_pt(path: Path) -> dict[str, Any]:
    """Load a torch-saved dictionary from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Missing THINGS-EEG2 dataset file: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Expected {path} to contain a dict, got {type(data).__name__}.")
    return data


class ThingsEEG2Dataset(Dataset):
    """THINGS-EEG2 preprocessed EEG aligned with offline extracted CLIP features.

    __getitem__ returns a dictionary with the following keys:

    Core fields (always present):
        - idx: Sample index in the dataset [torch.long]
        - eeg: Preprocessed EEG data [n_channels, n_timepoints] [torch.float32]
        - label: Concept/class label (0-1853 for ThingsEEG2) [torch.long]
        - img_path: Path to the stimulus image [str]
        - text: Text description of the concept [str]
        - subject: Subject identifier (e.g., "sub-01") [str]
        - subject_id: Numeric subject ID (e.g., 1 for "sub-01") [torch.long]
        - rep: Repetition index (-1 if average_reps=True, 0-3 otherwise) [torch.long]

    Optional fields (depending on configuration):
        - image_features: CLIP image embedding [D] [torch.float32]
        - text_features: CLIP text embedding [D] [torch.float32]

    Multi-view mode fields (when multiple image feature views are available):
        - selected_view: View name selected by match_label (e.g., "no_blur", "mid_blur") [str]
        - view_index: Numeric index of the selected view [int]
    """

    def __init__(
        self,
        eeg_data_dir: StrPath,
        clip_features_dir: StrPath,
        subjects: Subject | list[Subject] | tuple[Subject, ...] = "sub-01",
        partition: Partition = "training",
        average_reps: bool = False,
        selected_channels: list[str] | tuple[str, ...] | None = None,
        model_name: str = "ViT-B-32",
        model_id: str | None = None,
        feature_mode: str = "pooled",
    ) -> None:
        super().__init__()
        if partition not in ("training", "test"):
            raise ValueError("partition must be one of {'training', 'test'}.")

        self.eeg_data_dir = Path(eeg_data_dir)
        self.clip_features_dir = self._resolve_clip_features_dir(
            Path(clip_features_dir), model_name, model_id, feature_mode
        )
        self.subjects = self._resolve_subjects(subjects)
        if not self.subjects:
            raise ValueError("At least one subject is required.")

        self.partition = partition
        self.average_reps = average_reps
        self.selected_channels = list(selected_channels) if selected_channels is not None else None

        self._load_eeg()
        self._load_features()
        self._init_match_labels()

    @staticmethod
    def _resolve_subjects(subjects: Subject | list[Subject] | tuple[Subject, ...]) -> list[str]:
        """Normalize selected subject labels."""
        if isinstance(subjects, (str, int)):
            return [_subject_label(subjects)]
        return [_subject_label(subject) for subject in subjects]

    @staticmethod
    def _resolve_clip_features_dir(
        clip_features_dir: Path,
        model_name: str,
        model_id: str | None,
        feature_mode: str,
    ) -> Path:
        """Resolve either an exact feature-mode directory or an extractor root directory."""
        if (clip_features_dir / "training.pt").exists() or (
            clip_features_dir / "test.pt"
        ).exists():
            return clip_features_dir

        resolved_model_name, resolved_model_id = resolve_clip_model_id(model_name, model_id)
        return (
            clip_features_dir
            / sanitize_clip_model_name(resolved_model_name)
            / sanitize_clip_model_name(resolved_model_id)
            / feature_mode
        )

    def _load_eeg(self) -> None:
        """Load preprocessed EEG partitions for all selected subjects."""
        first = _load_pt(self.eeg_data_dir / self.subjects[0] / f"{self.partition}.pt")
        first_eeg = _to_numpy(first["eeg"]).astype(np.float32, copy=False)
        if first_eeg.ndim != 4:
            raise ValueError(
                "Expected preprocessed EEG shape (n_images, n_reps, n_channels, n_timepoints), "
                f"got {first_eeg.shape}."
            )

        channel_indices = self._resolve_channel_indices(first)
        first_eeg = first_eeg[:, :, channel_indices, :]
        n_images, n_reps, n_channels, n_timepoints = first_eeg.shape
        n_subjects = len(self.subjects)

        self.eeg_data = np.empty(
            (n_subjects, n_images, n_reps, n_channels, n_timepoints),
            dtype=np.float32,
        )
        self.img_paths = np.empty((n_subjects, n_images, n_reps), dtype=object)
        self.texts = np.empty((n_subjects, n_images, n_reps), dtype=object)
        self.labels = np.empty((n_subjects, n_images, n_reps), dtype=np.int64)

        self._assign_subject_data(0, first, channel_indices)
        for subject_idx, subject in enumerate(self.subjects[1:], start=1):
            data = _load_pt(self.eeg_data_dir / subject / f"{self.partition}.pt")
            self._validate_subject_shape(data, (n_images, n_reps), subject)
            self._assign_subject_data(subject_idx, data, channel_indices)

        self.n_subjects = n_subjects
        self.n_images = n_images
        self.n_reps = n_reps
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        if self.average_reps:
            self.eeg_data = self.eeg_data.mean(axis=2).astype(np.float32, copy=False)
            self.img_paths = self.img_paths[:, :, 0]
            self.texts = self.texts[:, :, 0]
            self.labels = self.labels[:, :, 0]

    def _resolve_channel_indices(self, data: dict[str, Any]) -> list[int]:
        """Resolve selected channel names to stored channel indices."""
        ch_names = list(data.get("ch_names", []))
        if self.selected_channels is None:
            if ch_names:
                self.ch_names = ch_names
            else:
                self.ch_names = [str(idx) for idx in range(_to_numpy(data["eeg"]).shape[2])]
            return list(range(len(self.ch_names)))

        if not ch_names:
            raise ValueError("selected_channels requires preprocessed data to include 'ch_names'.")

        missing = [channel for channel in self.selected_channels if channel not in ch_names]
        if missing:
            raise ValueError(f"Invalid channel names: {missing}. Valid channels: {ch_names}")

        self.ch_names = list(self.selected_channels)
        return [ch_names.index(channel) for channel in self.selected_channels]

    def _validate_subject_shape(
        self, data: dict[str, Any], expected_image_rep_shape: tuple[int, int], subject: str
    ) -> None:
        """Validate that every subject has the same image and repetition counts."""
        eeg = _to_numpy(data["eeg"])
        if tuple(eeg.shape[:2]) != expected_image_rep_shape:
            raise ValueError(
                f"{subject} {self.partition} EEG shape {eeg.shape[:2]} does not match "
                f"{expected_image_rep_shape}."
            )

    def _assign_subject_data(
        self, subject_idx: int, data: dict[str, Any], channel_indices: list[int]
    ) -> None:
        """Copy one subject partition into preallocated arrays."""
        self.eeg_data[subject_idx] = _to_numpy(data["eeg"])[:, :, channel_indices, :]
        self.img_paths[subject_idx] = _to_numpy(data["img"])
        self.texts[subject_idx] = _to_numpy(data["text"])
        self.labels[subject_idx] = _to_numpy(data["label"]).astype(np.int64, copy=False)

    def _load_features(self) -> None:
        """Load offline CLIP features for this partition."""
        data = _load_pt(self.clip_features_dir / f"{self.partition}.pt")
        self.image_paths = list(data.get("image_path", []))
        self.feature_texts = list(data.get("text", []))
        self.text_features = data.get("text_features")

        # Detect multi-view features (keys like "image_features_no_blur", "image_features_mid_blur")
        view_keys = [k for k in data.keys() if k.startswith("image_features_")]
        if view_keys:
            # Multi-view mode: load all views into a dict
            self.image_features = {
                k.replace("image_features_", ""): data[k].float() for k in view_keys
            }
            # Validate all views have correct length
            for view_name, features in self.image_features.items():
                self._validate_feature_length(features, f"image_features_{view_name}")
        else:
            # Single-view mode: backward compatible
            self.image_features = data.get("image_features")
            if self.image_features is not None:
                self.image_features = self.image_features.float()
                self._validate_feature_length(self.image_features, "image_features")

        if self.text_features is not None:
            self.text_features = self.text_features.float()
            self._validate_feature_length(self.text_features, "text_features")

    def _init_match_labels(self) -> None:
        """Initialize match_label for dynamic view selection."""
        if isinstance(self.image_features, dict):
            # Multi-view mode: initialize match_label
            self.view_names = list(self.image_features.keys())
            # Default to no_blur view
            default_idx = len(self.view_names) // 2
            self.match_label = np.full(len(self), default_idx, dtype=np.int32)
        else:
            # Single-view mode: no match_label needed
            self.view_names = None
            self.match_label = None

    def update_match_labels(self, indices: np.ndarray, labels: np.ndarray) -> None:
        """Update match_label for dynamic view selection based on model confidence.

        :param indices: Sample indices to update.
        :param labels: New view indices (0=first view, 1=second view, etc.).
        """
        if self.match_label is None:
            raise ValueError("match_label is not initialized. Multi-view features required.")
        self.match_label[indices] = labels

    def reset_match_labels(self) -> None:
        """Reset all match_labels to default (middle view)."""
        if self.match_label is not None:
            default_idx = len(self.view_names) // 2
            self.match_label[:] = default_idx

    def _validate_feature_length(self, features: torch.Tensor, key: str) -> None:
        """Validate feature rows align one-to-one with image indices."""
        if features.shape[0] != self.n_images:
            raise ValueError(
                f"{key} has {features.shape[0]} rows, expected {self.n_images} for "
                f"{self.partition} EEG images."
            )

    def describe(self) -> dict[str, Any]:
        """Return a compact summary of the loaded EEG and feature structure."""
        return {
            "partition": self.partition,
            "subjects": self.subjects,
            "n_subjects": self.n_subjects,
            "n_images": self.n_images,
            "n_reps": self.n_reps,
            "n_channels": self.n_channels,
            "n_timepoints": self.n_timepoints,
            "length": len(self),
            "average_reps": self.average_reps,
            "eeg_shape": tuple(self.eeg_data.shape),
            "label_shape": tuple(self.labels.shape),
            "selected_channels": self.ch_names,
            "image_features_shape": (
                {view: tuple(feat.shape) for view, feat in self.image_features.items()}
                if isinstance(self.image_features, dict)
                else (
                    tuple(self.image_features.shape) if self.image_features is not None else None
                )
            ),
            "text_features_shape": (
                tuple(self.text_features.shape) if self.text_features is not None else None
            ),
            "image_paths": len(self.image_paths),
            "texts": len(self.feature_texts),
            "clip_features_dir": str(self.clip_features_dir),
        }

    def __len__(self) -> int:
        """Return the number of subject-image samples, optionally expanded by repetitions."""
        if self.average_reps:
            return self.n_subjects * self.n_images
        return self.n_subjects * self.n_images * self.n_reps

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one EEG sample aligned with image-index CLIP features."""
        if index < 0 or index >= len(self):
            raise IndexError(index)

        if self.average_reps:
            image_idx = index % self.n_images
            subject_idx = index // self.n_images
            rep_idx = -1
            eeg = self.eeg_data[subject_idx, image_idx]
            img_path = self.img_paths[subject_idx, image_idx]
            text = self.texts[subject_idx, image_idx]
            label = self.labels[subject_idx, image_idx]
        else:
            rep_idx = index % self.n_reps
            remaining = index // self.n_reps
            image_idx = remaining % self.n_images
            subject_idx = remaining // self.n_images
            eeg = self.eeg_data[subject_idx, image_idx, rep_idx]
            img_path = self.img_paths[subject_idx, image_idx, rep_idx]
            text = self.texts[subject_idx, image_idx, rep_idx]
            label = self.labels[subject_idx, image_idx, rep_idx]

        sample = {
            "idx": torch.tensor(index, dtype=torch.long),
            "eeg": torch.from_numpy(np.asarray(eeg, dtype=np.float32)),
            "label": torch.tensor(label, dtype=torch.long),
            "img_path": str(img_path),
            "text": str(text),
            "subject": self.subjects[subject_idx],
            "subject_id": torch.tensor(_subject_id(self.subjects[subject_idx]), dtype=torch.long),
            "rep": torch.tensor(rep_idx, dtype=torch.long),
        }

        if self.image_features is not None:
            if isinstance(self.image_features, dict):
                # Multi-view mode: select view based on match_label
                selected_view_idx = self.match_label[index]
                selected_view_name = self.view_names[selected_view_idx]
                sample["image_features"] = self.image_features[selected_view_name][image_idx]
                sample["selected_view"] = selected_view_name
                sample["view_index"] = selected_view_idx
            else:
                # Single-view mode: backward compatible
                sample["image_features"] = self.image_features[image_idx]
        if self.text_features is not None:
            sample["text_features"] = self.text_features[image_idx]

        return sample
