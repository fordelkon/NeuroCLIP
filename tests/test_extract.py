from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from extractors.thingseeg2 import (
    Thingseeg2ClipExtractor,
    _projected_pooler_output,
    collect_partition_metadata,
    resolve_device,
    resolve_devices,
)
from src.extract import extract


class DummyExtractor:
    """Small configured extractor used to exercise the generic entry point."""

    def __init__(self, value: int) -> None:
        self.value = value

    def run(self) -> dict[str, Any]:
        return {"value": self.value}


class FakeClipBackend:
    """Deterministic backend that avoids loading a real CLIP model in tests."""

    def encode_images(self, image_paths: list[str]) -> torch.Tensor:
        rows = [[float(i), float(len(path))] for i, path in enumerate(image_paths)]
        return torch.tensor(rows, dtype=torch.float32)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        rows = [[float(i), float(len(text))] for i, text in enumerate(texts)]
        return torch.tensor(rows, dtype=torch.float32)

    def close(self) -> None:
        return None


class MultiViewFakeClipBackend:
    """Deterministic backend that returns multi-view features."""

    def __init__(self, blur_levels: dict[str, float]) -> None:
        self.blur_levels = blur_levels

    def encode_images(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        features_dict = {}
        for view_name, sigma in self.blur_levels.items():
            rows = [[float(i), float(len(path)), sigma] for i, path in enumerate(image_paths)]
            features_dict[view_name] = torch.tensor(rows, dtype=torch.float32)
        return features_dict

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        rows = [[float(i), float(len(text))] for i, text in enumerate(texts)]
        return torch.tensor(rows, dtype=torch.float32)

    def close(self) -> None:
        return None


class DeviceAwareFakeClipBackend:
    """Backend that records which logical device encoded each row."""

    def __init__(self, device: str) -> None:
        self.device = device

    def encode_images(self, image_paths: list[str]) -> torch.Tensor:
        device_index = int(self.device.split(":")[1])
        rows = [[float(device_index), float(len(path))] for path in image_paths]
        return torch.tensor(rows, dtype=torch.float32)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        device_index = int(self.device.split(":")[1])
        rows = [[float(device_index), float(len(text))] for text in texts]
        return torch.tensor(rows, dtype=torch.float32)

    def close(self) -> None:
        return None


class FakeModelOutput:
    """Minimal stand-in for transformers BaseModelOutputWithPooling."""

    def __init__(self, pooler_output: torch.Tensor) -> None:
        self.pooler_output = pooler_output


def _write_preprocessed_partition(path: Path) -> None:
    img = np.array(
        [
            ["img-a.jpg", "img-a.jpg"],
            ["img-b.jpg", "img-b.jpg"],
            ["img-c.jpg", "img-c.jpg"],
        ],
        dtype=object,
    )
    text = np.array(
        [
            ["alpha", "alpha"],
            ["beta", "beta"],
            ["gamma", "gamma"],
        ],
        dtype=object,
    )
    label = np.array([[7, 7], [8, 8], [9, 9]], dtype=np.int64)
    eeg = np.zeros((3, 2, 1, 4), dtype=np.float32)
    torch.save({"eeg": eeg, "img": img, "text": text, "label": label}, path)


def test_extract_run(cfg_extract: DictConfig) -> None:
    """Run the configured extraction entry point."""
    HydraConfig().set_config(cfg_extract)
    with open_dict(cfg_extract):
        cfg_extract.extract = {
            "_target_": "tests.test_extract.DummyExtractor",
            "value": 11,
        }

    metric_dict, object_dict = extract(cfg_extract)

    assert metric_dict == {"value": 11}
    assert isinstance(object_dict["extractor"], DummyExtractor)


def test_collect_partition_metadata_uses_first_repetition_as_image_index(tmp_path: Path) -> None:
    """Use EEG image order as the feature order so image_idx aligns directly."""
    partition_path = tmp_path / "training.pt"
    _write_preprocessed_partition(partition_path)

    metadata = collect_partition_metadata(partition_path)

    assert metadata.image_paths == ["img-a.jpg", "img-b.jpg", "img-c.jpg"]
    assert metadata.texts == ["alpha", "beta", "gamma"]
    assert metadata.labels.tolist() == [7, 8, 9]


def test_projected_pooler_output_accepts_transformers_model_output() -> None:
    """Use the projected CLIP pooler tensor from current transformers outputs."""
    features = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    output = FakeModelOutput(features)

    assert _projected_pooler_output(output) is features


def test_clip_extractor_saves_features_in_eeg_image_order(tmp_path: Path) -> None:
    """Save feature tensors in the same image order as the preprocessed EEG."""
    prep_dir = tmp_path / "prep"
    subject_dir = prep_dir / "sub-01"
    subject_dir.mkdir(parents=True)
    _write_preprocessed_partition(subject_dir / "training.pt")

    extractor = Thingseeg2ClipExtractor(
        prep_eeg_data_dir=prep_dir,
        save_dir=tmp_path / "features",
        reference_subject_id=1,
        model_name="fake-clip",
        model_id="fake/clip",
        model_cache_dir=None,
        partitions=["training"],
        batch_size=2,
        device="cpu",
        feature_mode="pooled",
        extract_image=True,
        extract_text=True,
        backend_factory=lambda _: FakeClipBackend(),
    )

    outputs = extractor.run()
    saved = torch.load(outputs["training"], weights_only=False)

    assert saved["image_path"] == ["img-a.jpg", "img-b.jpg", "img-c.jpg"]
    assert saved["text"] == ["alpha", "beta", "gamma"]
    assert saved["label"].tolist() == [7, 8, 9]
    assert saved["image_features"].shape == (3, 2)
    assert saved["text_features"].shape == (3, 2)
    assert saved["image_features"][1].tolist() == [1.0, float(len("img-b.jpg"))]
    assert saved["metadata"]["reference_subject_id"] == 1
    assert "reference_subject" not in saved["metadata"]


def test_clip_extractor_distributes_batches_across_all_cuda_devices(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Multi-GPU extraction should split each batch and preserve row order."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    prep_dir = tmp_path / "prep"
    subject_dir = prep_dir / "sub-01"
    subject_dir.mkdir(parents=True)
    _write_preprocessed_partition(subject_dir / "training.pt")
    created_devices: list[str] = []

    def backend_factory(_: Thingseeg2ClipExtractor, device: str) -> DeviceAwareFakeClipBackend:
        created_devices.append(device)
        return DeviceAwareFakeClipBackend(device)

    extractor = Thingseeg2ClipExtractor(
        prep_eeg_data_dir=prep_dir,
        save_dir=tmp_path / "features",
        reference_subject_id=1,
        model_name="fake-clip",
        model_id="fake/clip",
        partitions=["training"],
        batch_size=3,
        device="auto",
        extract_image=True,
        extract_text=False,
        backend_factory=backend_factory,
    )

    outputs = extractor.run()
    saved = torch.load(outputs["training"], weights_only=False)

    assert created_devices == ["cuda:0", "cuda:1"]
    assert saved["metadata"]["device"] == "cuda:0,cuda:1"
    assert saved["image_path"] == ["img-a.jpg", "img-b.jpg", "img-c.jpg"]
    assert saved["image_features"][:, 0].tolist() == [0.0, 0.0, 1.0]


def test_resolve_device_rejects_cuda_when_unavailable(monkeypatch: Any) -> None:
    """Fail early with a clear error when CUDA is requested but unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    try:
        resolve_device("cuda")
    except ValueError as error:
        assert "CUDA was requested" in str(error)
    else:
        raise AssertionError("resolve_device should reject unavailable CUDA")


def test_resolve_devices_uses_all_cuda_devices_for_auto(monkeypatch: Any) -> None:
    """Default CUDA extraction should use every visible GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    assert resolve_devices("auto") == ("cuda:0", "cuda:1", "cuda:2")
    assert resolve_device("auto") == "cuda:0,cuda:1,cuda:2"


def test_resolve_devices_keeps_explicit_cuda_device(monkeypatch: Any) -> None:
    """An explicit CUDA index should opt out of default all-GPU inference."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    assert resolve_devices("cuda:1") == ("cuda:1",)


def test_output_dir_includes_resolved_model_id(tmp_path: Path) -> None:
    """Avoid collisions when the same short model name points to different ids."""
    extractor_a = Thingseeg2ClipExtractor(
        prep_eeg_data_dir=tmp_path / "prep",
        save_dir=tmp_path / "features",
        reference_subject_id=1,
        model_name="custom",
        model_id="org/model-a",
        device="cpu",
    )
    extractor_b = Thingseeg2ClipExtractor(
        prep_eeg_data_dir=tmp_path / "prep",
        save_dir=tmp_path / "features",
        reference_subject_id=1,
        model_name="custom",
        model_id="org/model-b",
        device="cpu",
    )

    assert extractor_a.output_dir != extractor_b.output_dir
    assert "org-model-a" in str(extractor_a.output_dir)
    assert "org-model-b" in str(extractor_b.output_dir)


def test_collect_partition_metadata_rejects_empty_partition(tmp_path: Path) -> None:
    """Report malformed preprocessed partitions before batch encoding."""
    partition_path = tmp_path / "training.pt"
    torch.save(
        {
            "img": np.empty((0, 2), dtype=object),
            "text": np.empty((0, 2), dtype=object),
            "label": np.empty((0, 2), dtype=np.int64),
        },
        partition_path,
    )

    try:
        collect_partition_metadata(partition_path)
    except ValueError as error:
        assert "at least one image" in str(error)
    else:
        raise AssertionError("empty partitions should be rejected")


def test_clip_extractor_saves_multiview_features(tmp_path: Path) -> None:
    """Multi-view extraction should save features with view-specific keys."""
    prep_dir = tmp_path / "prep"
    subject_dir = prep_dir / "sub-01"
    subject_dir.mkdir(parents=True)
    _write_preprocessed_partition(subject_dir / "training.pt")

    blur_levels = {"sharp": 0.0, "mid_blur": 5.0, "heavy_blur": 10.0}
    extractor = Thingseeg2ClipExtractor(
        prep_eeg_data_dir=prep_dir,
        save_dir=tmp_path / "features",
        reference_subject_id=1,
        model_name="fake-clip",
        model_id="fake/clip",
        partitions=["training"],
        batch_size=2,
        device="cpu",
        extract_image=True,
        extract_text=True,
        blur_levels=blur_levels,
        backend_factory=lambda _: MultiViewFakeClipBackend(blur_levels),
    )

    outputs = extractor.run()
    saved = torch.load(outputs["training"], weights_only=False)

    assert "image_features_sharp" in saved
    assert "image_features_mid_blur" in saved
    assert "image_features_heavy_blur" in saved
    assert "image_features" not in saved
    assert saved["image_features_sharp"].shape == (3, 3)
    assert saved["image_features_mid_blur"].shape == (3, 3)
    assert saved["image_features_heavy_blur"].shape == (3, 3)
    assert saved["metadata"]["blur_levels"] == blur_levels
