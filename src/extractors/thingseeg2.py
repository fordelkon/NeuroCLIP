import gc
import inspect
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, Union

import numpy as np
import torch
from PIL import ImageFilter
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.utils.config_resolvers import resolve_clip_model_id, sanitize_clip_model_name

StrPath = Union[str, os.PathLike[str]]
ImageClipFeatureMode = Literal["pooled", "last_hidden_state_no_cls"]

console = Console()


def apply_gaussian_blur(image, sigma: float):
    """Apply Gaussian blur to a PIL image."""
    if sigma <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class ClipBackend(Protocol):
    """Interface used by the extractor to encode CLIP features."""

    def encode_images(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        """Encode a batch of image paths into multiple views."""

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text labels."""

    def close(self) -> None:
        """Release backend resources."""


@dataclass
class PartitionMetadata:
    """Image-level metadata collected from a preprocessed EEG partition."""

    image_paths: list[str]
    texts: list[str]
    labels: torch.Tensor


class HuggingFaceClipBackend:
    """Small adapter around Hugging Face CLIP classes."""

    def __init__(
        self,
        model_id: str,
        model_cache_dir: StrPath | None,
        device: str,
        feature_mode: ImageClipFeatureMode,
        blur_levels: dict[str, float] | None = None,
    ) -> None:
        from PIL import Image
        from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

        self.image_cls = Image
        self.device = device
        self.feature_mode = feature_mode
        self.blur_levels = blur_levels or {"no_blur": 0.0}
        self.model = CLIPModel.from_pretrained(model_id, cache_dir=model_cache_dir)
        self.model = self.model.to(device)
        self.model.eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_id, cache_dir=model_cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_cache_dir)

    def encode_images(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        """Encode a batch of images with multiple blur levels."""
        images = []
        for path in image_paths:
            with self.image_cls.open(path) as image:
                images.append(image.convert("RGB"))

        try:
            features_dict = {}
            for view_name, sigma in self.blur_levels.items():
                blurred_images = [apply_gaussian_blur(img, sigma) for img in images]
                inputs = self.image_processor(blurred_images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                with torch.inference_mode():
                    if self.feature_mode == "pooled":
                        features = _projected_pooler_output(
                            self.model.get_image_features(**inputs)
                        )
                        features = features / features.norm(dim=-1, keepdim=True)
                    else:
                        vision_outputs = self.model.vision_model(**inputs)
                        features = vision_outputs.last_hidden_state[:, 1:, :]
                features_dict[view_name] = features.detach().cpu().float()
            return features_dict
        finally:
            for image in images:
                image.close()

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text labels with the CLIP text tower."""
        prompts = [f"This picture is {text}" for text in texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            features = _projected_pooler_output(self.model.get_text_features(**inputs))
            features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().float()

    def close(self) -> None:
        """Release model memory."""
        del self.model, self.image_processor, self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class MultiDeviceClipBackend:
    """Distribute each encode call across device-local backends."""

    def __init__(self, backends: Sequence[ClipBackend]) -> None:
        if len(backends) < 2:
            raise ValueError("MultiDeviceClipBackend requires at least two backends.")
        self.backends = list(backends)

    def encode_images(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        """Encode a batch of images across all configured devices."""
        return self._encode_split(image_paths, "encode_images")

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text labels across all configured devices."""
        return self._encode_split(texts, "encode_texts")

    def close(self) -> None:
        """Release every device-local backend."""
        for backend in self.backends:
            backend.close()

    def _encode_split(
        self, values: list[str], method_name: str
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        chunks = _split_contiguous(values, len(self.backends))
        work = [
            (index, backend, chunk)
            for index, (backend, chunk) in enumerate(zip(self.backends, chunks))
            if chunk
        ]
        if not work:
            raise ValueError("Cannot encode an empty batch.")

        outputs: list[torch.Tensor | dict[str, torch.Tensor] | None] = [None] * len(work)
        with ThreadPoolExecutor(max_workers=len(work)) as executor:
            futures = [
                executor.submit(getattr(backend, method_name), chunk) for _, backend, chunk in work
            ]
            for output_index, future in enumerate(futures):
                outputs[output_index] = future.result()

        if isinstance(outputs[0], dict):
            view_names = outputs[0].keys()
            return {
                view_name: torch.cat(
                    [output[view_name] for output in outputs if output is not None], dim=0
                )
                for view_name in view_names
            }
        return torch.cat([output for output in outputs if output is not None], dim=0)


def create_progress() -> Progress:
    """Create an ASCII-friendly progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TextColumn("-"),
        TaskProgressColumn(),
        TextColumn("-"),
        TimeElapsedColumn(),
        TextColumn("-"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


def collect_partition_metadata(partition_path: StrPath) -> PartitionMetadata:
    """Collect image-level metadata in the same order as EEG image indices.

    Preprocessed THINGS-EEG2 partitions store repeated metadata as
    ``[n_images, n_reps]`` arrays. The first repetition is the canonical image
    order for CLIP feature extraction, so ``image_features[i]`` aligns with
    ``eeg[i, rep]`` for every repetition.
    """
    data = torch.load(partition_path, weights_only=False)
    try:
        image_paths = _first_repetition_as_list(data["img"], "img")
        texts = _first_repetition_as_list(data["text"], "text")
        labels = torch.as_tensor(_first_repetition(data["label"], "label"))
        _validate_partition_metadata(image_paths, texts, labels)
    finally:
        del data

    return PartitionMetadata(image_paths=image_paths, texts=texts, labels=labels)


def resolve_model_id(model_name: str, model_id: str | None = None) -> tuple[str, str]:
    """Resolve a simple CLIP model name to a Hugging Face model id."""
    return resolve_clip_model_id(model_name, model_id)


def resolve_device(device: str) -> str:
    """Resolve an auto/cpu/cuda device setting into a torch device string."""
    return ",".join(resolve_devices(device))


def resolve_devices(device: str) -> tuple[str, ...]:
    """Resolve an auto/cpu/cuda device setting into one or more torch devices."""
    if device == "auto":
        if not torch.cuda.is_available():
            return ("cpu",)
        return _all_cuda_devices()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
        return _all_cuda_devices()
    return (device,)


def normalize_feature_mode(feature_mode: str) -> ImageClipFeatureMode:
    """Return the canonical feature mode name used for saving and encoding."""
    if feature_mode in ("pooled", "last_hidden_state_no_cls"):
        return feature_mode
    raise ValueError("feature_mode must be one of {'pooled', 'last_hidden_state_no_cls'}.")


def _first_repetition_as_list(values: Any, field_name: str) -> list[str]:
    """Return first-repetition metadata values as strings."""
    return [str(value) for value in _first_repetition(values, field_name).tolist()]


def _first_repetition(values: Any, field_name: str) -> np.ndarray:
    """Return image-level metadata from the first repetition column."""
    array = np.asarray(values)
    if array.ndim == 1:
        return array
    if array.ndim < 2:
        raise ValueError(f"Expected {field_name} to have at least one dimension.")
    return array[:, 0]


def _validate_partition_metadata(
    image_paths: list[str],
    texts: list[str],
    labels: torch.Tensor,
) -> None:
    if len(image_paths) == 0:
        raise ValueError("Partition metadata must contain at least one image.")
    if len(image_paths) != len(texts) or len(image_paths) != len(labels):
        raise ValueError(
            "Partition metadata lengths must match: "
            f"img={len(image_paths)}, text={len(texts)}, label={len(labels)}."
        )


def _all_cuda_devices() -> tuple[str, ...]:
    """Return every visible CUDA device as torch device strings."""
    count = torch.cuda.device_count()
    if count < 1:
        raise ValueError("CUDA is available, but torch.cuda.device_count() is zero.")
    return tuple("cuda:" + str(index) for index in range(count))


def _split_contiguous(values: list[str], num_chunks: int) -> list[list[str]]:
    """Split values into contiguous chunks with stable row order."""
    chunk_count = min(len(values), num_chunks)
    base_size, remainder = divmod(len(values), chunk_count)
    chunks = []
    start_idx = 0
    for index in range(chunk_count):
        chunk_size = base_size + (1 if index < remainder else 0)
        end_idx = start_idx + chunk_size
        chunks.append(values[start_idx:end_idx])
        start_idx = end_idx
    return chunks


def _callable_accepts_device(factory: Callable[..., ClipBackend]) -> bool:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return False
    return len(signature.parameters) >= 2


def _projected_pooler_output(output: Any) -> torch.Tensor:
    """Return projected CLIP features from tensor or model-output objects."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


@dataclass
class Thingseeg2ClipExtractor:
    """Extract CLIP features for THINGS-EEG2 images using EEG metadata order."""

    prep_eeg_data_dir: StrPath
    save_dir: StrPath
    reference_subject_id: int
    model_name: str = "ViT-B-32"
    model_id: str | None = None
    model_cache_dir: StrPath | None = None
    partitions: Sequence[Literal["training", "test"]] = ("training", "test")
    batch_size: int = 64
    device: str = "auto"
    feature_mode: ImageClipFeatureMode = "pooled"
    extract_image: bool = True
    extract_text: bool = False
    blur_levels: dict[str, float] | None = None
    backend_factory: Callable[..., ClipBackend] | None = None

    def __post_init__(self) -> None:
        self.feature_mode = normalize_feature_mode(self.feature_mode)
        if not self.extract_image and not self.extract_text:
            raise ValueError("At least one of extract_image or extract_text must be true.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.blur_levels is None:
            self.blur_levels = {"no_blur": 0.0}

        self.prep_eeg_data_dir = Path(self.prep_eeg_data_dir)
        self.save_dir = Path(self.save_dir)
        self.model_name, self.resolved_model_id = resolve_model_id(self.model_name, self.model_id)
        self.resolved_devices = resolve_devices(self.device)
        self.resolved_device = ",".join(self.resolved_devices)

    @property
    def output_dir(self) -> Path:
        """Return the versioned output directory for this extractor config."""
        return (
            self.save_dir
            / sanitize_clip_model_name(self.model_name)
            / sanitize_clip_model_name(self.resolved_model_id)
            / self.feature_mode
        )

    @property
    def reference_subject_label(self) -> str:
        """Return the zero-padded subject label used by THINGS-EEG2 paths."""
        return "sub-" + str(self.reference_subject_id).zfill(2)

    def run(self) -> dict[str, str | int]:
        """Extract and save features for configured partitions."""
        console.print()
        console.print(
            Panel.fit(
                "[bold white]THINGS-EEG2 CLIP Extraction Pipeline[/bold white]\n\n"
                "[dim]Create image-index-aligned CLIP feature caches[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        self._print_config()

        backend = self._create_backend()
        outputs: dict[str, str] = {}
        try:
            for partition in self.partitions:
                outputs[str(partition)] = self._extract_partition(str(partition), backend)
        finally:
            backend.close()

        console.print()
        console.print(
            Panel.fit(
                "[bold green]Extraction Complete[/bold green]\n\n"
                f"[dim]Features saved to {self.output_dir}[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
        return {"num_partitions": len(outputs), **outputs}

    def _create_backend(self) -> ClipBackend:
        if len(self.resolved_devices) > 1:
            return MultiDeviceClipBackend(
                [self._create_backend_for_device(device) for device in self.resolved_devices]
            )
        return self._create_backend_for_device(self.resolved_devices[0])

    def _create_backend_for_device(self, device: str) -> ClipBackend:
        if self.backend_factory is not None:
            if _callable_accepts_device(self.backend_factory):
                return self.backend_factory(self, device)
            return self.backend_factory(self)
        return HuggingFaceClipBackend(
            model_id=self.resolved_model_id,
            model_cache_dir=self.model_cache_dir,
            device=device,
            feature_mode=self.feature_mode,
            blur_levels=self.blur_levels,
        )

    def _extract_partition(self, partition: str, backend: ClipBackend) -> str:
        partition_path = self.prep_eeg_data_dir / self.reference_subject_label / f"{partition}.pt"
        metadata = collect_partition_metadata(partition_path)
        image_features = None
        text_features = None

        if self.extract_image:
            image_features = self._encode_in_batches(
                values=metadata.image_paths,
                encode_batch=backend.encode_images,
                description=f"{partition} image features",
            )

        if self.extract_text:
            text_features = self._encode_in_batches(
                values=metadata.texts,
                encode_batch=backend.encode_texts,
                description=f"{partition} text features",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{partition}.pt"

        save_dict = {
            "image_path": metadata.image_paths,
            "text": metadata.texts,
            "label": metadata.labels,
            "metadata": {
                "reference_subject_id": self.reference_subject_id,
                "partition": partition,
                "model_name": self.model_name,
                "model_id": self.resolved_model_id,
                "feature_mode": self.feature_mode,
                "device": self.resolved_device,
                "extract_image": self.extract_image,
                "extract_text": self.extract_text,
                "blur_levels": self.blur_levels,
                "alignment": "image_features[i] aligns with eeg[i, rep]",
            },
        }

        if isinstance(image_features, dict):
            for view_name, features in image_features.items():
                save_dict[f"image_features_{view_name}"] = features
        else:
            save_dict["image_features"] = image_features

        save_dict["text_features"] = text_features

        torch.save(save_dict, output_path, pickle_protocol=5)
        console.print(
            f"  [green]OK[/green] Saved {partition} features: " f"[dim]{output_path}[/dim]"
        )
        return str(output_path)

    def _encode_in_batches(
        self,
        values: list[str],
        encode_batch: Callable[[list[str]], torch.Tensor | dict[str, torch.Tensor]],
        description: str,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features = []
        total_batches = (len(values) + self.batch_size - 1) // self.batch_size
        with create_progress() as progress:
            task = progress.add_task(f"[cyan]{description}", total=total_batches)
            for start_idx in range(0, len(values), self.batch_size):
                batch = values[start_idx : start_idx + self.batch_size]
                features.append(encode_batch(batch))
                progress.advance(task)

        if isinstance(features[0], dict):
            view_names = features[0].keys()
            return {
                view_name: torch.cat(
                    [batch_features[view_name] for batch_features in features], dim=0
                )
                for view_name in view_names
            }
        return torch.cat(features, dim=0)

    def _print_config(self) -> None:
        """Print the resolved extraction configuration."""
        table = Table(title="Configuration", box=box.ROUNDED, border_style="blue")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="yellow")
        table.add_row("Reference Subject ID", str(self.reference_subject_id))
        table.add_row("Partitions", ", ".join(self.partitions))
        table.add_row("Model", self.resolved_model_id)
        table.add_row("Feature Mode", self.feature_mode)
        table.add_row("Device", self.resolved_device)
        table.add_row("Batch Size", str(self.batch_size))
        table.add_row("Extract Image", str(self.extract_image))
        table.add_row("Extract Text", str(self.extract_text))
        console.print(table)
        console.print()
