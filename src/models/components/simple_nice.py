"""Compact NICE-style projection modules for EEG and image embeddings."""

import torch
from torch import nn

from src.models.components.transformers.embeddings import TDTEmbedding, TSEmbedding


class FlattenHead(nn.Module):
    """Flatten every non-batch dimension into a single feature dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor with shape ``[batch, features]``."""
        return x.contiguous().view(x.size(0), -1)


class ResidualAdd(nn.Module):
    """Wrap a module with a residual connection."""

    def __init__(self, fn: nn.Module):
        """Initialize the residual wrapper."""
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply ``fn`` and add the original input."""
        return x + self.fn(x, **kwargs)


class EEGEncoder(nn.Sequential):
    """Encode raw EEG signals into flattened temporal-spatial tokens."""

    def __init__(
        self,
        emb_size: int = 40,
        num_channels: int = 17,
        sequence_length: int | None = None,
        p: float = 0.5,
        ts_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 51,
        pool_stride: int = 5,
    ):
        """Build the temporal-spatial EEG encoder."""
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        super().__init__(
            TSEmbedding(
                emb_size=emb_size,
                k=ts_filters,
                m1=temporal_kernel,
                m2=pool_kernel,
                s=pool_stride,
                channels=num_channels,
                dropout=p,
            ),
            FlattenHead(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened EEG features after validating input shape."""
        self._validate_eeg_shape(x)
        return super().forward(x)

    def _validate_eeg_shape(self, x: torch.Tensor) -> None:
        """Validate the EEG tensor shape expected by the encoder."""
        if x.ndim != 3:
            raise ValueError(f"Expected EEG shape [batch, channels, time], got {tuple(x.shape)}.")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected EEG shape [batch, {self.num_channels}, time], got {tuple(x.shape)}."
            )
        if self.sequence_length is not None and x.shape[2] != self.sequence_length:
            raise ValueError(
                "Expected EEG shape "
                f"[batch, {self.num_channels}, {self.sequence_length}], got {tuple(x.shape)}."
            )


class ProjectionHead(nn.Sequential):
    """Project features into a CLIP-aligned embedding space."""

    def __init__(self, emb_dim: int = 512, proj_dim: int = 512, p: float = 0.5):
        """Build a linear projection with a residual MLP refinement block."""
        super().__init__(
            nn.Linear(emb_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(p),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class EEGProjectionHead(ProjectionHead):
    """Projection head for flattened EEG features."""


class ImageProjectionHead(ProjectionHead):
    """Projection head for image features."""


class TextProjectionHead(ProjectionHead):
    """Projection head for text features."""


class FlattenProjEEG(nn.Module):
    """Project raw EEG by flattening channels and time directly."""

    def __init__(
        self,
        num_channels: int = 17,
        time_points: int = 250,
        proj_dim: int = 512,
        p: float = 0.5,
    ):
        """Initialize the direct EEG projection module."""
        super().__init__()
        self.num_channels = num_channels
        self.time_points = time_points
        self.flatten_dim = num_channels * time_points
        self.flatten_eeg_proj = EEGProjectionHead(
            emb_dim=self.flatten_dim,
            proj_dim=proj_dim,
            p=p,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a CLIP-aligned EEG embedding."""
        self._validate_eeg_shape(x)
        x = x.contiguous().view(x.shape[0], self.flatten_dim)
        return {"eeg_clip": self.flatten_eeg_proj(x)}

    def _validate_eeg_shape(self, x: torch.Tensor) -> None:
        """Validate the EEG tensor shape expected by the flatten projector."""
        if x.ndim != 3:
            raise ValueError(f"Expected EEG shape [batch, channels, time], got {tuple(x.shape)}.")
        if x.shape[1] != self.num_channels or x.shape[2] != self.time_points:
            raise ValueError(
                "Expected EEG shape "
                f"[batch, {self.num_channels}, {self.time_points}], got {tuple(x.shape)}."
            )


class NICE(nn.Module):
    """Encode EEG and project it into a CLIP-aligned embedding."""

    def __init__(
        self,
        emb_size: int = 40,
        num_channels: int = 17,
        sequence_length: int | None = None,
        emb_dim: int = 1440,
        proj_dim: int = 512,
        p: float = 0.5,
        ts_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 51,
        pool_stride: int = 5,
    ):
        """Initialize the single-head NICE EEG projection model."""
        super().__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.enc_eeg = EEGEncoder(
            emb_size=emb_size,
            num_channels=num_channels,
            sequence_length=sequence_length,
            p=p,
            ts_filters=ts_filters,
            temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )
        self.proj_eeg = EEGProjectionHead(emb_dim=emb_dim, proj_dim=proj_dim, p=p)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``eeg_clip`` with shape ``[batch, proj_dim]``."""
        self._validate_eeg_shape(x)
        eeg_emb = self.enc_eeg(x)
        return {"eeg_clip": self.proj_eeg(eeg_emb)}

    def _validate_eeg_shape(self, x: torch.Tensor) -> None:
        """Validate the EEG tensor shape expected by NICE."""
        if x.ndim != 3:
            raise ValueError(f"Expected EEG shape [batch, channels, time], got {tuple(x.shape)}.")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected EEG shape [batch, {self.num_channels}, time], got {tuple(x.shape)}."
            )
        if self.sequence_length is not None and x.shape[2] != self.sequence_length:
            raise ValueError(
                "Expected EEG shape "
                f"[batch, {self.num_channels}, {self.sequence_length}], got {tuple(x.shape)}."
            )
