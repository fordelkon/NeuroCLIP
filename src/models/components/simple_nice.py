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
        p: float = 0.5,
        ts_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 51,
        pool_stride: int = 5,
    ):
        """Build the temporal-spatial EEG encoder."""
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
        self.flatten_dim = num_channels * time_points
        self.flatten_eeg_proj = EEGProjectionHead(
            emb_dim=self.flatten_dim,
            proj_dim=proj_dim,
            p=p,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a CLIP-aligned EEG embedding."""
        x = x.contiguous().view(x.shape[0], self.flatten_dim)
        return {"eeg_clip": self.flatten_eeg_proj(x)}


class NICE(nn.Module):
    """Encode EEG and project it into a CLIP-aligned embedding."""

    def __init__(
        self,
        emb_size: int = 40,
        num_channels: int = 17,
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
        self.enc_eeg = EEGEncoder(
            emb_size=emb_size,
            num_channels=num_channels,
            p=p,
            ts_filters=ts_filters,
            temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )
        self.proj_eeg = EEGProjectionHead(emb_dim=emb_dim, proj_dim=proj_dim, p=p)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``eeg_clip`` with shape ``[batch, proj_dim]``."""
        eeg_emb = self.enc_eeg(x)
        return {"eeg_clip": self.proj_eeg(eeg_emb)}
