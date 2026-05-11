"""ATMS EEG encoder with optional subject conditioning."""

import torch
from torch import nn

from src.models.components.simple_nice import NICE


class ATMS(nn.Module):
    """Apply channel-wise self-attention before NICE projection."""

    def __init__(
        self,
        num_channels: int = 17,
        sequence_length: int = 250,
        num_subjects: int = 10,
        nhead: int = 2,
        num_layers: int = 1,
        emb_size: int = 40,
        emb_dim: int = 1440,
        proj_dim: int = 512,
        p: float = 0.5,
        use_subject_embedding: bool = False,
        ts_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 51,
        pool_stride: int = 5,
    ) -> None:
        """Initialize ATMS.

        ``use_subject_embedding`` is intended for multi-subject training, especially
        cross-subject runs. THINGS-EEG2 subject ids are expected to be 1-based.
        """
        super().__init__()
        self.num_channels = num_channels
        self.sequence_length = sequence_length
        self.num_subjects = num_subjects
        self.use_subject_embedding = use_subject_embedding

        self.subject_embedding = nn.Embedding(num_subjects, sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=sequence_length,
            nhead=nhead,
            dim_feedforward=256,
            dropout=p / 2,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(sequence_length)
        self.dropout = nn.Dropout(p / 2)
        self.nice = NICE(
            emb_size=emb_size,
            num_channels=num_channels,
            emb_dim=emb_dim,
            proj_dim=proj_dim,
            p=p,
            ts_filters=ts_filters,
            temporal_kernel=temporal_kernel,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return ``eeg_clip`` with shape ``[batch, proj_dim]``."""
        self._validate_eeg_shape(x)
        if self.use_subject_embedding:
            if subject_ids is None:
                raise ValueError("subject_ids are required when subject embedding is enabled.")
            x = x + self._subject_embedding(subject_ids, x)

        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.nice(x)

    def _validate_eeg_shape(self, x: torch.Tensor) -> None:
        """Validate the EEG tensor shape expected by ATMS."""
        if x.ndim != 3:
            raise ValueError(f"Expected EEG shape [batch, channels, time], got {tuple(x.shape)}.")
        if x.shape[1] != self.num_channels or x.shape[2] != self.sequence_length:
            raise ValueError(
                "Expected EEG shape "
                f"[batch, {self.num_channels}, {self.sequence_length}], got {tuple(x.shape)}."
            )

    def _subject_embedding(
        self,
        subject_ids: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Return a broadcastable subject embedding for 1-based subject ids."""
        if subject_ids.ndim != 1 or subject_ids.shape[0] != x.shape[0]:
            raise ValueError("subject_ids must be a 1D tensor with one id per EEG sample.")

        subject_indices = subject_ids.to(device=x.device, dtype=torch.long) - 1
        if torch.any(subject_indices < 0) or torch.any(subject_indices >= self.num_subjects):
            raise ValueError(f"subject_ids must be in the 1-based range [1, {self.num_subjects}].")

        return self.subject_embedding(subject_indices).unsqueeze(1)
