"""Graph-based feature smoothing modules for knowledge graph integration."""

import torch
import torch.nn.functional as F
from torch import nn


class GraphAttentionSmoothing(nn.Module):
    """Graph attention network for dynamic neighbor aggregation.

    Uses multi-head attention to compute adaptive weights for each neighbor based on target
    features and KG similarity scores.

    Supports subject-specific score scaling for cross-subject experiments. When
    use_subject_specific=True, each subject learns an independent scale parameter to modulate KG
    neighbor importance.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_kg_scores: bool = True,
        num_subjects: int = 10,
        use_subject_specific: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.use_kg_scores = use_kg_scores
        self.use_subject_specific = use_subject_specific

        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if use_kg_scores:
            if use_subject_specific:
                self.score_scale = nn.Parameter(torch.ones(num_subjects))
            else:
                self.score_scale = nn.Parameter(torch.ones(1))

    def forward(
        self,
        target_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_scores: torch.Tensor | None = None,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply graph attention smoothing.

        Args:
            target_features: Target node features [B, D]
            neighbor_features: Neighbor node features [B, K, D]
            neighbor_scores: KG similarity scores [B, K], optional
            subject_ids: Subject IDs [B], 0-indexed, optional (required if use_subject_specific=True)

        Returns:
            Smoothed features [B, D]
        """
        B, K, D = neighbor_features.shape

        # Use KG scores as attention bias
        attn_mask = None
        if self.use_kg_scores and neighbor_scores is not None:
            # Get subject-specific or global scale
            if self.use_subject_specific:
                if subject_ids is None:
                    raise ValueError("subject_ids required when use_subject_specific=True")
                # Index per-subject scales with fallback for unseen subjects
                # Use mean scale for subjects outside training range
                max_subject_id = self.score_scale.size(0) - 1
                valid_mask = subject_ids <= max_subject_id
                scale = torch.zeros(B, device=subject_ids.device, dtype=self.score_scale.dtype)

                if valid_mask.any():
                    scale[valid_mask] = self.score_scale[subject_ids[valid_mask]]
                if (~valid_mask).any():
                    # Use mean of all learned scales for unseen subjects
                    scale[~valid_mask] = self.score_scale.mean()
            else:
                # Global scale: [1] -> broadcast to [B]
                scale = self.score_scale.expand(B)

            # Convert similarity scores to attention bias
            # Higher similarity -> higher attention weight
            attn_mask = scale.unsqueeze(1) * neighbor_scores  # [B, K]
            # Expand for multi-head attention: [B*n_heads, 1, K]
            attn_mask = attn_mask.unsqueeze(1)  # [B, 1, K]
            attn_mask = attn_mask.repeat_interleave(self.n_heads, dim=0)  # [B*n_heads, 1, K]

        # Multi-head attention: target as query, neighbors as key/value
        query = target_features.unsqueeze(1)  # [B, 1, D]
        smoothed, _ = self.attn(query, neighbor_features, neighbor_features, attn_mask=attn_mask)

        # Residual connection + layer norm
        output = self.norm(target_features + self.dropout(smoothed.squeeze(1)))

        return output
