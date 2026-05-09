import torch
from torch import nn


class MLP(nn.Module):
    """Two-layer feed-forward projection used by transformer blocks."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward projection to the input tensor."""
        return self.net(x)


class FFNet(nn.Module):
    """Position-wise transformer feed-forward network."""

    def __init__(self, dim: int, expansion: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")

        self.ffnet = MLP(
            in_dim=dim,
            hidden_dim=dim * expansion,
            out_dim=dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the transformer feed-forward block."""
        return self.ffnet(x)
