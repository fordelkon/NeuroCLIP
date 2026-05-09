"""Embedding layers and positional encoding helpers for transformer components."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.transformers.ffnets import MLP


def _check_positive_int(name: str, value: int) -> None:
    """Validate that a value is a positive integer."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def nerf_positional_embedding(coords: torch.Tensor, emb_size: int) -> torch.Tensor:
    """Encode coordinates with NeRF-style sinusoidal features."""
    if coords.ndim != 3:
        raise ValueError("coords must have shape [batch, count, dim]")
    _check_positive_int("emb_size", emb_size)

    batch_size, count, dim = coords.shape
    num_freqs = emb_size // (2 * dim)
    leftover = emb_size - num_freqs * 2 * dim
    freqs = 2.0 ** torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)
    freqs_scaled = coords.unsqueeze(-1) * freqs.view(1, 1, 1, -1)
    encoded = (
        torch.stack((freqs_scaled.sin(), freqs_scaled.cos()), dim=-1)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch_size, count, num_freqs * dim * 2)
    )

    if leftover > 0:
        pad = torch.zeros(
            batch_size,
            count,
            leftover,
            device=coords.device,
            dtype=coords.dtype,
        )
        encoded = torch.cat((encoded, pad), dim=-1)
    return encoded


class TDTEmbedding(nn.Module):
    """Build multiscale image tokens with depthwise smoothing diffusion."""

    def __init__(
        self,
        num_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        kernel_size: int = 3,
        share_across_steps: bool = True,
        *,
        K: int | None = None,
        H: int | None = None,
        W: int | None = None,
        ksize: int | None = None,
    ) -> None:
        super().__init__()
        self.num_steps = K if K is not None else num_steps
        self.height = H if H is not None else height
        self.width = W if W is not None else width
        self.kernel_size = ksize if ksize is not None else kernel_size
        self.share_across_steps = share_across_steps

        if self.num_steps is None:
            raise ValueError("num_steps must be provided")
        _check_positive_int("num_steps", self.num_steps)
        _check_positive_int("kernel_size", self.kernel_size)
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.pad = self.kernel_size // 2
        if self.num_steps > 1:
            num_kernels = 1 if self.share_across_steps else self.num_steps - 1
            self.kernel_raw = nn.Parameter(
                torch.zeros(num_kernels, self.kernel_size, self.kernel_size)
            )
        else:
            self.register_parameter("kernel_raw", None)

    def _infer_grid(self, num_tokens: int) -> tuple[int, int]:
        """Infer the 2D token grid from explicit or square dimensions."""
        if self.height is not None and self.width is not None:
            if self.height * self.width != num_tokens:
                raise ValueError("height * width must match the number of input tokens")
            return self.height, self.width

        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError("num_tokens must be a perfect square when grid is omitted")
        return side, side

    def get_kernel_weights(self) -> torch.Tensor:
        """Return normalized smoothing kernels for each diffusion step."""
        if self.num_steps <= 1 or self.kernel_raw is None:
            return torch.empty((0, self.kernel_size, self.kernel_size))

        weights = torch.softmax(
            self.kernel_raw.view(-1, self.kernel_size * self.kernel_size),
            dim=1,
        ).view(-1, self.kernel_size, self.kernel_size)
        if self.share_across_steps:
            return weights.repeat(self.num_steps - 1, 1, 1)
        return weights

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        """Return tokens for the original and diffused spatial scales."""
        if p0.ndim != 3:
            raise ValueError("p0 must have shape [batch, tokens, dim]")

        batch_size, num_tokens, embed_dim = p0.shape
        height, width = self._infer_grid(num_tokens)
        feature_map = p0.transpose(1, 2).reshape(batch_size, embed_dim, height, width)
        feature_maps = [feature_map.contiguous()]

        for step in range(1, self.num_steps):
            if self.kernel_raw is None:
                break
            kernel_idx = 0 if self.share_across_steps else step - 1
            step_kernel = torch.softmax(self.kernel_raw[kernel_idx].flatten(), dim=0).view(
                1, 1, self.kernel_size, self.kernel_size
            )
            step_kernel = step_kernel.to(device=p0.device, dtype=p0.dtype)
            depthwise_kernel = step_kernel.expand(
                embed_dim, 1, self.kernel_size, self.kernel_size
            ).contiguous()
            prev_feature_map = feature_maps[-1]
            if prev_feature_map.device.type == "cpu" and prev_feature_map.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                next_feature_map = F.conv2d(
                    prev_feature_map.float(),
                    depthwise_kernel.float(),
                    padding=self.pad,
                    groups=embed_dim,
                ).to(prev_feature_map.dtype)
            else:
                next_feature_map = F.conv2d(
                    prev_feature_map,
                    depthwise_kernel,
                    padding=self.pad,
                    groups=embed_dim,
                )
            feature_maps.append(next_feature_map)

        tokens = torch.stack(feature_maps, dim=1)
        return tokens.reshape(batch_size, self.num_steps, embed_dim, num_tokens).transpose(2, 3)


class TemporalConv(nn.Module):
    """Temporal convolution block for EEG inputs."""

    def __init__(self, num_filters: int, kernel_size: int, pool_size: int, stride: int):
        super().__init__()
        self.tconv = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size=(1, kernel_size), stride=(1, 1)),
            nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, stride)),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal convolution."""
        return self.tconv(x)


class TemporalConvTranspose(nn.Module):
    """Approximate inverse temporal convolution block."""

    def __init__(self, num_filters: int, kernel_size: int, pool_size: int, stride: int):
        super().__init__()
        self.tconv_t = nn.Sequential(
            nn.Upsample(scale_factor=(1, stride), mode="nearest"),
            nn.ConvTranspose2d(
                num_filters,
                1,
                kernel_size=(1, kernel_size + pool_size),
                stride=(1, 1),
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal transposed convolution."""
        return self.tconv_t(x)


class SpatialConv(nn.Module):
    """Spatial convolution block across EEG channels."""

    def __init__(self, num_filters: int, channels: int, dropout: float):
        super().__init__()
        self.sconv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=(channels, 1)),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial convolution."""
        return self.sconv(x)


class SpatialConvTranspose(nn.Module):
    """Approximate inverse spatial convolution block."""

    def __init__(self, num_filters: int, channels: int, dropout: float):
        super().__init__()
        self.sconv_t = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=(channels, 1)),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial transposed convolution."""
        return self.sconv_t(x)


class TSEmbedding(nn.Module):
    """Traditional temporal and spatial convolution embedding for EEG."""

    def __init__(
        self,
        emb_size: int,
        k: int = 40,
        m1: int = 25,
        m2: int = 51,
        s: int = 5,
        channels: int | None = None,
        dropout: float | None = None,
        *,
        C: int | None = None,
        p: float | None = None,
    ) -> None:
        super().__init__()
        channels = C if C is not None else channels
        dropout = p if p is not None else dropout
        channels = 63 if channels is None else channels
        dropout = 0.1 if dropout is None else dropout

        self.tconv = TemporalConv(k, m1, m2, s)
        self.sconv = SpatialConv(k, channels, dropout)
        self.proj = nn.Conv2d(k, emb_size, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project EEG samples from [batch, channels, time] to token embeddings."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, channels, time]")
        x = x.unsqueeze(1)
        x = self.sconv(self.tconv(x))
        x = self.proj(x)
        return x.squeeze(-2).transpose(-1, -2)


class PatchEmbedding(nn.Module):
    """Split each EEG channel into time patches and project each patch."""

    def __init__(
        self,
        patch_size: int,
        emb_size: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        _check_positive_int("patch_size", patch_size)
        self.patch_size = patch_size
        self.use_layer_norm = use_layer_norm
        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=(1, patch_size))
        self.proj = MLP(patch_size, hidden_dim=4 * patch_size, out_dim=emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-channel patch embeddings with shape [batch, channels, patches, dim]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, channels, time]")
        batch_size, channels, _ = x.shape
        x = x.unsqueeze(1).transpose(1, 2).contiguous()
        x = self.unfold(x)
        x = x.transpose(-1, -2).reshape(batch_size, channels, -1, self.patch_size)
        if self.use_layer_norm:
            x = torch.layer_norm(x, [self.patch_size])
        return self.proj(x)


class FrequencyEmbedding(nn.Module):
    """Project FFT magnitude and phase features for each time patch."""

    def __init__(
        self,
        patch_size: int,
        emb_size: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        _check_positive_int("patch_size", patch_size)
        self.patch_size = patch_size
        self.use_layer_norm = use_layer_norm
        self.unfold = nn.Unfold(kernel_size=(1, patch_size), stride=(1, patch_size))
        self.in_dim = 2 * (patch_size // 2 + 1)
        self.proj = MLP(self.in_dim, hidden_dim=4 * self.in_dim, out_dim=emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return FFT patch embeddings with shape [batch, channels, patches, dim]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, channels, time]")
        batch_size, channels, _ = x.shape
        x = x.unsqueeze(1).transpose(1, 2).contiguous()
        x = self.unfold(x).reshape(batch_size, channels, -1, self.patch_size)
        freq = torch.fft.rfft(x, dim=-1)
        freq_features = torch.cat((freq.abs(), freq.angle()), dim=-1)
        if self.use_layer_norm:
            freq_features = torch.layer_norm(freq_features, [self.in_dim])
        return self.proj(freq_features)


def fft_for_periods(x: torch.Tensor, top_k: int | None = None, *, k: int | None = None):
    """Estimate dominant periods from the real FFT magnitude."""
    top_k = k if k is not None else top_k
    if top_k is None:
        raise ValueError("top_k must be provided")
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, channels, time]")

    freq = torch.fft.rfft(x, dim=-1)
    magnitude = freq.abs()
    mean_magnitude = magnitude.mean(dim=0).mean(dim=0)
    mean_magnitude[0] = 0
    patch_sizes = torch.topk(mean_magnitude, k=top_k).indices.detach().cpu()
    periods = x.shape[-1] // patch_sizes
    weights = magnitude.mean(dim=1)[:, patch_sizes.to(x.device)]
    return periods, weights


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding frequency generator."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        learned_freq: bool = False,
        interpolate_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if interpolate_factor < 1.0:
            raise ValueError("interpolate_factor must be at least 1.0")
        self.freqs = nn.Parameter(
            torch.exp(-torch.arange(0, dim - 1, 2) * (math.log(theta) / dim)),
            requires_grad=learned_freq,
        )
        self.interpolate_factor = interpolate_factor
        self.cache: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}

    def prepare_freqs(self, t: torch.Tensor, offset: float = 0.0) -> torch.Tensor:
        """Prepare expanded rotary frequencies for a channel-patch grid."""
        if t.ndim != 4:
            raise ValueError("t must have shape [batch, channels, patches, dim]")
        _, channels, patches, _ = t.shape
        cache_key = (channels, patches, t.device, t.dtype)
        if cache_key in self.cache:
            return self.cache[cache_key]

        seq_pos = torch.arange(channels, device=t.device, dtype=t.dtype).repeat_interleave(patches)
        seq_pos = (seq_pos + offset) / self.interpolate_factor
        freqs_scaled = torch.outer(
            seq_pos, self.freqs.to(device=t.device, dtype=t.dtype)
        ).repeat_interleave(2, dim=-1)
        self.cache[cache_key] = freqs_scaled
        return freqs_scaled


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swap pairs of features and negate the first feature of each pair."""
    x = x.reshape((*x.shape[:-1], x.shape[-1] // 2, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Apply rotary positional embedding to a contiguous feature span."""
    freqs = freqs.to(device=t.device, dtype=t.dtype)
    rotary_dim = freqs.shape[-1]
    end_index = start_index + rotary_dim
    if rotary_dim > t.shape[-1]:
        raise ValueError("rotary dimensions exceed the input feature dimension")

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    flat_middle = t_middle.reshape(-1, t_middle.shape[-1])
    repeated_freqs = freqs.repeat(t_middle.shape[0], 1)
    rotated = flat_middle * repeated_freqs.cos() * scale
    rotated = rotated + rotate_half(flat_middle) * repeated_freqs.sin() * scale
    t_rotated_middle = rotated.reshape_as(t_middle)
    return torch.cat((t_left, t_rotated_middle, t_right), dim=-1)


class AbsoluteEmbedding(nn.Module):
    """Absolute sinusoidal positional embedding frequency generator."""

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        learned_freq: bool = False,
        interpolate_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if interpolate_factor < 1.0:
            raise ValueError("interpolate_factor must be at least 1.0")
        self.freqs = nn.Parameter(
            torch.exp(-torch.arange(0, dim - 1, 2) * (math.log(theta) / dim)),
            requires_grad=learned_freq,
        )
        self.interpolate_factor = interpolate_factor

    def prepare_freqs(self, t: torch.Tensor, offset: float = 0.0) -> torch.Tensor:
        """Prepare absolute embedding frequencies for a channel-patch grid."""
        if t.ndim != 4:
            raise ValueError("t must have shape [batch, channels, patches, dim]")
        _, channels, patches, _ = t.shape
        seq_pos = torch.arange(channels, device=t.device, dtype=t.dtype).repeat_interleave(patches)
        seq_pos = (seq_pos + offset) / self.interpolate_factor
        return torch.outer(seq_pos, self.freqs.to(device=t.device, dtype=t.dtype))


def apply_absolute_embedding(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
) -> torch.Tensor:
    """Add absolute sinusoidal embedding to a contiguous feature span."""
    freqs = freqs.to(device=t.device, dtype=t.dtype)
    add_dim = freqs.shape[-1] * 2
    end_index = start_index + add_dim
    if add_dim > t.shape[-1]:
        raise ValueError("absolute embedding dimensions exceed the input feature dimension")

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    grid_shape = t_middle.shape[1:-1]
    pos_embedding = torch.zeros_like(t_middle[0].reshape(-1, add_dim))
    pos_embedding[:, 0::2] = freqs.sin()
    pos_embedding[:, 1::2] = freqs.cos()
    pos_embedding = pos_embedding.reshape(*grid_shape, add_dim)
    return torch.cat((t_left, t_middle + pos_embedding, t_right), dim=-1)


class DataEmbedding(nn.Module):
    """Data embedding with circular 1D convolution."""

    def __init__(
        self,
        in_channels: int | None = None,
        emb_size: int | None = None,
        dropout: float = 0.1,
        *,
        c_in: int | None = None,
        d_model: int | None = None,
    ) -> None:
        super().__init__()
        in_channels = c_in if c_in is not None else in_channels
        emb_size = d_model if d_model is not None else emb_size
        if in_channels is None or emb_size is None:
            raise ValueError("in_channels and emb_size must be provided")
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=emb_size,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project inputs from [batch, time, channels] to [batch, time, dim]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, time, channels]")
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        return self.dropout(x)
