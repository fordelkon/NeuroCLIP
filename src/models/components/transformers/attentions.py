"""Attention layers for transformer components."""

import math
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.transformers.embeddings import (
    apply_absolute_embedding,
    apply_rotary_emb,
)


def _validate_attention_dims(dim: int, num_heads: int) -> None:
    """Validate the shared multi-head attention dimensions."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if dim % num_heads != 0:
        raise ValueError("dim must be divisible by num_heads")


def _prepare_attention_mask(
    mask: torch.Tensor | None,
    seq_len: int,
    device: torch.device,
    is_causal: bool,
) -> torch.Tensor | None:
    """Build a boolean mask where True means the key position is visible."""
    prepared = None
    if mask is not None:
        prepared = mask.to(device=device, dtype=torch.bool)
        if prepared.ndim == 2:
            prepared = prepared.unsqueeze(0).unsqueeze(0)
        elif prepared.ndim == 3:
            prepared = prepared.unsqueeze(1)
        elif prepared.ndim != 4:
            raise ValueError("mask must have 2, 3, or 4 dimensions")

    if is_causal:
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        causal = causal.unsqueeze(0).unsqueeze(0)
        prepared = causal if prepared is None else prepared & causal
    return prepared


def _attention_bias(mask: torch.Tensor | None, scores: torch.Tensor) -> torch.Tensor | int:
    """Convert a visibility mask to an additive attention bias."""
    if mask is None:
        return 0
    return torch.zeros_like(scores).masked_fill(~mask, -torch.inf)


def _attention_weights(
    scores: torch.Tensor,
    mask: torch.Tensor | None,
    dropout: nn.Dropout | None = None,
) -> torch.Tensor:
    """Normalize attention scores and optionally apply dropout."""
    weights = torch.softmax(scores + _attention_bias(mask, scores), dim=-1)
    return dropout(weights) if dropout is not None else weights


def _apply_rotary_to_heads(freqs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a [batch, heads, tokens, dim] tensor."""
    batch_size, num_heads, seq_len, head_dim = x.shape
    x = x.reshape(batch_size * num_heads, seq_len, head_dim)
    x = apply_rotary_emb(freqs=freqs, t=x)
    return x.reshape(batch_size, num_heads, seq_len, head_dim)


class Attention(nn.Module):
    """Multi-head attention with optional absolute or rotary positional embedding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        is_causal: bool = False,
        pos_emb_method: Literal["absolute", "rotary", "none"] = "none",
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        _validate_attention_dims(dim, num_heads)
        if pos_emb_method not in {"absolute", "rotary", "none"}:
            raise ValueError("pos_emb_method must be 'absolute', 'rotary', or 'none'")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_dropout = attn_dropout
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(p=proj_dropout)

        self.is_causal = is_causal
        self.pos_emb_method = pos_emb_method
        self.return_attention = return_attention

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention to tokens with shape [batch, tokens, dim]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, tokens, dim]")
        if self.pos_emb_method in {"absolute", "rotary"} and freqs is None:
            raise ValueError("freqs must be provided for positional attention")

        batch_size, seq_len, dim = x.shape
        if self.pos_emb_method == "absolute":
            x = apply_absolute_embedding(freqs=freqs, t=x)

        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(dim=0)

        if self.pos_emb_method == "rotary":
            q = _apply_rotary_to_heads(freqs=freqs, x=q)
            k = _apply_rotary_to_heads(freqs=freqs, x=k)

        prepared_mask = _prepare_attention_mask(mask, seq_len, x.device, self.is_causal)
        if self.return_attention:
            scores = q @ k.transpose(-2, -1) * self.scale
            return _attention_weights(scores, prepared_mask)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=prepared_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj_dropout(self.proj(y))


@lru_cache(maxsize=128)
def get_rel_pos_matrix(seq_len: int, device: torch.device) -> torch.Tensor:
    """Return query-minus-key relative position offsets."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    q_ids = torch.arange(seq_len, device=device)
    k_ids = torch.arange(seq_len, device=device)
    return q_ids.view(-1, 1) - k_ids.view(1, -1)


@lru_cache(maxsize=128)
def make_log_bucket_dict(
    bucket_len: int,
    rel_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Map relative offsets to signed logarithmic buckets."""
    if bucket_len <= 1:
        raise ValueError("bucket_len must be greater than 1")
    if rel_len <= bucket_len:
        raise ValueError("rel_len must be greater than bucket_len")

    rel_pos = torch.arange(-rel_len, rel_len, device=device)
    sign = torch.sign(rel_pos)
    abs_pos = rel_pos.abs()
    mid = bucket_len // 2

    linear = abs_pos <= mid
    safe_abs = torch.clamp(abs_pos, min=mid + 1)
    log_scale = math.log((rel_len - 1) / mid)
    log_pos = torch.ceil(torch.log(safe_abs / mid) / log_scale * (bucket_len - mid - 1))
    log_pos = torch.clamp(log_pos + mid, max=bucket_len - 1)
    bucket_abs = torch.where(linear, abs_pos, log_pos.to(abs_pos.dtype))
    return (bucket_abs * sign).to(torch.long)


def make_log_bucket_pos(
    rel_pos_matrix: torch.Tensor,
    bucket_len: int,
    rel_len: int,
) -> torch.Tensor:
    """Convert relative offsets to signed logarithmic bucket positions."""
    rel_pos_matrix = torch.clamp(rel_pos_matrix, -rel_len, rel_len - 1) + rel_len
    bucket_dict = make_log_bucket_dict(bucket_len, rel_len, rel_pos_matrix.device)
    return bucket_dict[rel_pos_matrix.long()]


class DisentangledAttention(nn.Module):
    """Relative disentangled attention with content-position interaction terms."""

    def __init__(
        self,
        dim: int,
        rel_len: int,
        num_heads: int,
        bucket_len: int | None = None,
        qkv_bias: bool = True,
        qk2p_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        _validate_attention_dims(dim, num_heads)
        if rel_len <= 0:
            raise ValueError("rel_len must be positive")
        if bucket_len is not None and bucket_len <= 1:
            raise ValueError("bucket_len must be greater than 1")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rel_len = rel_len
        self.bucket_len = bucket_len
        self.num_rel_positions = 2 * (bucket_len if bucket_len is not None else rel_len)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qk2p = nn.Linear(dim, dim * 2, bias=qk2p_bias)
        self.proj = nn.Linear(dim, dim)
        self.rel_emb = nn.Embedding(self.num_rel_positions, dim)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)
        self.is_causal = is_causal
        self.return_attention = return_attention

    def get_rel_emb_table(self) -> torch.Tensor:
        """Return the learnable relative position embedding table."""
        return self.rel_emb.weight

    def _position_index(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build relative position indexes for the active sequence length."""
        rel_pos = get_rel_pos_matrix(seq_len, device)
        offset = self.rel_len
        if self.bucket_len is not None:
            rel_pos = make_log_bucket_pos(rel_pos, self.bucket_len, self.rel_len)
            offset = self.bucket_len
        return torch.clamp(rel_pos + offset, 0, self.num_rel_positions - 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply disentangled attention to tokens with shape [batch, tokens, dim]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, tokens, dim]")

        batch_size, seq_len, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(dim=0)

        rel_table = self.get_rel_emb_table()
        qk2p = (
            self.qk2p(rel_table)
            .reshape(self.num_rel_positions, 2, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
            .contiguous()
        )
        q2p, k2p = qk2p.unbind(dim=0)
        q2p = q2p.unsqueeze(0).expand(batch_size, -1, -1, -1)
        k2p = k2p.unsqueeze(0).expand(batch_size, -1, -1, -1)

        pos = self._position_index(seq_len, x.device)
        pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        c2c_attn = q @ k.transpose(-1, -2)
        c2p_attn = q @ k2p.transpose(-1, -2)
        c2p_attn = c2p_attn.gather(dim=-1, index=pos)
        p2c_attn = q2p @ k.transpose(-1, -2)
        p2c_attn = p2c_attn.gather(dim=-2, index=pos)

        scores = (c2c_attn + c2p_attn + p2c_attn) / math.sqrt(3 * self.head_dim)
        prepared_mask = _prepare_attention_mask(mask, seq_len, x.device, self.is_causal)
        attn_weight = _attention_weights(
            scores,
            prepared_mask,
            None if self.return_attention else self.attn_dropout,
        )
        if self.return_attention:
            return attn_weight

        y = attn_weight @ v
        y = y.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj_dropout(self.proj(y))


class RelativeXLAttention(nn.Module):
    """Transformer-XL style relative attention with learnable position biases."""

    def __init__(
        self,
        dim: int,
        rel_len: int,
        num_heads: int,
        bucket_len: int | None = None,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        is_causal: bool = False,
        return_attention: bool = False,
    ) -> None:
        super().__init__()
        _validate_attention_dims(dim, num_heads)
        if rel_len <= 0:
            raise ValueError("rel_len must be positive")
        if bucket_len is not None and bucket_len <= 1:
            raise ValueError("bucket_len must be greater than 1")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rel_len = rel_len
        self.bucket_len = bucket_len
        self.num_rel_positions = 2 * (bucket_len if bucket_len is not None else rel_len)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.rel_emb = nn.Embedding(self.num_rel_positions, dim)
        self.rel_bias_emb = nn.Embedding(self.num_rel_positions, num_heads)
        self.abs_bias = nn.Embedding(num_heads, self.head_dim)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)
        self.is_causal = is_causal
        self.return_attention = return_attention

    def get_rel_emb_table(self) -> torch.Tensor:
        """Return relative key-position embeddings."""
        return self.rel_emb.weight

    def get_rel_bias_emb_table(self) -> torch.Tensor:
        """Return relative position-position bias embeddings."""
        return self.rel_bias_emb.weight

    def get_abs_bias(self) -> torch.Tensor:
        """Return head-specific query content bias."""
        return self.abs_bias.weight

    def _position_index(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build relative position indexes for the active sequence length."""
        rel_pos = get_rel_pos_matrix(seq_len, device)
        offset = self.rel_len
        if self.bucket_len is not None:
            rel_pos = make_log_bucket_pos(rel_pos, self.bucket_len, self.rel_len)
            offset = self.bucket_len
        return torch.clamp(rel_pos + offset, 0, self.num_rel_positions - 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply Transformer-XL relative attention to [batch, tokens, dim]."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, tokens, dim]")

        batch_size, seq_len, dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv.unbind(dim=0)

        rel_key = (
            self.get_rel_emb_table()
            .reshape(self.num_rel_positions, self.num_heads, self.head_dim)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
        )
        rel_bias = (
            self.get_rel_bias_emb_table()
            .transpose(0, 1)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, -1, seq_len, -1)
        )
        query_bias = self.get_abs_bias().unsqueeze(0).unsqueeze(2)

        pos = self._position_index(seq_len, x.device)
        pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        c2c_attn = q @ k.transpose(-1, -2)
        c2p_attn = q @ rel_key.transpose(-1, -2)
        c2p_attn = c2p_attn.gather(dim=-1, index=pos)
        p2c_attn = query_bias @ k.transpose(-1, -2)
        p2c_attn = p2c_attn.expand(-1, -1, seq_len, -1)
        p2p_attn = rel_bias.gather(dim=-1, index=pos)

        scores = (c2c_attn + c2p_attn + p2c_attn + p2p_attn) / math.sqrt(4 * self.head_dim)
        prepared_mask = _prepare_attention_mask(mask, seq_len, x.device, self.is_causal)
        attn_weight = _attention_weights(
            scores,
            prepared_mask,
            None if self.return_attention else self.attn_dropout,
        )
        if self.return_attention:
            return attn_weight

        y = attn_weight @ v
        y = y.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj_dropout(self.proj(y))
