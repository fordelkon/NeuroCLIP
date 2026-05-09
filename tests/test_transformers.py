import pytest
import torch

from src.models.components.transformers.attentions import (
    Attention,
    DisentangledAttention,
    RelativeXLAttention,
    get_rel_pos_matrix,
    make_log_bucket_pos,
)
from src.models.components.transformers.embeddings import (
    AbsoluteEmbedding,
    DataEmbedding,
    FrequencyEmbedding,
    PatchEmbedding,
    RotaryEmbedding,
    TDTEmbedding,
    TSEmbedding,
    apply_absolute_embedding,
    apply_rotary_emb,
    fft_for_periods,
    nerf_positional_embedding,
)
from src.models.components.transformers.ffnets import MLP, FFNet


def test_attention_projects_tokens_without_positional_embedding() -> None:
    x = torch.randn(2, 5, 12)
    attention = Attention(dim=12, num_heads=3)

    output = attention(x)

    assert output.shape == x.shape


def test_attention_returns_attention_weights_when_requested() -> None:
    x = torch.randn(2, 5, 12)
    attention = Attention(dim=12, num_heads=3, return_attention=True)

    weights = attention(x)

    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_attention_applies_rotary_embedding() -> None:
    x = torch.randn(2, 5, 12)
    grid = torch.randn(2, 5, 1, 4)
    freqs = RotaryEmbedding(dim=4).prepare_freqs(grid)
    attention = Attention(dim=12, num_heads=3, pos_emb_method="rotary")

    output = attention(x, freqs=freqs)

    assert output.shape == x.shape


def test_attention_rejects_dimensions_not_divisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        Attention(dim=10, num_heads=3)


def test_relative_position_matrix_contains_query_minus_key_offsets() -> None:
    matrix = get_rel_pos_matrix(seq_len=4, device=torch.device("cpu"))

    assert matrix.tolist() == [
        [0, -1, -2, -3],
        [1, 0, -1, -2],
        [2, 1, 0, -1],
        [3, 2, 1, 0],
    ]


def test_log_bucket_positions_fit_requested_bucket_range() -> None:
    rel_pos = get_rel_pos_matrix(seq_len=6, device=torch.device("cpu"))

    bucket_pos = make_log_bucket_pos(rel_pos, bucket_len=4, rel_len=6)

    assert bucket_pos.shape == rel_pos.shape
    assert bucket_pos.min() >= -4
    assert bucket_pos.max() <= 3


def test_disentangled_attention_projects_tokens() -> None:
    x = torch.randn(2, 5, 12)
    attention = DisentangledAttention(dim=12, rel_len=5, num_heads=3)

    output = attention(x)

    assert output.shape == x.shape


def test_disentangled_attention_returns_weights() -> None:
    x = torch.randn(2, 5, 12)
    attention = DisentangledAttention(dim=12, rel_len=5, num_heads=3, return_attention=True)

    weights = attention(x)

    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_relative_xl_attention_projects_tokens() -> None:
    x = torch.randn(2, 5, 12)
    attention = RelativeXLAttention(dim=12, rel_len=5, num_heads=3)

    output = attention(x)

    assert output.shape == x.shape


def test_relative_xl_attention_returns_weights() -> None:
    x = torch.randn(2, 5, 12)
    attention = RelativeXLAttention(dim=12, rel_len=5, num_heads=3, return_attention=True)

    weights = attention(x)

    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_nerf_positional_embedding_returns_requested_width() -> None:
    coords = torch.randn(2, 5, 3)

    output = nerf_positional_embedding(coords, emb_size=20)

    assert output.shape == (2, 5, 20)


def test_tdt_embedding_returns_multiscale_tokens() -> None:
    tokens = torch.randn(2, 4, 8)
    embedding = TDTEmbedding(num_steps=3, height=2, width=2)

    output = embedding(tokens)

    assert output.shape == (2, 3, 4, 8)


def test_tdt_embedding_rejects_non_square_tokens_without_grid() -> None:
    embedding = TDTEmbedding(num_steps=2)

    with pytest.raises(ValueError, match="perfect square"):
        embedding(torch.randn(2, 6, 8))


def test_temporal_spatial_embedding_projects_eeg_to_tokens() -> None:
    x = torch.randn(2, 4, 32)
    embedding = TSEmbedding(emb_size=8, k=3, m1=3, m2=3, s=2, channels=4, dropout=0.0)

    output = embedding(x)

    assert output.shape == (2, 14, 8)


def test_patch_embedding_projects_time_patches() -> None:
    x = torch.randn(2, 3, 16)
    embedding = PatchEmbedding(patch_size=4, emb_size=6)

    output = embedding(x)

    assert output.shape == (2, 3, 4, 6)


def test_frequency_embedding_projects_frequency_features() -> None:
    x = torch.randn(2, 3, 16)
    embedding = FrequencyEmbedding(patch_size=4, emb_size=6)

    output = embedding(x)

    assert output.shape == (2, 3, 4, 6)


def test_fft_for_periods_returns_top_periods_and_weights() -> None:
    x = torch.randn(2, 3, 16)

    periods, weights = fft_for_periods(x, top_k=2)

    assert periods.shape == (2,)
    assert weights.shape == (2, 2)


def test_rotary_embedding_applies_to_selected_feature_span() -> None:
    x = torch.randn(2, 3, 4, 8)
    rotary = RotaryEmbedding(dim=4)
    freqs = rotary.prepare_freqs(x)

    output = apply_rotary_emb(freqs=freqs, t=x, start_index=2)

    assert output.shape == x.shape
    assert torch.equal(output[..., :2], x[..., :2])


def test_absolute_embedding_applies_to_selected_feature_span() -> None:
    x = torch.randn(2, 3, 4, 8)
    absolute = AbsoluteEmbedding(dim=4)
    freqs = absolute.prepare_freqs(x)

    output = apply_absolute_embedding(freqs=freqs, t=x, start_index=2)

    assert output.shape == x.shape
    assert torch.equal(output[..., :2], x[..., :2])


def test_data_embedding_projects_feature_channels() -> None:
    x = torch.randn(2, 10, 3)
    embedding = DataEmbedding(in_channels=3, emb_size=8, dropout=0.0)

    output = embedding(x)

    assert output.shape == (2, 10, 8)


def test_mlp_preserves_leading_dimensions_and_sets_output_width() -> None:
    x = torch.randn(2, 3, 4)
    mlp = MLP(in_dim=4, hidden_dim=8, out_dim=6, dropout=0.0)

    output = mlp(x)

    assert output.shape == (2, 3, 6)


def test_ffnet_preserves_input_shape() -> None:
    x = torch.randn(2, 3, 4)
    ffnet = FFNet(dim=4, expansion=2, dropout=0.0)

    output = ffnet(x)

    assert output.shape == x.shape


def test_ffnet_rejects_non_positive_expansion() -> None:
    with pytest.raises(ValueError, match="expansion must be positive"):
        FFNet(dim=4, expansion=0)
