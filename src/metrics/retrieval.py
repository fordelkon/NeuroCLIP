"""Retrieval metrics for CLIP-style EEG alignment."""

import torch
import torch.nn.functional as F


def _as_scalar_scale(logit_scale: torch.Tensor | float) -> torch.Tensor | float:
    """Return a scale value that broadcasts cleanly with similarity matrices."""
    if torch.is_tensor(logit_scale) and logit_scale.ndim > 0:
        return logit_scale.squeeze()
    return logit_scale


def get_kway_metrics(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    candidate_features: torch.Tensor,
    logit_scale: torch.Tensor | float = 1.0,
    k: int = 200,
) -> dict[str, torch.Tensor]:
    """Compute k-way top-1 and top-5 retrieval accuracy.

    ``query_features`` has shape ``[groups, queries, dim]`` and ``query_labels``
    stores the candidate row index for each query.
    """
    if query_features.ndim != 3:
        raise ValueError("query_features must have shape [groups, queries, dim].")
    if query_labels.shape != query_features.shape[:2]:
        raise ValueError("query_labels must have shape [groups, queries].")
    if candidate_features.ndim != 2:
        raise ValueError("candidate_features must have shape [candidates, dim].")

    num_candidates = candidate_features.shape[0]
    retrieval_k = min(k, num_candidates)
    labels = query_labels.to(device=query_features.device, dtype=torch.long)
    if labels.min() < 0 or labels.max() >= num_candidates:
        raise ValueError("query_labels must contain valid candidate row indices.")

    query_features = F.normalize(query_features, dim=-1)
    candidate_features = F.normalize(candidate_features.to(query_features.device), dim=-1)

    if retrieval_k == num_candidates:
        candidate_indices = torch.arange(num_candidates, device=query_features.device)
        candidate_indices = candidate_indices.expand(*labels.shape, -1)
    else:
        offsets = torch.arange(retrieval_k - 1, device=query_features.device) + 1
        negative_indices = (labels.unsqueeze(-1) + offsets) % num_candidates
        candidate_indices = torch.cat([labels.unsqueeze(-1), negative_indices], dim=-1)

    selected_candidates = candidate_features[candidate_indices]
    scale = _as_scalar_scale(logit_scale)
    logits = scale * torch.einsum("gqd,gqkd->gqk", query_features, selected_candidates)
    target_positions = torch.where(candidate_indices == labels.unsqueeze(-1))[2]
    target_positions = target_positions.view_as(labels)

    topk = logits.topk(k=retrieval_k, dim=-1).indices

    top1_acc = topk[..., :1].eq(target_positions.unsqueeze(-1)).any(dim=-1).float().mean()
    metrics = {"top1_acc": top1_acc}
    if retrieval_k >= 5:
        metrics["top5_acc"] = (
            topk[..., :5].eq(target_positions.unsqueeze(-1)).any(dim=-1).float().mean()
        )
    return metrics
