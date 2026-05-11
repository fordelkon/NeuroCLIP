import inspect

import torch
from torch import nn

import src.models.clipv1_module as clipv1_module
from src.metrics.retrieval import get_kway_metrics
from src.models.clipv1_module import ClipV1LitModule


class DummyEEGNet(nn.Module):
    def __init__(self, in_dim: int = 4, out_dim: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"eeg_clip": self.proj(x)}


class DummySubjectAwareEEGNet(DummyEEGNet):
    use_subject_embedding = True

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if subject_ids is None:
            raise ValueError("subject_ids are required")
        return super().forward(x + subject_ids.float().unsqueeze(-1) * 0.0)


def test_forward_returns_l2_normalized_eeg_features():
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
    )

    features = module(torch.randn(3, 4))

    assert features.shape == (3, 4)
    assert torch.allclose(features.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_model_step_passes_subject_ids_to_subject_aware_eegnet():
    module = ClipV1LitModule(
        eegnet=DummySubjectAwareEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        modality="eeg2img",
    )
    batch = {
        "eeg": torch.eye(4),
        "subject_id": torch.tensor([1, 2, 3, 4]),
        "image_features": torch.nn.functional.normalize(torch.eye(4), dim=-1),
        "text_features": torch.nn.functional.normalize(torch.flip(torch.eye(4), dims=[0]), dim=-1),
        "label": torch.arange(4),
    }

    _, eeg_features, *_ = module.model_step(batch)

    assert eeg_features.shape == (4, 4)


def test_model_step_ignores_subject_ids_for_plain_eegnet():
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        modality="eeg2img",
    )
    batch = {
        "eeg": torch.eye(4),
        "subject_id": torch.tensor([1, 2, 3, 4]),
        "image_features": torch.nn.functional.normalize(torch.eye(4), dim=-1),
        "text_features": torch.nn.functional.normalize(torch.flip(torch.eye(4), dims=[0]), dim=-1),
        "label": torch.arange(4),
    }

    _, eeg_features, *_ = module.model_step(batch)

    assert eeg_features.shape == (4, 4)


def test_model_step_computes_loss_from_batch_features():
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        modality="eeg2img",
    )
    batch = {
        "eeg": torch.eye(4),
        "image_features": torch.nn.functional.normalize(torch.eye(4), dim=-1),
        "text_features": torch.nn.functional.normalize(torch.flip(torch.eye(4), dims=[0]), dim=-1),
        "label": torch.arange(4),
    }

    loss, eeg_features, image_features, text_features, labels = module.model_step(batch)

    assert loss.ndim == 0
    assert eeg_features.shape == image_features.shape == text_features.shape
    assert torch.equal(labels, torch.arange(4))


def test_clipv1_module_does_not_expose_local_metric_modes():
    signature = inspect.signature(ClipV1LitModule)

    assert "val_metrics" not in signature.parameters
    assert "test_metrics" not in signature.parameters


def test_validation_metrics_are_mapped_from_open_clip_train_metrics(monkeypatch):
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
    )

    def fake_open_clip_metrics(image_features, text_features, logit_scale):
        assert image_features.shape == text_features.shape == (4, 4)
        assert logit_scale.ndim == 0
        return {
            "image_to_text_R@1": 0.25,
            "image_to_text_R@5": 0.5,
            "image_to_text_R@10": 0.75,
            "text_to_image_R@1": 0.125,
            "text_to_image_R@5": 0.375,
            "text_to_image_R@10": 0.625,
        }

    monkeypatch.setattr(clipv1_module, "open_clip_get_clip_metrics", fake_open_clip_metrics)

    metrics = module._compute_clip_metrics(torch.eye(4), torch.eye(4))

    assert metrics["eeg_to_img_R@1"] == torch.tensor(0.25)
    assert metrics["eeg_to_img_R@5"] == torch.tensor(0.5)
    assert metrics["eeg_to_img_R@10"] == torch.tensor(0.75)


def test_kway_metrics_report_perfect_top1_for_aligned_features():
    features = torch.nn.functional.normalize(torch.eye(4), dim=-1)
    labels = torch.arange(4).unsqueeze(0)

    metrics = get_kway_metrics(
        query_features=features.unsqueeze(0),
        query_labels=labels,
        candidate_features=features,
        logit_scale=torch.tensor(10.0),
        k=4,
    )

    assert metrics["top1_acc"] == torch.tensor(1.0)


def test_kway_metrics_restrict_candidates_to_requested_k():
    query_features = torch.tensor([[[1.0, 0.0]]])
    candidate_features = torch.tensor(
        [
            [0.8, 0.6],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    labels = torch.tensor([[0]])

    two_way = get_kway_metrics(
        query_features=query_features,
        query_labels=labels,
        candidate_features=candidate_features,
        k=2,
    )
    four_way = get_kway_metrics(
        query_features=query_features,
        query_labels=labels,
        candidate_features=candidate_features,
        k=4,
    )

    assert two_way["top1_acc"] == torch.tensor(1.0)
    assert four_way["top1_acc"] == torch.tensor(0.0)
