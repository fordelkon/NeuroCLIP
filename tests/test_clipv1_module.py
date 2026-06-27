import inspect
from types import SimpleNamespace

import numpy as np
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


class DummyUBPDataset:
    def __init__(self, view_names: list[str] | None, length: int = 4) -> None:
        self.view_names = view_names
        self.match_label = np.ones(length, dtype=np.int32) if view_names is not None else None
        self.updated_indices: np.ndarray | None = None
        self.updated_labels: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.match_label) if self.match_label is not None else 4

    def update_match_labels(self, indices: np.ndarray, labels: np.ndarray) -> None:
        self.updated_indices = indices
        self.updated_labels = labels
        self.match_label[indices] = labels


def _attach_fake_trainer(
    module: ClipV1LitModule,
    *,
    train_dataset: object | None = None,
    test_dataset: object | None = None,
    current_epoch: int = 0,
) -> None:
    train_dataloader = (
        SimpleNamespace(dataset=train_dataset) if train_dataset is not None else None
    )
    datamodule = SimpleNamespace(data_test=test_dataset) if test_dataset is not None else None
    module._trainer = SimpleNamespace(
        train_dataloader=train_dataloader,
        datamodule=datamodule,
        world_size=1,
        is_global_zero=True,
        current_epoch=current_epoch,
    )


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


def test_on_train_start_auto_enables_ubp_for_multiview_dataset():
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        enable_ubp=False,
    )
    dataset = DummyUBPDataset(["mid_blur", "no_blur", "heavy_blur"], length=6)
    _attach_fake_trainer(module, train_dataset=dataset)

    module.on_train_start()

    assert module.enable_ubp is True
    assert torch.equal(module.ubp_sim, torch.zeros(6))


def test_on_train_start_disables_ubp_without_multiview_dataset():
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        enable_ubp=True,
    )
    dataset = DummyUBPDataset(None)
    _attach_fake_trainer(module, train_dataset=dataset)

    module.on_train_start()

    assert module.enable_ubp is False
    assert not hasattr(module, "ubp_sim")


def test_training_step_collects_ubp_confidences(monkeypatch):
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        enable_ubp=True,
        loss_type="cliploss",
    )
    module.ubp_indices = []
    module.ubp_confidences = []
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    batch = {
        "idx": torch.tensor([2, 5]),
        "eeg": torch.eye(4)[:2],
        "image_features": torch.nn.functional.normalize(torch.eye(4)[:2], dim=-1),
        "text_features": torch.nn.functional.normalize(
            torch.flip(torch.eye(4), dims=[0])[:2], dim=-1
        ),
        "label": torch.arange(2),
    }

    loss = module.training_step(batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.equal(module.ubp_indices[0], torch.tensor([2, 5]))
    assert torch.allclose(module.ubp_confidences[0], torch.ones(2))


def test_on_train_epoch_end_updates_dataset_match_labels_from_confidence():
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        enable_ubp=True,
        ubp_gamma=0.5,
        ubp_ci_alpha=0.5,
    )
    dataset = DummyUBPDataset(["mid_blur", "no_blur", "heavy_blur"], length=3)
    _attach_fake_trainer(module, train_dataset=dataset, current_epoch=0)
    module.ubp_sim = torch.zeros(3)
    module.ubp_indices = [torch.tensor([0, 1, 2])]
    module.ubp_confidences = [torch.tensor([-1.0, 0.0, 1.0])]

    module.on_train_epoch_end()

    assert dataset.updated_indices.tolist() == [0, 1, 2]
    assert dataset.updated_labels.tolist() == [2, 1, 0]
    assert dataset.match_label.tolist() == [2, 1, 0]
    assert torch.allclose(module.ubp_sim, torch.tensor([-0.5, 0.0, 0.5]))
    assert module.ubp_indices == []
    assert module.ubp_confidences == []


def test_on_test_start_sets_test_dataset_to_no_blur_view():
    module = ClipV1LitModule(
        eegnet=DummyEEGNet(),
        optimizer=torch.optim.SGD,
        scheduler=None,
        compile=False,
        enable_ubp=True,
    )
    dataset = DummyUBPDataset(["mid_blur", "no_blur", "heavy_blur"], length=4)
    dataset.match_label[:] = 2
    _attach_fake_trainer(module, test_dataset=dataset)

    module.on_test_start()

    assert dataset.match_label.tolist() == [1, 1, 1, 1]


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
