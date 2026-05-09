"""Lightning module for CLIP-style EEG feature alignment."""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from lightning import LightningModule
from open_clip.loss import ClipLoss, SigLipLoss
from open_clip_train.train import get_clip_metrics as open_clip_get_clip_metrics
from torch import nn
from torchmetrics import MaxMetric, MeanMetric

from src.metrics.retrieval import get_kway_metrics
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ClipV1LitModule(LightningModule):
    """Train an EEG encoder against offline CLIP image or text features."""

    def __init__(
        self,
        eegnet: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        compile: bool,
        retrieval_k_list: list[int] | None = None,
        modality: Literal["eeg2img", "eeg2txt", "eeg2img2txt"] = "eeg2img",
        loss_type: Literal["cliploss", "sigliploss"] = "sigliploss",
        alpha: float = 0.99,
    ) -> None:
        """Initialize the CLIP alignment module."""
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["eegnet"])

        self.eegnet = eegnet
        self.retrieval_k_list = retrieval_k_list or [2, 4, 10, 200]
        self.modality = modality
        self.alpha = alpha
        self.criterion = ClipLoss() if loss_type == "cliploss" else SigLipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_top1_acc_best = MaxMetric()
        self.val_top5_acc_best = MaxMetric()
        self.val_top10_acc_best = MaxMetric()

    def forward(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized EEG CLIP features."""
        eeg_features = self.eegnet(eeg_data)["eeg_clip"]
        return F.normalize(eeg_features, p=2, dim=-1)

    @staticmethod
    def _get_batch_feature(batch: dict[str, Any], primary: str, fallback: str) -> torch.Tensor:
        """Read a feature tensor while supporting dataset and legacy key names."""
        if primary in batch:
            return batch[primary]
        if fallback in batch:
            return batch[fallback]
        raise KeyError(f"Batch is missing {primary!r} or {fallback!r}.")

    def model_step(self, batch: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        """Compute loss and return features used by epoch-level metrics."""
        eeg_data = batch["eeg"]
        image_features = F.normalize(
            self._get_batch_feature(batch, "image_features", "img_features"), dim=-1
        )
        text_features = F.normalize(
            self._get_batch_feature(batch, "text_features", "txt_features"), dim=-1
        )
        labels = batch.get("label")

        eeg_features = self.forward(eeg_data)
        logit_scale = self.logit_scale.exp()

        if self.modality == "eeg2img":
            loss = self._contrastive_loss(eeg_features, image_features, logit_scale)
        elif self.modality == "eeg2txt":
            loss = self._contrastive_loss(eeg_features, text_features, logit_scale)
        else:
            image_loss = self._contrastive_loss(eeg_features, image_features, logit_scale)
            text_loss = self._contrastive_loss(eeg_features, text_features, logit_scale)
            loss = image_loss * self.alpha + text_loss * (1 - self.alpha)

        return loss, eeg_features, image_features, text_features, labels

    def _contrastive_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Call the selected OpenCLIP loss with its expected argument shape."""
        if isinstance(self.criterion, SigLipLoss):
            return self.criterion(source_features, target_features, logit_scale, None)
        return self.criterion(source_features, target_features, logit_scale)

    def on_train_start(self) -> None:
        """Reset validation trackers before Lightning's first sanity check."""
        self.val_loss.reset()
        self.val_top1_acc_best.reset()
        self.val_top5_acc_best.reset()
        self.val_top10_acc_best.reset()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one optimization step."""
        loss, eeg_features, *_ = self.model_step(batch)
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=eeg_features.shape[0],
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        """Prepare validation feature buffers."""
        self.val_loss.reset()
        self.all_eeg_features_val: list[torch.Tensor] = []
        self.all_img_features_val: list[torch.Tensor] = []

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Run one validation step."""
        loss, eeg_features, image_features, *_ = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, sync_dist=True)

        self.all_eeg_features_val.append(eeg_features.detach())
        self.all_img_features_val.append(image_features.detach())

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation retrieval metrics."""
        eeg_features = self._gather_feature_buffer(self.all_eeg_features_val)
        image_features = self._gather_feature_buffer(self.all_img_features_val)
        metrics = self._compute_clip_metrics(eeg_features, image_features)
        self._log_val_recall(metrics)
        self.all_eeg_features_val = []
        self.all_img_features_val = []

    def on_test_epoch_start(self) -> None:
        """Prepare test feature buffers."""
        self.test_loss.reset()
        self.all_eeg_features_test: list[torch.Tensor] = []
        self.all_img_features_test: list[torch.Tensor] = []

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Run one test step."""
        loss, eeg_features, image_features, *_ = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, sync_dist=True)

        self.all_eeg_features_test.append(eeg_features.detach().cpu())
        self.all_img_features_test.append(image_features.detach().cpu())

    def on_test_epoch_end(self) -> None:
        """Compute and log test retrieval metrics."""
        eeg_features = self._gather_feature_buffer(self.all_eeg_features_test)
        image_features = self._gather_feature_buffer(self.all_img_features_test)
        labels = torch.arange(eeg_features.shape[0], device=eeg_features.device).unsqueeze(0)
        for retrieval_k in self.retrieval_k_list:
            metrics = get_kway_metrics(
                eeg_features.unsqueeze(0),
                labels,
                image_features,
                self.logit_scale.exp(),
                k=retrieval_k,
            )
            self._log_test_kway(retrieval_k, metrics)
        self.all_eeg_features_test = []
        self.all_img_features_test = []

    def setup(self, stage: str) -> None:
        """Optionally compile the EEG network for training."""
        if self.hparams.compile and stage == "fit":
            self.eegnet = torch.compile(self.eegnet)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and optional scheduler."""
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _gather_feature_buffer(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate local features and gather them across distributed ranks."""
        local_features = torch.cat(features, dim=0).to(self.device)
        if getattr(self.trainer, "world_size", 1) <= 1:
            return local_features
        gathered = self.all_gather(local_features)
        return gathered.reshape(-1, local_features.shape[-1])

    def _compute_clip_metrics(
        self,
        eeg_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute validation retrieval metrics with OpenCLIP's metric helper."""
        open_clip_metrics = open_clip_get_clip_metrics(
            eeg_features,
            image_features,
            self.logit_scale.exp(),
        )
        return {
            "eeg_to_img_R@1": torch.as_tensor(
                open_clip_metrics["image_to_text_R@1"],
                device=self.device,
            ),
            "eeg_to_img_R@5": torch.as_tensor(
                open_clip_metrics["image_to_text_R@5"],
                device=self.device,
            ),
            "eeg_to_img_R@10": torch.as_tensor(
                open_clip_metrics["image_to_text_R@10"],
                device=self.device,
            ),
        }

    def _log_val_recall(self, metrics: dict[str, torch.Tensor]) -> None:
        """Log validation recall metrics and best-so-far trackers."""
        top1 = metrics["eeg_to_img_R@1"]
        top5 = metrics["eeg_to_img_R@5"]
        top10 = metrics["eeg_to_img_R@10"]

        self.val_top1_acc_best(top1)
        self.val_top5_acc_best(top5)
        self.val_top10_acc_best(top10)

        self.log("val/top1_acc", top1, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val/top5_acc", top5, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/top10_acc", top10, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val/top1_acc_best",
            self.val_top1_acc_best.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "val/top5_acc_best",
            self.val_top5_acc_best.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val/top10_acc_best",
            self.val_top10_acc_best.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def _log_test_kway(self, retrieval_k: int, metrics: dict[str, torch.Tensor]) -> None:
        """Log k-way retrieval metrics."""
        self.log(
            f"test/top1_acc_retrieval_{retrieval_k}",
            metrics["top1_acc"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        if "top5_acc" in metrics:
            self.log(
                f"test/top5_acc_retrieval_{retrieval_k}",
                metrics["top5_acc"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
