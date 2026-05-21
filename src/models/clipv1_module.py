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
        enable_ubp: bool = False,
        ubp_update_freq: int = 1,
        ubp_gamma: float = 0.9,
        ubp_ci_alpha: float = 0.05,
        enable_kg_smooth: bool = False,
        kg_lambda: float = 0.2,
        kg_save_dir: str | None = None,
        kg_k: int = 10,
        kg_n_neighbors: int = 5,
    ) -> None:
        """Initialize the CLIP alignment module."""
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["eegnet"])

        self.eegnet = eegnet
        self.retrieval_k_list = retrieval_k_list or [2, 4, 10, 200]
        self.modality = modality
        self.alpha = alpha
        self.enable_ubp = enable_ubp
        self.ubp_update_freq = ubp_update_freq
        self.ubp_gamma = ubp_gamma
        self.ubp_ci_alpha = ubp_ci_alpha
        self.criterion = ClipLoss() if loss_type == "cliploss" else SigLipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self.kg = None
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_top1_acc_best = MaxMetric()
        self.val_top5_acc_best = MaxMetric()
        self.val_top10_acc_best = MaxMetric()

    def forward(
        self,
        eeg_data: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return L2-normalized EEG CLIP features."""
        if subject_ids is not None and getattr(self.eegnet, "use_subject_embedding", False):
            eeg_features = self.eegnet(eeg_data, subject_ids=subject_ids)["eeg_clip"]
        else:
            eeg_features = self.eegnet(eeg_data)["eeg_clip"]
        return F.normalize(eeg_features, p=2, dim=-1)

    def _load_kg(self, partition: str) -> None:
        """Load knowledge graph for the specified partition."""
        if not self.hparams.enable_kg_smooth:
            return
        if self.hparams.kg_save_dir is None:
            raise ValueError("kg_save_dir required when enable_kg_smooth=True")

        from pathlib import Path

        from src.data.components.knowledge_graph import KnowledgeGraph

        kg_path = (
            Path(self.hparams.kg_save_dir) / f"thingseeg2_kg_{partition}_k{self.hparams.kg_k}.pt"
        )
        self.kg = KnowledgeGraph()
        self.kg.load(kg_path)
        log.info(f"Loaded knowledge graph from {kg_path}")

    @staticmethod
    def _get_batch_feature(batch: dict[str, Any], primary: str, fallback: str) -> torch.Tensor:
        """Read a feature tensor while supporting dataset and legacy key names."""
        if primary in batch:
            return batch[primary]
        if fallback in batch:
            return batch[fallback]
        raise KeyError(f"Batch is missing {primary!r} or {fallback!r}.")

    def get_graph_smoothed_target(
        self,
        image_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute graph-smoothed CLIP targets.

        Supports two modes:
        - Open vocab KG: uses concept embeddings directly
        - Dataset-only KG: uses image-based neighbor sampling
        """
        smoothed = image_features.clone()

        # Open vocab mode: use concept embeddings
        if self.kg.concept_embeddings is not None:
            for i, concept_id in enumerate(labels):
                neighbor_embs = self.kg.get_concept_neighbor_embeddings(
                    concept_id.item(),
                    self.hparams.kg_n_neighbors,
                )
                if neighbor_embs is not None and len(neighbor_embs) > 0:
                    neighbor_avg = neighbor_embs.mean(dim=0).to(image_features.device)
                    smoothed[i] = (1 - self.hparams.kg_lambda) * image_features[
                        i
                    ] + self.hparams.kg_lambda * neighbor_avg

            return F.normalize(smoothed, dim=1)

        # Dataset-only mode: use image-based neighbors (backward compatibility)
        if self.kg.image_to_concept is None:
            return image_features

        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            dataset = self.trainer.datamodule.data_train
        elif self.trainer.train_dataloader is not None:
            dataset = self.trainer.train_dataloader.dataset
        else:
            return image_features

        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset

        for i, concept_id in enumerate(labels):
            neighbor_indices = self.kg.get_image_neighbors(
                concept_id.item(),
                n_same_concept=self.hparams.kg_n_neighbors,
                n_neighbor_concept=0,
            )
            if len(neighbor_indices) > 0:
                neighbor_features = torch.stack(
                    [dataset[n_idx]["image_features"] for n_idx in neighbor_indices]
                ).to(image_features.device)
                neighbor_avg = neighbor_features.mean(dim=0)
                smoothed[i] = (1 - self.hparams.kg_lambda) * image_features[
                    i
                ] + self.hparams.kg_lambda * neighbor_avg

        return F.normalize(smoothed, dim=1)

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

        eeg_features = self.forward(eeg_data, subject_ids=batch.get("subject_id"))
        logit_scale = self.logit_scale.exp()

        # Apply graph smoothing if enabled
        if self.hparams.enable_kg_smooth and self.kg is not None and labels is not None:
            image_features = self.get_graph_smoothed_target(image_features, labels)

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
        self._load_kg("training")

        self.val_loss.reset()
        self.val_top1_acc_best.reset()
        self.val_top5_acc_best.reset()
        self.val_top10_acc_best.reset()

        # Auto-detect multi-view features and enable UBP
        train_dataloader = self.trainer.train_dataloader
        if hasattr(train_dataloader, "dataset"):
            # Unwrap dataset to access the actual ThingsEEG2Dataset instance
            dataset = train_dataloader.dataset
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset

            has_multiview = hasattr(dataset, "view_names") and dataset.view_names is not None
            if has_multiview and len(dataset.view_names) > 1:
                if not self.enable_ubp:
                    log.info("Multi-view features detected, auto-enabling UBP")
                self.enable_ubp = True
            elif self.enable_ubp and not has_multiview:
                log.warning("UBP enabled but no multi-view features found, disabling UBP")
                self.enable_ubp = False

        log.info(f"UBP status: {'enabled' if self.enable_ubp else 'disabled'}")

        # Initialize UBP similarity tracking array
        if self.enable_ubp:
            n_samples = len(dataset)
            self.ubp_sim = torch.zeros(n_samples, dtype=torch.float32)
            log.info(f"Initialized UBP tracking for {n_samples} samples")

    def on_train_epoch_start(self) -> None:
        """Initialize UBP collection lists at epoch start."""
        if self.enable_ubp:
            self.ubp_indices = []
            self.ubp_confidences = []

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one optimization step."""
        loss, eeg_features, image_features, *_ = self.model_step(batch)
        self.train_loss(loss)

        # Track confidence for UBP
        if self.enable_ubp:
            with torch.no_grad():
                # Cosine similarity as confidence measure
                confidences = (eeg_features * image_features).sum(dim=-1)
                self.ubp_indices.append(batch["idx"])
                self.ubp_confidences.append(confidences)

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

    def on_train_epoch_end(self) -> None:
        """Update UBP match_labels based on confidence scores with EMA and CI thresholding."""
        if not self.enable_ubp:
            return

        # Only update every ubp_update_freq epochs
        if (self.current_epoch + 1) % self.ubp_update_freq != 0:
            return

        if not self.ubp_indices:
            return

        import numpy as np
        from scipy.stats import norm

        local_indices = torch.cat(self.ubp_indices, dim=0)
        local_confidences = torch.cat(self.ubp_confidences, dim=0)

        # Multi-GPU: gather confidences for global threshold computation
        world_size = self.trainer.world_size
        if world_size > 1:
            local_size = torch.tensor([local_confidences.size(0)], device=self.device)
            all_sizes = self.all_gather(local_size).view(-1)
            max_size = all_sizes.max().item()

            padded_conf = torch.zeros(max_size, dtype=local_confidences.dtype, device=self.device)
            padded_conf[: local_confidences.size(0)] = local_confidences.to(self.device)
            gathered_conf = self.all_gather(padded_conf).view(world_size, -1)

            all_conf_list = []
            for rank in range(world_size):
                valid_size = all_sizes[rank].item()
                all_conf_list.append(gathered_conf[rank, :valid_size])
            global_confidences = torch.cat(all_conf_list, dim=0).cpu().numpy()
        else:
            global_confidences = local_confidences.cpu().numpy()

        # Access underlying dataset
        dataset = self.trainer.train_dataloader.dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset

        # Compute global thresholds using confidence intervals
        mean_conf = np.mean(global_confidences)
        std_conf = np.std(global_confidences, ddof=1) if len(global_confidences) > 1 else 1e-6
        z_alpha_2 = norm.ppf(1 - self.ubp_ci_alpha / 2)
        lower_bound = mean_conf - z_alpha_2 * std_conf
        upper_bound = mean_conf + z_alpha_2 * std_conf

        # Process local samples with EMA smoothing
        local_indices_np = local_indices.cpu().numpy()
        local_conf_np = local_confidences.cpu().numpy()

        local_sim = (
            self.ubp_gamma * local_conf_np
            + (1 - self.ubp_gamma) * self.ubp_sim[local_indices_np].numpy()
        )
        self.ubp_sim[local_indices_np] = torch.from_numpy(local_sim)

        # Assign blur levels with index-based assignment
        new_labels = np.ones(len(local_indices_np), dtype=np.int32)
        new_labels[local_conf_np > upper_bound] = 0
        new_labels[local_conf_np < lower_bound] = 2

        dataset.update_match_labels(local_indices_np, new_labels)

        # Log statistics
        if self.trainer.is_global_zero:
            if world_size > 1:
                local_counts = torch.tensor(
                    [(new_labels == 0).sum(), (new_labels == 1).sum(), (new_labels == 2).sum()],
                    device=self.device,
                    dtype=torch.long,
                )
                global_counts = self.all_gather(local_counts).sum(dim=0).cpu().numpy()
                n_mid_blur, n_no_blur, n_heavy_blur = global_counts
            else:
                n_mid_blur, n_no_blur, n_heavy_blur = (
                    (new_labels == 0).sum(),
                    (new_labels == 1).sum(),
                    (new_labels == 2).sum(),
                )

            log.info(
                f"UBP epoch {self.current_epoch + 1}: "
                f"mid_blur={n_mid_blur}, no_blur={n_no_blur}, heavy_blur={n_heavy_blur}, "
                f"bounds=[{format(lower_bound, '.4f')}, {format(upper_bound, '.4f')}]"
            )
        elif world_size > 1:
            local_counts = torch.tensor(
                [(new_labels == 0).sum(), (new_labels == 1).sum(), (new_labels == 2).sum()],
                device=self.device,
                dtype=torch.long,
            )
            self.all_gather(local_counts)

        self.ubp_indices = []
        self.ubp_confidences = []

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

    def on_test_start(self) -> None:
        """Set all samples to use no_blur view for testing."""
        self._load_kg("test")

        if self.enable_ubp:
            # Access dataset through datamodule
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
                datamodule = self.trainer.datamodule
                if hasattr(datamodule, "data_test"):
                    dataset = datamodule.data_test
                    if hasattr(dataset, "match_label") and dataset.match_label is not None:
                        # Find no_blur view index
                        no_blur_idx = 0
                        if hasattr(dataset, "view_names") and dataset.view_names:
                            for idx, view_name in enumerate(dataset.view_names):
                                if "no_blur" in view_name.lower():
                                    no_blur_idx = idx
                                    break
                        dataset.match_label[:] = no_blur_idx
                        log.info(f"Test: using no_blur view (index {no_blur_idx})")

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
