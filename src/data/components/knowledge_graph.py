"""Knowledge graph for CLIP feature smoothing in EEG-to-CLIP alignment."""

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

StrPath = Union[str, Path]


class KnowledgeGraph:
    """Concept-level knowledge graph for semantic neighbor-based target smoothing."""

    def __init__(self):
        self.concept_to_images: dict[int, list[int]] = {}
        self.image_to_concept: np.ndarray | None = None
        self.concept_neighbors: torch.Tensor | None = None
        self.concept_neighbor_scores: torch.Tensor | None = None
        self.concepts: dict[int, dict] = {}
        self.concept_embeddings: torch.Tensor | None = None

    def build_concept_neighbors(
        self,
        clip_features: torch.Tensor,
        labels: torch.Tensor,
        k: int = 10,
    ) -> None:
        """Build concept-level neighbor graph from CLIP features.

        Args:
            clip_features: CLIP image embeddings [N, D]
            labels: Concept labels for each image [N]
            k: Number of neighbors per concept
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
            MofNCompleteColumn(),
            TextColumn("-"),
            TaskProgressColumn(),
            TextColumn("-"),
            TimeElapsedColumn(),
            TextColumn("-"),
            TimeRemainingColumn(),
            expand=False,
        )

        with progress:
            task = progress.add_task("[cyan]Building neighbor graph", total=4)

            # Group images by concept
            self.concept_to_images = {}
            for img_idx, concept_id in enumerate(labels.tolist()):
                self.concept_to_images.setdefault(concept_id, []).append(img_idx)
            progress.advance(task)

            # Compute concept embeddings (average pooling)
            n_concepts = len(self.concept_to_images)
            concept_embeddings = torch.zeros(n_concepts, clip_features.shape[1])

            for concept_id, img_indices in self.concept_to_images.items():
                concept_embeddings[concept_id] = clip_features[img_indices].mean(dim=0)
            progress.advance(task)

            # Compute concept similarity matrix
            concept_embeddings = F.normalize(concept_embeddings, dim=1)
            similarity = concept_embeddings @ concept_embeddings.T
            progress.advance(task)

            # Extract top-K neighbors (exclude self)
            similarity.fill_diagonal_(-1)
            self.concept_neighbor_scores, self.concept_neighbors = similarity.topk(k, dim=1)

            # Build image-to-concept mapping
            self.image_to_concept = np.zeros(len(labels), dtype=np.int64)
            for concept_id, img_indices in self.concept_to_images.items():
                self.image_to_concept[img_indices] = concept_id
            progress.advance(task)

    def build_from_hybrid_embeddings(
        self,
        dataset_concepts: list[str],
        dataset_embeddings: torch.Tensor,
        extended_concepts: list[str],
        extended_embeddings: torch.Tensor,
        k: int = 10,
        quota_dataset: int = 5,
        quota_extended: int = 5,
    ) -> None:
        """Build knowledge graph from hybrid embeddings (open vocabulary mode).

        Args:
            dataset_concepts: Concept names from dataset
            dataset_embeddings: CLIP image feature averages [N_dataset, D]
            extended_concepts: Extended concept names
            extended_embeddings: CLIP text embeddings [N_extended, D]
            k: Total neighbors per concept
            quota_dataset: Max neighbors from dataset
            quota_extended: Max neighbors from extended vocabulary
        """
        n_dataset = len(dataset_concepts)
        n_extended = len(extended_concepts)
        n_total = n_dataset + n_extended

        # Store concept metadata
        for i, name in enumerate(dataset_concepts):
            self.concepts[i] = {"name": name, "source": "thingseeg2", "emb_type": "image_avg"}

        for i, name in enumerate(extended_concepts):
            self.concepts[n_dataset + i] = {"name": name, "source": "wordnet", "emb_type": "text"}

        # Combine embeddings
        all_embeddings = torch.cat([dataset_embeddings, extended_embeddings], dim=0)
        all_embeddings = F.normalize(all_embeddings, dim=1)

        # Store embeddings for hybrid smoothing
        self.concept_embeddings = all_embeddings

        # Compute similarity matrix
        similarity = all_embeddings @ all_embeddings.T
        similarity.fill_diagonal_(-1)

        # Select neighbors with quota mechanism
        self.concept_neighbors = torch.zeros(n_total, k, dtype=torch.long)
        self.concept_neighbor_scores = torch.zeros(n_total, k)

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
            MofNCompleteColumn(),
            TextColumn("-"),
            TaskProgressColumn(),
            TextColumn("-"),
            TimeElapsedColumn(),
            TextColumn("-"),
            TimeRemainingColumn(),
            expand=False,
        )

        with progress:
            task = progress.add_task("[cyan]Building neighbor graph", total=n_total)
            for i in range(n_total):
                scores, indices = similarity[i].sort(descending=True)

                # Apply quota: separate dataset and extended neighbors
                dataset_neighbors = [idx for idx in indices if idx < n_dataset][:quota_dataset]
                extended_neighbors = [idx for idx in indices if idx >= n_dataset][:quota_extended]

                selected = (dataset_neighbors + extended_neighbors)[:k]
                selected_scores = [similarity[i, idx].item() for idx in selected]

                self.concept_neighbors[i, : len(selected)] = torch.tensor(selected)
                self.concept_neighbor_scores[i, : len(selected)] = torch.tensor(selected_scores)
                progress.advance(task)

    def get_image_neighbors(
        self,
        img_idx: int,
        n_same_concept: int = 5,
        n_neighbor_concept: int = 5,
        seed: int = 42,
    ) -> list[int]:
        """Get neighbor images for smoothing.

        Args:
            img_idx: Image index
            n_same_concept: Number of same-concept neighbors
            n_neighbor_concept: Number of neighbor-concept images
            seed: Random seed for deterministic sampling

        Returns:
            List of neighbor image indices
        """
        rng = np.random.RandomState(seed + img_idx)
        concept_id = self.image_to_concept[img_idx]
        neighbors = []

        # Same-concept neighbors
        same_concept_imgs = [i for i in self.concept_to_images[concept_id] if i != img_idx]
        if len(same_concept_imgs) > 0:
            n_sample = min(n_same_concept, len(same_concept_imgs))
            neighbors.extend(rng.choice(same_concept_imgs, n_sample, replace=False))

        # Neighbor-concept images
        neighbor_concepts = self.concept_neighbors[concept_id][:n_neighbor_concept]
        for neighbor_concept_id in neighbor_concepts:
            neighbor_imgs = self.concept_to_images[neighbor_concept_id.item()]
            if len(neighbor_imgs) > 0:
                neighbors.append(rng.choice(neighbor_imgs))

        return neighbors

    def get_concept_neighbor_embeddings(
        self,
        concept_id: int,
        n_neighbors: int = 5,
    ) -> torch.Tensor | None:
        """Get neighbor concept embeddings for hybrid smoothing.

        Args:
            concept_id: Concept ID
            n_neighbors: Number of neighbors to retrieve

        Returns:
            Neighbor embeddings [n_neighbors, D] or None if not available
        """
        if self.concept_embeddings is None or self.concept_neighbors is None:
            return None

        neighbor_ids = self.concept_neighbors[concept_id][:n_neighbors]
        return self.concept_embeddings[neighbor_ids]

    def save(self, path: StrPath) -> None:
        """Save knowledge graph (auto-detects format from extension)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".jsonl":
            self._save_jsonl(path)
        else:
            self._save_pt(path)

    def load(self, path: StrPath) -> None:
        """Load knowledge graph (auto-detects format from extension)."""
        path = Path(path)

        if path.suffix == ".jsonl":
            self._load_jsonl(path)
        else:
            self._load_pt(path)

    def _save_pt(self, path: Path) -> None:
        """Save as .pt format."""
        if self.concepts:
            torch.save(
                {
                    "concepts": self.concepts,
                    "concept_neighbors": self.concept_neighbors,
                    "concept_neighbor_scores": self.concept_neighbor_scores,
                    "concept_embeddings": self.concept_embeddings,
                },
                path,
            )
        else:
            torch.save(
                {
                    "concept_to_images": self.concept_to_images,
                    "image_to_concept": self.image_to_concept,
                    "concept_neighbors": self.concept_neighbors,
                    "concept_neighbor_scores": self.concept_neighbor_scores,
                },
                path,
            )

    def _save_jsonl(self, path: Path) -> None:
        """Save as JSONL format (supports both open vocabulary and dataset-only modes)."""
        with open(path, "w", encoding="utf-8") as f:
            if self.concepts:
                # Open vocabulary mode
                for concept_id, metadata in self.concepts.items():
                    neighbors = []
                    for j in range(self.concept_neighbors.shape[1]):
                        neighbor_id = self.concept_neighbors[concept_id, j].item()
                        score = self.concept_neighbor_scores[concept_id, j].item()
                        if score > 0:
                            neighbors.append({"id": neighbor_id, "score": round(score, 4)})

                    record = {
                        "id": concept_id,
                        "name": metadata["name"],
                        "source": metadata["source"],
                        "emb_type": metadata["emb_type"],
                        "neighbors": neighbors,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # Dataset-only mode
                for concept_id in range(self.concept_neighbors.shape[0]):
                    neighbors = []
                    for j in range(self.concept_neighbors.shape[1]):
                        neighbor_id = self.concept_neighbors[concept_id, j].item()
                        score = self.concept_neighbor_scores[concept_id, j].item()
                        if score > 0:
                            neighbors.append({"id": neighbor_id, "score": round(score, 4)})

                    record = {
                        "id": concept_id,
                        "n_images": len(self.concept_to_images[concept_id]),
                        "neighbors": neighbors,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_pt(self, path: Path) -> None:
        """Load from .pt format."""
        data = torch.load(path, map_location="cpu", weights_only=False)

        if "concepts" in data:
            self.concepts = data["concepts"]
            self.concept_neighbors = data["concept_neighbors"]
            self.concept_neighbor_scores = data["concept_neighbor_scores"]
            self.concept_embeddings = data.get("concept_embeddings")
        else:
            self.concept_to_images = data["concept_to_images"]
            self.image_to_concept = data["image_to_concept"]
            self.concept_neighbors = data["concept_neighbors"]
            self.concept_neighbor_scores = data["concept_neighbor_scores"]

    def _load_jsonl(self, path: Path) -> None:
        """Load from JSONL format."""
        self.concepts = {}
        neighbor_list = []
        score_list = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                concept_id = record["id"]
                self.concepts[concept_id] = {
                    "name": record["name"],
                    "source": record["source"],
                    "emb_type": record["emb_type"],
                }
                neighbor_list.append([n["id"] for n in record["neighbors"]])
                score_list.append([n["score"] for n in record["neighbors"]])

        max_k = max(len(n) for n in neighbor_list)
        n_concepts = len(self.concepts)
        self.concept_neighbors = torch.zeros(n_concepts, max_k, dtype=torch.long)
        self.concept_neighbor_scores = torch.zeros(n_concepts, max_k)

        for i, (neighbors, scores) in enumerate(zip(neighbor_list, score_list)):
            self.concept_neighbors[i, : len(neighbors)] = torch.tensor(neighbors)
            self.concept_neighbor_scores[i, : len(scores)] = torch.tensor(scores)
