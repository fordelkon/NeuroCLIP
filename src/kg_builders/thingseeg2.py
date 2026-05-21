"""Knowledge graph builder for THINGS-EEG2 dataset."""

from pathlib import Path
from typing import Union

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.data.components.knowledge_graph import KnowledgeGraph
from src.data.components.vocabulary_expander import WordNetExpander
from src.utils.config_resolvers import (
    create_clip_backend,
    resolve_clip_model_id,
    resolve_devices,
    sanitize_clip_model_name,
)

StrPath = Union[str, Path]
console = Console()


class Thingseeg2KGBuilder:
    """Build concept-level knowledge graph for THINGS-EEG2."""

    def __init__(
        self,
        k: int,
        clip_features_base_dir: StrPath,
        model_name: str,
        model_id: str | None,
        save_dir: StrPath,
        feature_mode: str,
        partition: str,
        seed: int,
        use_open_vocab: bool = False,
        max_extended_concepts: int = 3000,
        quota_dataset: int = 5,
        quota_extended: int = 5,
        output_format: str = "pt",
        model_cache_dir: StrPath | None = None,
        device: str = "auto",
    ):
        self.k = k
        self.clip_features_base_dir = Path(clip_features_base_dir)
        self.model_name, self.resolved_model_id = resolve_clip_model_id(model_name, model_id)
        self.save_dir = Path(save_dir)
        self.feature_mode = feature_mode
        self.partition = partition
        self.seed = seed
        self.use_open_vocab = use_open_vocab
        self.max_extended_concepts = max_extended_concepts
        self.quota_dataset = quota_dataset
        self.quota_extended = quota_extended
        self.output_format = output_format
        self.model_cache_dir = model_cache_dir
        self.device = device
        self.resolved_devices = resolve_devices(device)

    @property
    def clip_features_dir(self) -> Path:
        """Dynamically construct CLIP features directory path."""
        return (
            self.clip_features_base_dir
            / sanitize_clip_model_name(self.model_name)
            / sanitize_clip_model_name(self.resolved_model_id)
            / self.feature_mode
        )

    def run(self) -> dict[str, str | int]:
        """Build and save knowledge graph."""
        console.print()
        console.print(
            Panel.fit(
                "[bold white]THINGS-EEG2 Knowledge Graph Builder[/bold white]\n\n"
                "[dim]Build concept-level semantic neighbors for target smoothing[/dim]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        self._print_config()

        # Load CLIP features
        console.print("\n[cyan]Loading CLIP features...[/cyan]")
        features_path = self.clip_features_dir / f"{self.partition}.pt"
        data = torch.load(features_path, map_location="cpu", weights_only=False)

        # Find image features key (handles both single-view and multi-view)
        if "image_features" in data:
            clip_features = data["image_features"]
        else:
            # Multi-view: find first image_features_* key
            feature_keys = [k for k in data.keys() if k.startswith("image_features_")]
            if not feature_keys:
                raise KeyError(f"No image features found in {features_path}")
            clip_features = data[feature_keys[0]]
            console.print(f"  Using multi-view key: {feature_keys[0]}")

        labels = data["label"]

        if not self.use_open_vocab:
            return self._run_dataset_only(clip_features, labels)
        else:
            return self._run_open_vocab(clip_features, labels, data)

    def _run_dataset_only(self, clip_features: torch.Tensor, labels: torch.Tensor) -> dict:
        """Build dataset-only knowledge graph."""
        console.print(
            f"  Loaded {len(clip_features)} images, {len(torch.unique(labels))} concepts"
        )

        console.print(f"\n[cyan]Building knowledge graph (k={self.k})...[/cyan]")
        kg = KnowledgeGraph()
        kg.build_concept_neighbors(clip_features, labels, k=self.k)

        console.print(
            f"  Built graph: {len(kg.concept_to_images)} concepts, "
            f"{len(kg.image_to_concept)} images"
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "n_concepts": len(kg.concept_to_images),
            "n_images": len(kg.image_to_concept),
            "k": self.k,
            "output_format": self.output_format,
        }

        if self.output_format == "jsonl":
            output_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.jsonl"
            kg.save(output_path)
            console.print(f"\n[green]Saved JSONL to {output_path}[/green]\n")
            result["output_path"] = str(output_path)
        elif self.output_format == "all":
            jsonl_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.jsonl"
            pt_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.pt"
            kg.save(jsonl_path)
            kg.save(pt_path)
            console.print(f"\n[green]Saved JSONL to {jsonl_path}[/green]")
            console.print(f"[green]Saved .pt to {pt_path}[/green]\n")
            result["output_path_jsonl"] = str(jsonl_path)
            result["output_path_pt"] = str(pt_path)
        else:
            output_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.pt"
            kg.save(output_path)
            console.print(f"\n[green]Saved .pt to {output_path}[/green]\n")
            result["output_path"] = str(output_path)

        return result

    def _run_open_vocab(
        self, clip_features: torch.Tensor, labels: torch.Tensor, data: dict
    ) -> dict:
        """Build open vocabulary knowledge graph."""
        # Step 1: Compute dataset concept embeddings
        console.print("\n[cyan]Step 1: Computing dataset concept embeddings...[/cyan]")
        unique_labels = torch.unique(labels)
        n_dataset = len(unique_labels)
        dataset_embeddings = torch.zeros(n_dataset, clip_features.shape[1])
        dataset_concepts = []

        for concept_id in unique_labels:
            mask = labels == concept_id
            dataset_embeddings[concept_id] = clip_features[mask].mean(dim=0)
            # Use text labels if available, otherwise use concept_id
            if "text" in data:
                texts = data["text"]
                concept_text = (
                    texts[mask.nonzero()[0].item()] if mask.any() else f"concept_{concept_id}"
                )
                dataset_concepts.append(concept_text)
            else:
                dataset_concepts.append(f"concept_{concept_id}")

        console.print(f"  Computed {n_dataset} dataset concept embeddings")

        # Step 2: Expand vocabulary
        console.print(
            f"\n[cyan]Step 2: Expanding vocabulary (max={self.max_extended_concepts})...[/cyan]"
        )
        expander = WordNetExpander()
        extended_concept_objs = expander.expand(dataset_concepts, self.max_extended_concepts)
        extended_concepts = [c.name for c in extended_concept_objs]

        console.print(f"  Expanded to {len(extended_concepts)} additional concepts")

        # Step 3: Encode extended concepts with CLIP text
        console.print(
            "\n[cyan]Step 3: Encoding extended concepts with CLIP text encoder...[/cyan]"
        )
        backend = create_clip_backend(
            model_id=self.resolved_model_id,
            devices=self.resolved_devices,
            feature_mode=self.feature_mode,
            model_cache_dir=self.model_cache_dir,
        )
        extended_embeddings = backend.encode_texts(extended_concepts)
        backend.close()

        console.print(f"  Encoded {len(extended_concepts)} extended concepts")

        # Step 4: Build open knowledge graph
        console.print(f"\n[cyan]Step 4: Building open knowledge graph (k={self.k})...[/cyan]")
        kg = KnowledgeGraph()
        kg.build_from_hybrid_embeddings(
            dataset_concepts=dataset_concepts,
            dataset_embeddings=dataset_embeddings,
            extended_concepts=extended_concepts,
            extended_embeddings=extended_embeddings,
            k=self.k,
            quota_dataset=self.quota_dataset,
            quota_extended=self.quota_extended,
        )

        console.print(f"  Built graph: {len(kg.concepts)} total concepts")

        # Step 5: Save
        self.save_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "n_dataset_concepts": n_dataset,
            "n_extended_concepts": len(extended_concepts),
            "n_total_concepts": len(kg.concepts),
            "k": self.k,
            "output_format": self.output_format,
        }

        if self.output_format == "jsonl":
            output_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.jsonl"
            kg.save(output_path)
            console.print(f"\n[green]Saved JSONL to {output_path}[/green]\n")
            result["output_path"] = str(output_path)
        elif self.output_format == "all":
            jsonl_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.jsonl"
            pt_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.pt"
            kg.save(jsonl_path)
            kg.save(pt_path)
            console.print(f"\n[green]Saved JSONL to {jsonl_path}[/green]")
            console.print(f"[green]Saved .pt to {pt_path}[/green]\n")
            result["output_path_jsonl"] = str(jsonl_path)
            result["output_path_pt"] = str(pt_path)
        else:
            output_path = self.save_dir / f"thingseeg2_kg_{self.partition}_k{self.k}.pt"
            kg.save(output_path)
            console.print(f"\n[green]Saved .pt to {output_path}[/green]\n")
            result["output_path"] = str(output_path)

        return result

    def _print_config(self) -> None:
        """Print configuration table."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="white")

        table.add_row("Mode", "Open Vocabulary" if self.use_open_vocab else "Dataset Only")
        table.add_row("Partition", self.partition)
        table.add_row("Model", f"{self.model_name} / {self.resolved_model_id}")
        table.add_row("Feature mode", self.feature_mode)
        table.add_row("Neighbors (k)", str(self.k))

        if self.use_open_vocab:
            table.add_row("Max extended concepts", str(self.max_extended_concepts))
            table.add_row("Quota dataset", str(self.quota_dataset))
            table.add_row("Quota extended", str(self.quota_extended))
            table.add_row("Output format", self.output_format)

        table.add_row("CLIP features", str(self.clip_features_dir))
        table.add_row("Output dir", str(self.save_dir))
        table.add_row("Seed", str(self.seed))

        console.print(table)
