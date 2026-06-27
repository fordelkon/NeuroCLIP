# Open Vocabulary Knowledge Graph - Design Ideas

## Motivation

### Current Limitations

- Knowledge graph limited to 1854 dataset concepts
- Uses averaged CLIP image features for concept similarity
- Neighbor relationships bounded by dataset scope

### Key Insight

**Human cognition extends beyond dataset boundaries**: In RSVP paradigm, when subjects see image stimuli, they associate with:

1. Similar images seen in the dataset (visual similarity)
2. Broader category concepts (e.g., "dog" → "canine", "animal")
3. Personally familiar related concepts (semantic associations)

Therefore, the knowledge graph should expand beyond the dataset by integrating external knowledge bases.

______________________________________________________________________

## Design Solution

### 1. Vocabulary Expansion

**Goal**: Expand from 1854 to ~5000-12000 concepts

**External Knowledge Bases**:

- **WordNet**: Hierarchical semantic network (hypernyms, hyponyms, siblings)
- **ConceptNet**: Common-sense knowledge graph (RelatedTo, IsA, PartOf)
- **ImageNet**: Visual category hierarchy

**Initial Implementation**: WordNet only (easiest to integrate, no external API required)

### 2. Hybrid Encoding Strategy

**Core Tradeoff**:

- RSVP paradigm emphasizes **visual processing** → dataset concepts should preserve visual similarity
- Extended concepts lack corresponding images → require text encoding

**Solution**:

```
Dataset concepts (1854)      → CLIP image feature averaging → Preserve visual similarity
Extended concepts (~3k-10k)  → CLIP text encoding           → Semantic associations
```

### 3. Neighbor Selection Strategy

**Quota Mechanism**: For each concept, select k neighbors:

- 5 from dataset (visual similarity)
- 5 from extended vocabulary (semantic associations)

**Benefits**:

- Balance visual and semantic dimensions
- Prevent one type from dominating

### 4. Storage Format

**JSONL as Primary Format**:

```jsonl
{"id": 0, "name": "dog", "source": "thingseeg2", "emb_type": "image_avg", "n_images": 50, "neighbors": [...]}
{"id": 1854, "name": "canine", "source": "wordnet", "emb_type": "text", "synset": "canine.n.01", "neighbors": [...]}
```

**Advantages**:

- Human-readable for inspecting external concept integration quality
- Version control friendly (git diff)
- Iterative debugging (adjust expansion parameters without re-encoding)
- Reusable (same KG for different experiments)

**.pt as Training Format**:

- Converted from JSONL
- Used only during training for loading efficiency

______________________________________________________________________

## Implementation Plan

### Phase 1: Core Modules

- `vocabulary_expander.py`: Base class + WordNetExpander
- `open_knowledge_graph.py`: Open KG class + JSONL serialization

### Phase 2: Build Pipeline

- `open_kg_builder.py`: Complete build workflow
  1. Load dataset concepts and CLIP image features
  2. Expand vocabulary using WordNet
  3. Encode extended concepts with CLIP text encoder
  4. Build hybrid knowledge graph
  5. Save as JSONL (+ optional .pt)

### Phase 3: Validation & Tuning

- Generate sample JSONL and manually inspect
- Verify neighbor relationship quality
- Tune expansion parameters (k value, quota ratio, expansion sources)

______________________________________________________________________

## Configuration Interface

```yaml
expansion:
  sources: ['wordnet']
  max_concepts: 5000

encoding:
  clip_model: "ViT-L/14"
  text_template: "a photo of {}"

graph:
  k: 10
  quota_dataset: 5
  quota_extended: 5

output:
  format: "jsonl"
  save_pt: true
```

______________________________________________________________________

## Key Design Decisions

1. **Why not use text encoding for all concepts?**

   - Visual similarity is important in RSVP paradigm
   - Dataset concepts have rich image samples, averaging is more accurate

2. **Why JSONL instead of direct .pt?**

   - Open KG integrates external knowledge, requires inspectability
   - Facilitates iterative debugging and version control

3. **Why implement WordNet first?**

   - No external API required, easy to integrate
   - Clear hierarchical structure (hypernym/hyponym)
   - Broad coverage, high quality

4. **Why quota mechanism?**

   - Prevent extended concepts from "drowning out" dataset concepts
   - Ensure both visual and semantic dimensions are represented

______________________________________________________________________

## Future Extensions

- Integrate ConceptNet (common-sense reasoning)
- Integrate ImageNet hierarchy (visual categories)
- Multi-hop neighbors (first-order, second-order)
- Dynamic weighting (adjust visual/semantic weights based on EEG signal stage)
