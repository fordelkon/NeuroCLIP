"""Vocabulary expansion for open knowledge graph."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import nltk
from nltk.corpus import wordnet as wn


def _ensure_wordnet_downloaded() -> None:
    """Ensure WordNet corpus is downloaded."""
    try:
        wn.synsets("test")
    except LookupError:
        print("Downloading WordNet corpus (first time only)...")
        nltk.download("wordnet", quiet=True)
        print("WordNet corpus downloaded successfully.")


@dataclass
class ExtendedConcept:
    """Extended concept from external knowledge base."""

    name: str
    source: str
    metadata: dict


class VocabularyExpander(ABC):
    """Base class for vocabulary expansion."""

    @abstractmethod
    def expand(self, seed_concepts: list[str], max_concepts: int) -> list[ExtendedConcept]:
        """Expand vocabulary from seed concepts."""
        pass


class WordNetExpander(VocabularyExpander):
    """Expand vocabulary using WordNet hierarchy."""

    def __init__(self):
        _ensure_wordnet_downloaded()

    def expand(self, seed_concepts: list[str], max_concepts: int) -> list[ExtendedConcept]:
        """Expand vocabulary from seed concepts using WordNet relations."""
        extended = []
        seen = {c.lower() for c in seed_concepts}

        for concept in seed_concepts:
            if len(extended) >= max_concepts:
                break

            synsets = wn.synsets(concept.replace(" ", "_"), pos=wn.NOUN)
            if not synsets:
                continue

            synset = synsets[0]

            # Hypernyms (parent concepts)
            for hyper in synset.hypernyms():
                name = hyper.lemmas()[0].name().replace("_", " ")
                if name.lower() not in seen:
                    extended.append(
                        ExtendedConcept(
                            name=name,
                            source="wordnet",
                            metadata={"synset": hyper.name(), "relation": "hypernym"},
                        )
                    )
                    seen.add(name.lower())
                    if len(extended) >= max_concepts:
                        break

            # Hyponyms (child concepts)
            for hypo in synset.hyponyms():
                if len(extended) >= max_concepts:
                    break
                name = hypo.lemmas()[0].name().replace("_", " ")
                if name.lower() not in seen:
                    extended.append(
                        ExtendedConcept(
                            name=name,
                            source="wordnet",
                            metadata={"synset": hypo.name(), "relation": "hyponym"},
                        )
                    )
                    seen.add(name.lower())

        return extended[:max_concepts]
