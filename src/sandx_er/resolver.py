"""EntityResolver — main pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


BlockingMethod = Literal["lsh", "snm", "embedding", "standard"]
SimilarityMethod = Literal["embedding", "features", "learned"]


@dataclass
class EntityCluster:
    canonical_id: str
    record_ids: list[str]
    confidence: float
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.record_ids)


@dataclass
class ResolutionResult:
    clusters: list[EntityCluster]
    n_records: int
    n_clusters: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_clusters = len(self.clusters)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {"canonical_id": c.canonical_id, "record_id": rid, "confidence": c.confidence}
            for c in self.clusters
            for rid in c.record_ids
        ]
        return pd.DataFrame(rows)


class EntityResolver:
    """Modular entity resolution pipeline.

    Stages: blocking → matching → clustering → resolved identity graph.
    Each stage is independently configurable.

    Args:
        blocking:   Candidate generation method.
        similarity: Pairwise similarity scoring method.
        threshold:  Minimum confidence score to assert a match (0–1).
    """

    def __init__(
        self,
        blocking: BlockingMethod = "lsh",
        similarity: SimilarityMethod = "embedding",
        threshold: float = 0.85,
    ) -> None:
        self.blocking = blocking
        self.similarity = similarity
        self.threshold = threshold

    def resolve(self, records: pd.DataFrame) -> ResolutionResult:
        """Resolve entity identity across the given record set.

        Args:
            records: DataFrame where each row is a record to be resolved.

        Returns:
            ResolutionResult containing entity clusters with confidence scores.
        """
        raise NotImplementedError(
            "sandx-er v0.1 is a Phase 1 skeleton. "
            "Full implementation ships in Phase 2."
        )
