"""EntityResolver — main pipeline orchestrator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd

from .blocking import BlockingMethod, EmbeddingANNBlocking, LSHBlocking, SortedNeighborhoodBlocking
from .clustering import (
    ClusteringMethod,
    ConnectedComponentsClustering,
    CorrelationClustering,
)
from .matching import CosineSimilarityScorer, JaccardScorer, SimilarityScorer


@dataclass
class EntityCluster:
    """A resolved entity cluster — a set of records that refer to the same entity."""

    canonical_id: str
    record_ids: list[str]
    confidence: float
    size: int = field(init=False)

    def __post_init__(self) -> None:
        self.size = len(self.record_ids)


@dataclass
class ResolutionResult:
    """Output of EntityResolver.resolve()."""

    clusters: list[EntityCluster]
    n_records: int
    n_clusters: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_clusters = len(self.clusters)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns: record_id, canonical_id, confidence."""
        rows = [
            {"record_id": rid, "canonical_id": c.canonical_id, "confidence": c.confidence}
            for c in self.clusters
            for rid in c.record_ids
        ]
        return pd.DataFrame(rows)


class EntityResolver:
    """Modular entity resolution pipeline.

    Pipeline stages: blocking → matching → clustering → resolved identity graph.
    Each stage is independently configurable via string shortcuts or custom objects.

    Args:
        blocking:   Candidate generation strategy.
                    String shortcuts: "lsh" (default), "snm", "embedding".
                    Or pass a custom BlockingMethod instance.
        similarity: Pairwise similarity scoring strategy.
                    String shortcuts: "jaccard" (default), "embedding".
                    Or pass a custom SimilarityScorer instance.
        clustering: Cluster assignment strategy.
                    String shortcuts: "connected_components" (default), "correlation".
                    Or pass a custom ClusteringMethod instance.
        threshold:  Minimum similarity score to assert a match [0, 1].
        key_field:  Column name used by SortedNeighborhoodBlocking (blocking="snm").
        embed_model: sandx-embed model name for embedding-based stages.

    Example:
        er = EntityResolver(blocking="lsh", similarity="jaccard", threshold=0.5)
        result = er.resolve(records_df)
        df = result.to_dataframe()
    """

    def __init__(
        self,
        blocking: Union[str, BlockingMethod] = "lsh",
        similarity: Union[str, SimilarityScorer] = "jaccard",
        clustering: Union[str, ClusteringMethod] = "connected_components",
        threshold: float = 0.5,
        key_field: str | None = None,
        embed_model: str = "sentence-bert",
    ) -> None:
        _VALID_BLOCKING = {"lsh", "snm", "embedding"}
        _VALID_SIMILARITY = {"jaccard", "embedding"}
        _VALID_CLUSTERING = {"connected_components", "correlation"}

        if isinstance(blocking, str) and blocking not in _VALID_BLOCKING:
            raise ValueError(
                f"Unknown blocking method: {blocking!r}. "
                f"Choose one of {sorted(_VALID_BLOCKING)} or pass a BlockingMethod instance."
            )
        if isinstance(similarity, str) and similarity not in _VALID_SIMILARITY:
            raise ValueError(
                f"Unknown similarity method: {similarity!r}. "
                f"Choose one of {sorted(_VALID_SIMILARITY)} or pass a SimilarityScorer instance."
            )
        if isinstance(clustering, str) and clustering not in _VALID_CLUSTERING:
            raise ValueError(
                f"Unknown clustering method: {clustering!r}. "
                f"Choose one of {sorted(_VALID_CLUSTERING)} or pass a ClusteringMethod instance."
            )

        self.blocking = blocking
        self.similarity = similarity
        self.clustering = clustering
        self.threshold = threshold
        self.key_field = key_field
        self.embed_model = embed_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, records: pd.DataFrame) -> ResolutionResult:
        """Resolve entity identity across the given record set.

        Args:
            records: DataFrame where each row is one record observation.
                     The index is used as the record ID; it must be unique.

        Returns:
            ResolutionResult containing entity clusters with confidence scores.

        Raises:
            ValueError: On empty input, duplicate index, or invalid threshold.
        """
        self._validate(records)

        records = records.copy()
        records.index = records.index.astype(str)

        if not records.index.is_unique:
            raise ValueError("records.index must be unique.")

        # 1. Blocking
        blocker = self._make_blocker()
        candidates = list(blocker.generate_candidates(records))

        if not candidates:
            return self._singleton_result(records)

        # 2. Scoring
        scorer = self._make_scorer()
        pair_scores = scorer.score(records, candidates)

        # 3. Build match graph (edges above threshold)
        match_graph: dict[str, list[tuple[str, float]]] = {
            rid: [] for rid in records.index
        }
        for (a, b), score in pair_scores.items():
            if score >= self.threshold:
                match_graph.setdefault(a, []).append((b, score))
                match_graph.setdefault(b, []).append((a, score))

        # 4. Clustering
        clusterer = self._make_clusterer()
        raw_clusters = clusterer.cluster(match_graph)

        # 5. Add singletons for any record not placed in a cluster
        clustered = {rid for cluster in raw_clusters for rid in cluster}
        for rid in records.index:
            if rid not in clustered:
                raw_clusters.append([rid])

        # 6. Build result with confidence scores
        clusters = [
            EntityCluster(
                canonical_id=str(uuid.uuid4()),
                record_ids=cluster,
                confidence=self._cluster_confidence(cluster, pair_scores),
            )
            for cluster in raw_clusters
            if cluster
        ]

        return ResolutionResult(clusters=clusters, n_records=len(records))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, records: pd.DataFrame) -> None:
        if records.empty:
            raise ValueError("records DataFrame cannot be empty.")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}.")

    def _singleton_result(self, records: pd.DataFrame) -> ResolutionResult:
        clusters = [
            EntityCluster(canonical_id=str(uuid.uuid4()), record_ids=[rid], confidence=1.0)
            for rid in records.index
        ]
        return ResolutionResult(clusters=clusters, n_records=len(records))

    def _make_blocker(self) -> BlockingMethod:
        if isinstance(self.blocking, BlockingMethod):
            return self.blocking
        if self.blocking == "lsh":
            return LSHBlocking()
        if self.blocking == "snm":
            if not self.key_field:
                raise ValueError("blocking='snm' requires key_field to be set.")
            return SortedNeighborhoodBlocking(key_field=self.key_field)
        if self.blocking == "embedding":
            return EmbeddingANNBlocking(model=self.embed_model)
        raise ValueError(
            f"Unknown blocking method: {self.blocking!r}. "
            "Use 'lsh', 'snm', 'embedding', or a BlockingMethod instance."
        )

    def _make_scorer(self) -> SimilarityScorer:
        if isinstance(self.similarity, SimilarityScorer):
            return self.similarity
        if self.similarity == "jaccard":
            return JaccardScorer()
        if self.similarity == "embedding":
            return CosineSimilarityScorer(model=self.embed_model)
        raise ValueError(
            f"Unknown similarity method: {self.similarity!r}. "
            "Use 'jaccard', 'embedding', or a SimilarityScorer instance."
        )

    def _make_clusterer(self) -> ClusteringMethod:
        if isinstance(self.clustering, ClusteringMethod):
            return self.clustering
        if self.clustering == "connected_components":
            return ConnectedComponentsClustering()
        if self.clustering == "correlation":
            return CorrelationClustering()
        raise ValueError(
            f"Unknown clustering method: {self.clustering!r}. "
            "Use 'connected_components', 'correlation', or a ClusteringMethod instance."
        )

    @staticmethod
    def _cluster_confidence(
        cluster: list[str], pair_scores: dict[tuple[str, str], float]
    ) -> float:
        if len(cluster) == 1:
            return 1.0
        scores: list[float] = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                a, b = cluster[i], cluster[j]
                s = pair_scores.get((a, b), pair_scores.get((b, a), None))
                if s is not None:
                    scores.append(s)
        return float(np.mean(scores)) if scores else 1.0
