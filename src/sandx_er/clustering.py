"""Clustering — reconcile pairwise match decisions into entity clusters."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


MatchGraph = dict[str, list[tuple[str, float]]]


class ClusteringMethod(ABC):
    """Base class for clustering strategies."""

    @abstractmethod
    def cluster(self, match_graph: MatchGraph) -> list[list[str]]:
        """Return list of clusters; each cluster is a list of record IDs."""
        ...


class ConnectedComponentsClustering(ClusteringMethod):
    """Union-Find connected components.

    Every pair above the match threshold is merged into a component.
    Fast and transitive; may over-merge when edge scores are noisy.
    """

    def cluster(self, match_graph: MatchGraph) -> list[list[str]]:
        raise NotImplementedError("Phase 2")


class CorrelationClustering(ClusteringMethod):
    """Correlation clustering via local search.

    Optimizes a global objective balancing positive and negative edges.
    More accurate than connected components under noisy pairwise scores;
    NP-hard in general — solved via greedy pivoting heuristic.
    """

    def __init__(self, max_iterations: int = 100) -> None:
        self.max_iterations = max_iterations

    def cluster(self, match_graph: MatchGraph) -> list[list[str]]:
        raise NotImplementedError("Phase 2")
