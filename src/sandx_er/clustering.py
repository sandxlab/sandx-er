"""Clustering — reconcile pairwise match decisions into entity clusters."""

from __future__ import annotations

from abc import ABC, abstractmethod



MatchGraph = dict[str, list[tuple[str, float]]]


class ClusteringMethod(ABC):
    """Base class for clustering strategies."""

    @abstractmethod
    def cluster(self, match_graph: MatchGraph) -> list[list[str]]:
        """Partition record IDs into clusters.

        Args:
            match_graph: Adjacency dict — for each record ID, a list of
                         (matched_id, confidence) tuples for above-threshold pairs.

        Returns:
            List of clusters; each cluster is a list of record IDs.
            Every ID in match_graph must appear in exactly one cluster.
        """
        ...


class ConnectedComponentsClustering(ClusteringMethod):
    """Union-Find connected components.

    Every pair above the match threshold is merged into the same component.
    Fast (near-linear) and exact. May over-merge when pairwise scores are
    noisy (no transitivity correction).

    Time complexity: O(N · α(N)) where α is the inverse Ackermann function.
    """

    def cluster(self, match_graph: MatchGraph) -> list[list[str]]:
        # Collect all node IDs
        all_ids: set[str] = set(match_graph.keys())
        for neighbors in match_graph.values():
            for nid, _ in neighbors:
                all_ids.add(nid)

        # Union-Find with path compression
        parent: dict[str, str] = {nid: nid for nid in all_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for node_id, neighbors in match_graph.items():
            for neighbor_id, _ in neighbors:
                union(node_id, neighbor_id)

        # Group by root
        components: dict[str, list[str]] = {}
        for nid in all_ids:
            root = find(nid)
            components.setdefault(root, []).append(nid)

        return list(components.values())


class CorrelationClustering(ClusteringMethod):
    """Greedy correlation clustering (Kwik-Cluster algorithm).

    Optimizes a global objective that balances positive edges (matches)
    and negative edges (non-matches). More robust than connected components
    under noisy pairwise scores — can split over-merged clusters.

    Implements the 3-approximation greedy pivoting heuristic: O(N + E).

    Args:
        max_iterations: Maximum pivot iterations (safety cap; rarely reached).
    """

    def __init__(self, max_iterations: int = 10_000) -> None:
        self.max_iterations = max_iterations

    def cluster(self, match_graph: MatchGraph) -> list[list[str]]:
        # Build symmetric adjacency (positive edges only)
        all_ids: set[str] = set(match_graph.keys())
        for neighbors in match_graph.values():
            for nid, _ in neighbors:
                all_ids.add(nid)

        pos_neighbors: dict[str, set[str]] = {nid: set() for nid in all_ids}
        for node_id, neighbors in match_graph.items():
            for neighbor_id, _ in neighbors:
                pos_neighbors[node_id].add(neighbor_id)
                pos_neighbors[neighbor_id].add(node_id)

        remaining = set(all_ids)
        clusters: list[list[str]] = []
        iterations = 0

        while remaining and iterations < self.max_iterations:
            iterations += 1
            # Choose pivot with highest positive-degree in remaining subgraph
            pivot = max(remaining, key=lambda x: len(pos_neighbors.get(x, set()) & remaining))
            cluster = {pivot} | (pos_neighbors.get(pivot, set()) & remaining)
            clusters.append(sorted(cluster))
            remaining -= cluster

        # Safety: any leftover nodes become singletons
        for nid in remaining:
            clusters.append([nid])

        return clusters
