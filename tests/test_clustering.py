"""Tests for sandx_er.clustering."""

from __future__ import annotations


from sandx_er.clustering import ConnectedComponentsClustering, CorrelationClustering


def _graph(*edges: tuple[str, str, float]) -> dict[str, list[tuple[str, float]]]:
    g: dict[str, list[tuple[str, float]]] = {}
    for a, b, w in edges:
        g.setdefault(a, []).append((b, w))
        g.setdefault(b, []).append((a, w))
    return g


# ---------------------------------------------------------------------------
# ConnectedComponentsClustering
# ---------------------------------------------------------------------------

class TestConnectedComponents:
    def test_single_cluster(self):
        g = _graph(("a", "b", 0.9), ("b", "c", 0.8))
        clusters = ConnectedComponentsClustering().cluster(g)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a", "b", "c"}

    def test_two_separate_clusters(self):
        g = _graph(("a", "b", 0.9), ("c", "d", 0.8))
        clusters = ConnectedComponentsClustering().cluster(g)
        assert len(clusters) == 2
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [2, 2]

    def test_singletons(self):
        g: dict[str, list] = {"a": [], "b": [], "c": []}
        clusters = ConnectedComponentsClustering().cluster(g)
        assert len(clusters) == 3
        assert all(len(c) == 1 for c in clusters)

    def test_empty_graph(self):
        clusters = ConnectedComponentsClustering().cluster({})
        assert clusters == []

    def test_all_ids_accounted_for(self):
        g = _graph(("x", "y", 0.9), ("y", "z", 0.7), ("p", "q", 0.8))
        clusters = ConnectedComponentsClustering().cluster(g)
        all_ids = {rid for cluster in clusters for rid in cluster}
        assert all_ids == {"x", "y", "z", "p", "q"}

    def test_transitivity(self):
        # a-b and b-c → a, b, c should be in one cluster
        g = _graph(("a", "b", 0.9), ("b", "c", 0.9))
        clusters = ConnectedComponentsClustering().cluster(g)
        assert len(clusters) == 1

    def test_no_duplicate_ids_across_clusters(self):
        g = _graph(("a", "b", 0.9), ("a", "c", 0.8), ("d", "e", 0.7))
        clusters = ConnectedComponentsClustering().cluster(g)
        all_ids = [rid for cluster in clusters for rid in cluster]
        assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# CorrelationClustering
# ---------------------------------------------------------------------------

class TestCorrelationClustering:
    def test_returns_list_of_lists(self):
        g = _graph(("a", "b", 0.9))
        clusters = CorrelationClustering().cluster(g)
        assert isinstance(clusters, list)
        assert all(isinstance(c, list) for c in clusters)

    def test_all_ids_accounted_for(self):
        g = _graph(("a", "b", 0.9), ("c", "d", 0.8), ("e", "f", 0.7))
        clusters = CorrelationClustering().cluster(g)
        all_ids = {rid for cluster in clusters for rid in cluster}
        assert all_ids == {"a", "b", "c", "d", "e", "f"}

    def test_no_duplicate_ids(self):
        g = _graph(("a", "b", 0.9), ("b", "c", 0.8), ("c", "d", 0.7))
        clusters = CorrelationClustering().cluster(g)
        all_ids = [rid for cluster in clusters for rid in cluster]
        assert len(all_ids) == len(set(all_ids))

    def test_isolated_nodes(self):
        g: dict[str, list] = {"a": [], "b": [], "c": []}
        clusters = CorrelationClustering().cluster(g)
        assert len(clusters) == 3

    def test_empty_graph(self):
        clusters = CorrelationClustering().cluster({})
        assert clusters == []

    def test_dense_cluster(self):
        g = _graph(("a", "b", 0.9), ("a", "c", 0.9), ("b", "c", 0.9))
        clusters = CorrelationClustering().cluster(g)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"a", "b", "c"}
