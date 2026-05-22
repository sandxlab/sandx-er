"""Tests for EntityResolver pipeline."""

import pytest
import pandas as pd

from sandx_er import EntityResolver


SAMPLE_RECORDS = pd.DataFrame([
    {"id": "r1", "name": "Acme Corp",      "city": "Boston"},
    {"id": "r2", "name": "Acme Corp.",     "city": "Boston"},
    {"id": "r3", "name": "GlobalTech Inc", "city": "New York"},
    {"id": "r4", "name": "Global Tech",    "city": "New York"},
])


class TestEntityResolverInit:
    def test_default_params(self):
        er = EntityResolver()
        assert er.blocking == "lsh"
        assert er.similarity == "embedding"
        assert er.threshold == 0.85

    def test_custom_params(self):
        er = EntityResolver(blocking="snm", similarity="features", threshold=0.7)
        assert er.blocking == "snm"
        assert er.threshold == 0.7


class TestEntityResolverResolve:
    def test_resolve_raises_not_implemented(self):
        er = EntityResolver()
        with pytest.raises(NotImplementedError):
            er.resolve(SAMPLE_RECORDS)


# Phase 2 tests (to be implemented):
#
# class TestBlocking:
#     def test_lsh_reduces_pairs(self): ...
#     def test_embedding_ann_blocking(self): ...
#
# class TestClustering:
#     def test_connected_components(self): ...
#     def test_correlation_clustering(self): ...
#
# class TestEndToEnd:
#     def test_abt_buy_benchmark(self): ...
#     def test_dblp_acm_benchmark(self): ...
