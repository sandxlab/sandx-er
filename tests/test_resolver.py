"""Tests for sandx_er.resolver (EntityResolver integration tests)."""

from __future__ import annotations

import pandas as pd
import pytest

from sandx_er import EntityResolver, ResolutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _product_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [
                "Sony 65-Inch 4K TV",
                "Sony 65 Inch 4K Television",
                "Samsung 55-Inch QLED TV",
                "Samsung 55 QLED Television",
                "LG OLED C2 55-Inch",
            ],
            "brand": ["Sony", "Sony", "Samsung", "Samsung", "LG"],
        }
    )


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

class TestEntityResolverAPI:
    def test_default_params(self):
        er = EntityResolver()
        assert er.blocking == "lsh"
        assert er.similarity == "jaccard"
        assert er.threshold == 0.5

    def test_returns_resolution_result(self):
        er = EntityResolver(blocking="lsh", similarity="jaccard", threshold=0.3)
        result = er.resolve(_product_records())
        assert isinstance(result, ResolutionResult)

    def test_n_records_correct(self):
        records = _product_records()
        er = EntityResolver(threshold=0.3)
        result = er.resolve(records)
        assert result.n_records == len(records)

    def test_all_records_assigned(self):
        records = _product_records()
        er = EntityResolver(threshold=0.2)
        result = er.resolve(records)
        assigned_ids = {rid for c in result.clusters for rid in c.record_ids}
        expected_ids = {str(i) for i in records.index}
        assert assigned_ids == expected_ids

    def test_no_duplicate_record_ids(self):
        records = _product_records()
        er = EntityResolver(threshold=0.3)
        result = er.resolve(records)
        all_ids = [rid for c in result.clusters for rid in c.record_ids]
        assert len(all_ids) == len(set(all_ids))

    def test_to_dataframe_columns(self):
        records = _product_records()
        er = EntityResolver(threshold=0.3)
        result = er.resolve(records)
        df = result.to_dataframe()
        assert set(df.columns) >= {"record_id", "canonical_id", "confidence"}

    def test_to_dataframe_row_count(self):
        records = _product_records()
        er = EntityResolver(threshold=0.3)
        result = er.resolve(records)
        df = result.to_dataframe()
        assert len(df) == len(records)

    def test_confidence_in_range(self):
        records = _product_records()
        er = EntityResolver(threshold=0.3)
        result = er.resolve(records)
        for cluster in result.clusters:
            assert 0.0 <= cluster.confidence <= 1.0

    def test_cluster_size_matches_record_ids(self):
        records = _product_records()
        er = EntityResolver(threshold=0.3)
        result = er.resolve(records)
        for cluster in result.clusters:
            assert cluster.size == len(cluster.record_ids)


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_high_threshold_more_singletons(self):
        records = _product_records()
        er_low = EntityResolver(threshold=0.1)
        er_high = EntityResolver(threshold=0.99)
        result_low = er_low.resolve(records)
        result_high = er_high.resolve(records)
        assert result_low.n_clusters <= result_high.n_clusters

    def test_threshold_one_singletons_only(self):
        records = _product_records()
        er = EntityResolver(threshold=1.0)
        result = er.resolve(records)
        assert result.n_clusters == len(records)


# ---------------------------------------------------------------------------
# Blocking methods
# ---------------------------------------------------------------------------

class TestBlocking:
    def test_lsh_blocking(self):
        records = _product_records()
        er = EntityResolver(blocking="lsh", threshold=0.3)
        result = er.resolve(records)
        assert result.n_records == len(records)

    def test_snm_blocking(self):
        records = _product_records()
        er = EntityResolver(blocking="snm", key_field="brand", threshold=0.3)
        result = er.resolve(records)
        assert result.n_records == len(records)

    def test_snm_missing_key_field_raises(self):
        records = _product_records()
        er = EntityResolver(blocking="snm")
        with pytest.raises(ValueError, match="key_field"):
            er.resolve(records)

    def test_unknown_blocking_raises(self):
        records = _product_records()
        er = EntityResolver(blocking="invalid")
        with pytest.raises(ValueError, match="Unknown blocking"):
            er.resolve(records)


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

class TestClustering:
    def test_connected_components(self):
        records = _product_records()
        er = EntityResolver(clustering="connected_components", threshold=0.3)
        result = er.resolve(records)
        assert result.n_records == len(records)

    def test_correlation_clustering(self):
        records = _product_records()
        er = EntityResolver(clustering="correlation", threshold=0.3)
        result = er.resolve(records)
        assert result.n_records == len(records)

    def test_unknown_clustering_raises(self):
        records = _product_records()
        er = EntityResolver(clustering="bad_method")
        with pytest.raises(ValueError, match="Unknown clustering"):
            er.resolve(records)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe_raises(self):
        er = EntityResolver()
        with pytest.raises(ValueError, match="empty"):
            er.resolve(pd.DataFrame())

    def test_invalid_threshold_raises(self):
        records = _product_records()
        er = EntityResolver(threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            er.resolve(records)

    def test_single_record_is_singleton(self):
        records = pd.DataFrame({"text": ["only one record"]})
        er = EntityResolver(threshold=0.5)
        result = er.resolve(records)
        assert result.n_clusters == 1
        assert result.clusters[0].confidence == 1.0

    def test_two_identical_records_merged(self):
        records = pd.DataFrame({"text": ["exact duplicate text", "exact duplicate text"]})
        er = EntityResolver(blocking="lsh", similarity="jaccard", threshold=0.5)
        result = er.resolve(records)
        assert result.n_clusters == 1

    def test_custom_index_preserved(self):
        records = pd.DataFrame({"text": ["a b c", "x y z"]}, index=["rec_a", "rec_b"])
        er = EntityResolver(threshold=0.5)
        result = er.resolve(records)
        df = result.to_dataframe()
        assert set(df["record_id"]) == {"rec_a", "rec_b"}
