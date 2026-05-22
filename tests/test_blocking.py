"""Tests for sandx_er.blocking."""

from __future__ import annotations

import pandas as pd
import pytest

from sandx_er.blocking import LSHBlocking, SortedNeighborhoodBlocking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_records(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"text": texts})


# ---------------------------------------------------------------------------
# LSH Blocking
# ---------------------------------------------------------------------------

class TestLSHBlocking:
    def test_returns_candidate_pairs(self):
        records = _make_records(["apple juice", "apple juce", "orange soda"])
        blocker = LSHBlocking(n_bands=10, n_rows=3)
        pairs = list(blocker.generate_candidates(records))
        assert isinstance(pairs, list)
        assert all(len(p) == 2 for p in pairs)

    def test_no_self_pairs(self):
        records = _make_records(["a b c", "a b c", "x y z"])
        blocker = LSHBlocking()
        pairs = list(blocker.generate_candidates(records))
        assert all(a != b for a, b in pairs)

    def test_no_duplicate_pairs(self):
        records = _make_records(["abc def", "abc def", "ghi jkl", "abc dex"])
        blocker = LSHBlocking()
        pairs = list(blocker.generate_candidates(records))
        assert len(pairs) == len(set(pairs))

    def test_similar_records_are_candidates(self):
        # Two nearly-identical records should share at least one bucket
        records = _make_records(["entity resolution system", "entity resolution systems", "unrelated content here"])
        blocker = LSHBlocking(n_bands=30, n_rows=3)
        pairs = list(blocker.generate_candidates(records))
        pair_ids = {(min(a, b), max(a, b)) for a, b in pairs}
        assert ("0", "1") in pair_ids

    def test_single_record_no_pairs(self):
        records = _make_records(["only one"])
        blocker = LSHBlocking()
        pairs = list(blocker.generate_candidates(records))
        assert pairs == []

    def test_two_records(self):
        records = _make_records(["foo bar baz", "foo bar baz"])
        blocker = LSHBlocking(n_bands=20, n_rows=3)
        pairs = list(blocker.generate_candidates(records))
        assert len(pairs) >= 1

    def test_works_with_multiple_columns(self):
        records = pd.DataFrame({"name": ["John Smith", "John Smyth"], "city": ["Boston", "Boston"]})
        blocker = LSHBlocking(n_bands=20, n_rows=3)
        pairs = list(blocker.generate_candidates(records))
        assert isinstance(pairs, list)


# ---------------------------------------------------------------------------
# Sorted Neighborhood Blocking
# ---------------------------------------------------------------------------

class TestSortedNeighborhoodBlocking:
    def test_basic_candidates(self):
        records = pd.DataFrame({"name": ["Alice", "Alice B", "Charlie"]})
        blocker = SortedNeighborhoodBlocking(key_field="name", window_size=2)
        pairs = list(blocker.generate_candidates(records))
        assert len(pairs) > 0

    def test_window_size_1_no_pairs(self):
        records = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
        blocker = SortedNeighborhoodBlocking(key_field="name", window_size=1)
        pairs = list(blocker.generate_candidates(records))
        assert pairs == []

    def test_no_self_pairs(self):
        records = pd.DataFrame({"name": ["Alice", "Alice", "Bob"]})
        blocker = SortedNeighborhoodBlocking(key_field="name", window_size=3)
        pairs = list(blocker.generate_candidates(records))
        assert all(a != b for a, b in pairs)

    def test_no_duplicate_pairs(self):
        records = pd.DataFrame({"name": ["A", "B", "C", "D", "E"]})
        blocker = SortedNeighborhoodBlocking(key_field="name", window_size=3)
        pairs = list(blocker.generate_candidates(records))
        assert len(pairs) == len(set(pairs))

    def test_missing_key_field_raises(self):
        records = pd.DataFrame({"name": ["Alice", "Bob"]})
        blocker = SortedNeighborhoodBlocking(key_field="nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            list(blocker.generate_candidates(records))

    def test_all_within_window_are_candidates(self):
        records = pd.DataFrame({"key": ["a", "b", "c"]})
        blocker = SortedNeighborhoodBlocking(key_field="key", window_size=3)
        pairs = set(blocker.generate_candidates(records))
        # With window_size=3 on sorted ["a","b","c"]: (0,1), (0,2), (1,2)
        assert len(pairs) == 3
