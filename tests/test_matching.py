"""Tests for sandx_er.matching."""

from __future__ import annotations

import pandas as pd
import pytest

from sandx_er.matching import JaccardScorer


def _records(*texts: str) -> pd.DataFrame:
    return pd.DataFrame({"text": list(texts)})


# ---------------------------------------------------------------------------
# JaccardScorer
# ---------------------------------------------------------------------------

class TestJaccardScorer:
    def test_identical_records_score_one(self):
        records = _records("apple juice", "orange soda")
        scorer = JaccardScorer()
        scores = scorer.score(records, [("0", "0")])
        assert scores[("0", "0")] == pytest.approx(1.0)

    def test_identical_pair_scores_one(self):
        records = _records("apple juice", "apple juice")
        scorer = JaccardScorer()
        scores = scorer.score(records, [("0", "1")])
        assert scores[("0", "1")] == pytest.approx(1.0)

    def test_disjoint_records_score_zero(self):
        records = _records("aaaa bbbb", "zzzz yyyy")
        scorer = JaccardScorer()
        scores = scorer.score(records, [("0", "1")])
        assert scores[("0", "1")] == pytest.approx(0.0)

    def test_partial_overlap(self):
        records = _records("abc def ghi", "abc xyz")
        scorer = JaccardScorer()
        scores = scorer.score(records, [("0", "1")])
        s = scores[("0", "1")]
        assert 0.0 < s < 1.0

    def test_scores_in_range(self):
        records = _records("entity resolution", "entity matching", "record linkage")
        pairs = [("0", "1"), ("0", "2"), ("1", "2")]
        scorer = JaccardScorer()
        scores = scorer.score(records, pairs)
        for s in scores.values():
            assert 0.0 <= s <= 1.0

    def test_returns_all_pairs(self):
        records = _records("a b c", "b c d", "e f g")
        pairs = [("0", "1"), ("1", "2")]
        scorer = JaccardScorer()
        scores = scorer.score(records, pairs)
        assert set(scores.keys()) == {("0", "1"), ("1", "2")}

    def test_custom_fields(self):
        records = pd.DataFrame({"name": ["Alice Smith", "Alice S."], "city": ["Boston", "Cambridge"]})
        scorer = JaccardScorer(fields=["name"])
        scores = scorer.score(records, [("0", "1")])
        assert ("0", "1") in scores

    def test_multicolumn_records(self):
        records = pd.DataFrame({
            "first": ["John", "Jon"],
            "last": ["Smith", "Smith"],
            "dob": ["1980-01-01", "1980-01-01"],
        })
        scorer = JaccardScorer()
        scores = scorer.score(records, [("0", "1")])
        assert scores[("0", "1")] > 0.5
