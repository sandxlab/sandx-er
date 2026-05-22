"""Matching — pairwise similarity scoring for candidate pairs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


CandidatePair = tuple[str, str]


class SimilarityScorer(ABC):
    """Abstract base for pairwise record similarity scorers."""

    @abstractmethod
    def score(
        self, records: pd.DataFrame, candidates: list[CandidatePair]
    ) -> dict[CandidatePair, float]:
        """Compute confidence scores for each candidate pair.

        Returns:
            Dict mapping each pair to a score in [0, 1].
            1.0 = certain match; 0.0 = certain non-match.
        """
        ...


class JaccardScorer(SimilarityScorer):
    """Character shingle Jaccard similarity scorer.

    No external dependencies. Works for any string fields.

    Args:
        shingle_size: Character n-gram size for tokenization.
        fields:       Column names to include. None = all string-typed columns.
    """

    def __init__(self, shingle_size: int = 3, fields: list[str] | None = None) -> None:
        self.shingle_size = shingle_size
        self.fields = fields

    def _record_text(self, row: pd.Series) -> str:
        if self.fields:
            parts = [str(row[f]) for f in self.fields if f in row.index and pd.notna(row[f])]
        else:
            parts = [str(v) for v in row.values if pd.notna(v)]
        return " ".join(parts).lower()

    def _shingles(self, text: str) -> set[str]:
        k = self.shingle_size
        if len(text) < k:
            return {text}
        return {text[i : i + k] for i in range(len(text) - k + 1)}

    def score(
        self, records: pd.DataFrame, candidates: list[CandidatePair]
    ) -> dict[CandidatePair, float]:
        cache: dict[str, set[str]] = {}
        for idx in records.index:
            rid = str(idx)
            cache[rid] = self._shingles(self._record_text(records.loc[idx]))

        result: dict[CandidatePair, float] = {}
        for pair in candidates:
            a, b = pair
            sa, sb = cache.get(a, set()), cache.get(b, set())
            union = len(sa | sb)
            result[pair] = len(sa & sb) / union if union else 0.0
        return result


class CosineSimilarityScorer(SimilarityScorer):
    """Embedding-based cosine similarity scorer.

    Encodes all records once via sandx-embed, then scores each pair
    from the precomputed embedding matrix. O(N·D + P) where P = |candidates|.

    Requires the sandx-embed package (``pip install sandx-embed``).

    Args:
        model:  sandx-embed model name (default: "sentence-bert").
        fields: Column names to include in the text representation.
                None = all columns.
    """

    def __init__(self, model: str = "sentence-bert", fields: list[str] | None = None) -> None:
        self.model = model
        self.fields = fields

    def _record_text(self, row: pd.Series) -> str:
        if self.fields:
            parts = [str(row[f]) for f in self.fields if f in row.index and pd.notna(row[f])]
        else:
            parts = [str(v) for v in row.values if pd.notna(v)]
        return " ".join(parts)

    def score(
        self, records: pd.DataFrame, candidates: list[CandidatePair]
    ) -> dict[CandidatePair, float]:
        try:
            from sandx_embed import Encoder  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "CosineSimilarityScorer requires sandx-embed. "
                "Install with: pip install sandx-embed"
            ) from exc

        enc = Encoder(model=self.model)
        ids = [str(idx) for idx in records.index]
        texts = [self._record_text(records.loc[idx]) for idx in records.index]
        vectors = enc.encode(texts, normalize=True)
        id_to_vec: dict[str, np.ndarray] = {rid: vec for rid, vec in zip(ids, vectors)}

        result: dict[CandidatePair, float] = {}
        for pair in candidates:
            a, b = pair
            va, vb = id_to_vec.get(a), id_to_vec.get(b)
            if va is None or vb is None:
                result[pair] = 0.0
            else:
                # Cosine ∈ [-1, 1] for normalized vectors; map to [0, 1]
                cosine = float(np.dot(va, vb))
                result[pair] = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        return result
