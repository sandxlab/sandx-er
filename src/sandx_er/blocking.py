"""Blocking pipeline — candidate pair generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import pandas as pd


CandidatePair = tuple[str, str]


class BlockingMethod(ABC):
    """Base class for all blocking strategies."""

    @abstractmethod
    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        """Yield (record_id_a, record_id_b) candidate pairs without duplicates."""
        ...


class LSHBlocking(BlockingMethod):
    """MinHash LSH blocking — Jaccard-similarity-based candidate generation.

    Decomposes each record into character shingles, computes a MinHash signature,
    and uses band hashing to find records with high Jaccard similarity in O(N)
    expected time.

    Args:
        n_bands:     Number of LSH bands. More bands → higher recall, more candidates.
        n_rows:      Rows per band. Similarity threshold ≈ (1/n_bands)^(1/n_rows).
        shingle_size: Character n-gram size for tokenization.
    """

    def __init__(self, n_bands: int = 20, n_rows: int = 5, shingle_size: int = 3) -> None:
        self.n_bands = n_bands
        self.n_rows = n_rows
        self.shingle_size = shingle_size
        self._n_hashes = n_bands * n_rows
        # Deterministic random hash parameters (universal hashing)
        rng = np.random.default_rng(42)
        _p = (1 << 31) - 1  # Mersenne prime
        self._a = rng.integers(1, _p, size=self._n_hashes, dtype=np.int64)
        self._b = rng.integers(0, _p, size=self._n_hashes, dtype=np.int64)
        self._p = np.int64(_p)

    def _record_text(self, row: pd.Series) -> str:
        return " ".join(str(v) for v in row.values if pd.notna(v)).lower()

    def _shingles(self, text: str) -> list[int]:
        k = self.shingle_size
        if not text:
            return [hash("")]
        return [hash(text[i : i + k]) for i in range(max(1, len(text) - k + 1))]

    def _minhash(self, shingles: list[int]) -> np.ndarray:
        if not shingles:
            return np.zeros(self._n_hashes, dtype=np.int64)
        sh = np.array(shingles, dtype=np.int64)
        # shape: (n_hashes, n_shingles)
        hashed = (self._a[:, None] * sh[None, :] + self._b[:, None]) % self._p
        return hashed.min(axis=1)

    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        # Compute signatures
        ids: list[str] = [str(idx) for idx in records.index]
        sigs: list[np.ndarray] = []
        for idx in records.index:
            text = self._record_text(records.loc[idx])
            sigs.append(self._minhash(self._shingles(text)))

        sig_matrix = np.array(sigs)  # (N, n_hashes)
        seen: set[CandidatePair] = set()

        for band in range(self.n_bands):
            start, end = band * self.n_rows, (band + 1) * self.n_rows
            band_sigs = sig_matrix[:, start:end]  # (N, n_rows)
            buckets: dict[tuple, list[int]] = {}
            for i, row_sig in enumerate(band_sigs):
                key = tuple(row_sig.tolist())
                buckets.setdefault(key, []).append(i)

            for bucket in buckets.values():
                if len(bucket) < 2:
                    continue
                for j in range(len(bucket)):
                    for k in range(j + 1, len(bucket)):
                        a, b = ids[bucket[j]], ids[bucket[k]]
                        pair: CandidatePair = (min(a, b), max(a, b))
                        if pair not in seen:
                            seen.add(pair)
                            yield pair


class SortedNeighborhoodBlocking(BlockingMethod):
    """Sorted Neighborhood Method blocking.

    Sorts records by ``key_field`` and emits all pairs within a sliding
    window of size ``window_size``. Linear in N for fixed window size.

    Args:
        key_field:   Column name to sort by.
        window_size: Number of adjacent records to compare.
    """

    def __init__(self, key_field: str, window_size: int = 3) -> None:
        self.key_field = key_field
        self.window_size = window_size

    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        if self.key_field not in records.columns:
            raise ValueError(
                f"Blocking key '{self.key_field}' not found in records. "
                f"Available columns: {list(records.columns)}"
            )

        sorted_idx = (
            records[self.key_field]
            .fillna("")
            .astype(str)
            .argsort()
            .values
        )
        ids = [str(records.index[i]) for i in sorted_idx]
        seen: set[CandidatePair] = set()
        n = len(ids)

        for i in range(n):
            for j in range(i + 1, min(i + self.window_size, n)):
                a, b = ids[i], ids[j]
                pair: CandidatePair = (min(a, b), max(a, b))
                if pair not in seen:
                    seen.add(pair)
                    yield pair


class EmbeddingANNBlocking(BlockingMethod):
    """Embedding-based Approximate Nearest Neighbor blocking.

    Encodes all records as dense vectors (via sandx-embed) and retrieves
    the k nearest neighbors for each record as candidate pairs.
    Captures semantic similarity that key-based blocking misses.

    Requires the sandx-embed package (``pip install sandx-embed``).

    Args:
        k:     Number of nearest neighbors per record.
        model: sandx-embed encoder name (default: "sentence-bert").
        fields: Column names to include. None = all columns.
    """

    def __init__(
        self, k: int = 10, model: str = "sentence-bert", fields: list[str] | None = None
    ) -> None:
        self.k = k
        self.model = model
        self.fields = fields

    def _record_text(self, row: pd.Series) -> str:
        if self.fields:
            parts = [str(row[f]) for f in self.fields if f in row.index and pd.notna(row[f])]
        else:
            parts = [str(v) for v in row.values if pd.notna(v)]
        return " ".join(parts)

    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        try:
            from sandx_embed import Encoder, VectorIndex  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "EmbeddingANNBlocking requires sandx-embed. "
                "Install with: pip install sandx-embed"
            ) from exc

        enc = Encoder(model=self.model)
        ids = [str(idx) for idx in records.index]
        texts = [self._record_text(records.loc[idx]) for idx in records.index]

        vectors = enc.encode(texts, normalize=True)
        idx_obj = VectorIndex(method="hnsw", metric="cosine")
        idx_obj.build(vectors, ids)

        seen: set[CandidatePair] = set()
        for rid, vec in zip(ids, vectors):
            result = idx_obj.query(vec, k=self.k + 1)  # +1: self is always top-1
            for neighbor_id in result.ids:
                if neighbor_id == rid:
                    continue
                pair: CandidatePair = (min(rid, neighbor_id), max(rid, neighbor_id))
                if pair not in seen:
                    seen.add(pair)
                    yield pair
