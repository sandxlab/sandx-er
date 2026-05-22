"""Blocking pipeline — candidate pair generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd


CandidatePair = tuple[str, str]


class BlockingMethod(ABC):
    """Base class for all blocking strategies."""

    @abstractmethod
    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        """Yield (record_id_a, record_id_b) candidate pairs."""
        ...


class LSHBlocking(BlockingMethod):
    """Locality-Sensitive Hashing blocking.

    Groups records into buckets based on hash signatures of their content.
    Pairs within the same bucket are emitted as candidates.
    Complexity: O(N) expected, tunable by band/row parameters.
    """

    def __init__(self, n_bands: int = 20, n_rows: int = 5) -> None:
        self.n_bands = n_bands
        self.n_rows = n_rows

    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        raise NotImplementedError("Phase 2")


class EmbeddingANNBlocking(BlockingMethod):
    """Approximate Nearest Neighbor blocking via embedding similarity.

    Requires sandx-embed. Encodes records into dense vectors and retrieves
    the k nearest neighbors for each record as candidates.
    """

    def __init__(self, k: int = 10, model: str = "default") -> None:
        self.k = k
        self.model = model

    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        raise NotImplementedError("Phase 2")


class SortedNeighborhoodBlocking(BlockingMethod):
    """Sorted Neighborhood Method blocking.

    Sorts records by a blocking key and slides a window of size w,
    emitting all pairs within each window as candidates.
    """

    def __init__(self, key_field: str, window_size: int = 3) -> None:
        self.key_field = key_field
        self.window_size = window_size

    def generate_candidates(self, records: pd.DataFrame) -> Iterator[CandidatePair]:
        raise NotImplementedError("Phase 2")
