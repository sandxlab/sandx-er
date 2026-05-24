"""Febrl4 benchmark for sandx-er.

Evaluates the entity resolution pipeline on the Febrl4 person record linkage
dataset, included in the ``recordlinkage`` package (no network required).

Dataset (Febrl4):
    - tableA: 5,000 synthetic Australian person records
    - tableB: 5,000 records with known noise (typos, transpositions, missing)
    - links:  5,000 ground-truth matching pairs (1:1)

Install: pip install recordlinkage

Usage:
    python -m benchmarks.abt_buy
    python -m benchmarks.abt_buy --blocking lsh --threshold 0.3
    python -m benchmarks.abt_buy --blocking snm --key-field surname
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import pandas as pd

from sandx_er import EntityResolver


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, str]]]:
    """Load Febrl4 via the recordlinkage package."""
    try:
        from recordlinkage.datasets import load_febrl4  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "recordlinkage is required for this benchmark.\n"
            "Install with: pip install recordlinkage"
        ) from exc

    print("Loading Febrl4 dataset (built-in, no download required)...")
    dfA, dfB, links = load_febrl4(return_links=True)
    print(f"  tableA: {len(dfA)} records")
    print(f"  tableB: {len(dfB)} records")
    print(f"  True matches: {len(links)}")

    # Build ground-truth set as (A_id, B_id) pairs
    ground_truth: set[tuple[str, str]] = set()
    for idx_a, idx_b in links:
        a_id = f"A_{idx_a}"
        b_id = f"B_{idx_b}"
        ground_truth.add((min(a_id, b_id), max(a_id, b_id)))

    return dfA, dfB, ground_truth


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    precision: float
    recall: float
    f1: float
    n_true_matches: int
    n_predicted_matches: int
    n_true_positives: int
    total_time_s: float
    n_records: int

    def __str__(self) -> str:
        return (
            f"\n--- Febrl4 Benchmark Results ---\n"
            f"Records   : {self.n_records:,} ({self.n_records // 2:,} per table)\n"
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1        : {self.f1:.4f}\n"
            f"TP / Pred / True : {self.n_true_positives} / "
            f"{self.n_predicted_matches:,} / {self.n_true_matches:,}\n"
            f"Time      : {self.total_time_s:.1f}s"
        )


def evaluate(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
    blocking: str = "lsh",
    similarity: str = "jaccard",
    threshold: float = 0.3,
    key_field: str | None = None,
) -> BenchmarkResult:
    # Prefix IDs to distinguish sources
    a = dfA.copy()
    b = dfB.copy()
    a.index = "A_" + a.index.astype(str)
    b.index = "B_" + b.index.astype(str)
    records = pd.concat([a, b])

    er = EntityResolver(
        blocking=blocking,
        similarity=similarity,
        clustering="connected_components",
        threshold=threshold,
        key_field=key_field,
    )

    t0 = time.perf_counter()
    result = er.resolve(records)
    total_time = time.perf_counter() - t0

    # Extract cross-table pairs from resolved clusters
    predicted: set[tuple[str, str]] = set()
    for cluster in result.clusters:
        a_ids = [r for r in cluster.record_ids if r.startswith("A_")]
        b_ids = [r for r in cluster.record_ids if r.startswith("B_")]
        for aid in a_ids:
            for bid in b_ids:
                predicted.add((min(aid, bid), max(aid, bid)))

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return BenchmarkResult(
        precision=precision,
        recall=recall,
        f1=f1,
        n_true_matches=len(ground_truth),
        n_predicted_matches=len(predicted),
        n_true_positives=tp,
        total_time_s=total_time,
        n_records=len(records),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Febrl4 ER benchmark for sandx-er")
    parser.add_argument("--blocking", default="lsh", choices=["lsh", "snm"],
                        help="Blocking method (default: lsh)")
    parser.add_argument("--similarity", default="jaccard", choices=["jaccard"],
                        help="Similarity scoring (default: jaccard)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Match threshold (default: 0.3)")
    parser.add_argument("--key-field", default=None,
                        help="Key field for SNM blocking (e.g. surname)")
    args = parser.parse_args()

    try:
        dfA, dfB, ground_truth = load_data()
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nRunning: blocking={args.blocking}, "
        f"similarity={args.similarity}, threshold={args.threshold}"
    )

    bm = evaluate(
        dfA, dfB, ground_truth,
        blocking=args.blocking,
        similarity=args.similarity,
        threshold=args.threshold,
        key_field=args.key_field,
    )
    print(bm)


if __name__ == "__main__":
    main()
