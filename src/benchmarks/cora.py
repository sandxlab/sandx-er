"""Cora citation deduplication benchmark for sandx-er.

Single-table deduplication task — no tableA/tableB split. Ground truth: clusters
of records that cite the same paper. Evaluation is pair-based F1 (enumerate all
within-cluster pairs, compute precision/recall against predicted clusters).

Dataset (Cora):
    1,879 citation records across ~900 unique papers.
    Columns expected: id (or rec_id), author, title, venue, year, cluster_id.
    Data: Magellan ER benchmark collection / DeepMatcher benchmarks.
    Download: https://github.com/anhaidgroup/py_entitymatching

Usage:
    python -m benchmarks.cora --data-path /path/to/cora.csv
    python -m benchmarks.cora --data-path /path/to/cora.csv --blocking snm --key-field title
    python -m benchmarks.cora --data-path /path/to/cora.csv --threshold 0.4
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

# Candidate column names for the record ID and cluster ID, tried in order
_ID_CANDIDATES      = ["id", "rec_id", "_id", "record_id"]
_CLUSTER_CANDIDATES = ["cluster_id", "entity_id", "label", "entity"]


def _find_col(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find {role} column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_data(data_path: str) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    """Load Cora from a CSV with a cluster_id (or equivalent) column.

    Returns:
        records:      DataFrame indexed by record ID, feature columns only.
        ground_truth: Set of (id_a, id_b) pairs sharing the same cluster.
    """
    df = pd.read_csv(data_path, dtype=str).fillna("")

    id_col      = _find_col(df, _ID_CANDIDATES, "record ID")
    cluster_col = _find_col(df, _CLUSTER_CANDIDATES, "cluster ID")

    df = df.set_index(id_col)

    print(f"  Records  : {len(df):,}")
    print(f"  ID col   : {id_col!r}  |  cluster col: {cluster_col!r}")

    # Build ground-truth pairs from cluster assignments
    cluster_to_ids: dict[str, list[str]] = {}
    for rec_id, cluster_id in df[cluster_col].items():
        cluster_to_ids.setdefault(str(cluster_id), []).append(str(rec_id))

    n_clusters = sum(1 for ids in cluster_to_ids.values() if len(ids) > 1)
    ground_truth: set[tuple[str, str]] = set()
    for ids in cluster_to_ids.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ground_truth.add((min(a, b), max(a, b)))

    print(f"  Clusters : {len(cluster_to_ids):,} ({n_clusters:,} with >1 record)")
    print(f"  True pairs: {len(ground_truth):,}")

    # Feature columns: everything except the cluster column
    feature_cols = [c for c in df.columns if c != cluster_col]
    return df[feature_cols], ground_truth


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    precision: float
    recall: float
    f1: float
    n_true_pairs: int
    n_predicted_pairs: int
    n_true_positives: int
    total_time_s: float
    n_records: int

    def __str__(self) -> str:
        return (
            f"\n--- Cora Benchmark Results ---\n"
            f"Records   : {self.n_records:,}\n"
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1        : {self.f1:.4f}\n"
            f"TP / Pred / True : {self.n_true_positives:,} / "
            f"{self.n_predicted_pairs:,} / {self.n_true_pairs:,}\n"
            f"Time      : {self.total_time_s:.1f}s"
        )


def evaluate(
    records: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
    blocking: str = "lsh",
    similarity: str = "jaccard",
    threshold: float = 0.5,
    key_field: str | None = None,
) -> BenchmarkResult:
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

    # Convert predicted clusters to within-cluster pairs
    predicted: set[tuple[str, str]] = set()
    for cluster in result.clusters:
        ids = cluster.record_ids
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                predicted.add((min(a, b), max(a, b)))

    tp = len(predicted & ground_truth)
    precision = tp / len(predicted) if predicted else 0.0
    recall    = tp / len(ground_truth) if ground_truth else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return BenchmarkResult(
        precision=precision,
        recall=recall,
        f1=f1,
        n_true_pairs=len(ground_truth),
        n_predicted_pairs=len(predicted),
        n_true_positives=tp,
        total_time_s=total_time,
        n_records=len(records),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cora citation deduplication benchmark for sandx-er")
    parser.add_argument("--data-path", required=True,
                        help="Path to cora.csv (with cluster_id column)")
    parser.add_argument("--blocking", default="lsh", choices=["lsh", "snm"],
                        help="Blocking method (default: lsh)")
    parser.add_argument("--similarity", default="jaccard", choices=["jaccard"],
                        help="Similarity scoring (default: jaccard)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Match threshold (default: 0.5)")
    parser.add_argument("--key-field", default=None,
                        help="Key field for SNM blocking (e.g. title)")
    args = parser.parse_args()

    print(f"Loading Cora from {args.data_path} ...")
    try:
        records, ground_truth = load_data(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nRunning: blocking={args.blocking}, "
        f"similarity={args.similarity}, threshold={args.threshold}"
    )

    bm = evaluate(
        records, ground_truth,
        blocking=args.blocking,
        similarity=args.similarity,
        threshold=args.threshold,
        key_field=args.key_field,
    )
    print(bm)


if __name__ == "__main__":
    main()
