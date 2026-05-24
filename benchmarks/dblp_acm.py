"""DBLP-ACM benchmark for sandx-er.

Record linkage of academic publications across DBLP and ACM databases.
Evaluates the sandx-er pipeline using an external data file (no built-in download).

Dataset (DBLP-ACM):
    - tableA: 2,616 DBLP publication records
    - tableB: 2,294 ACM publication records
    - links:  2,224 ground-truth matching pairs (same paper in both databases)
    Data: Magellan ER benchmark collection (Köpcke & Rahm, 2010).
      Obtain the CSV at: https://github.com/anhaidgroup/py_entitymatching

Usage:
    python -m benchmarks.dblp_acm --data-path /path/to/dblp_acm.csv
    python -m benchmarks.dblp_acm --data-path /path/to/dblp_acm.csv --blocking snm --key-field title
    python -m benchmarks.dblp_acm --data-path /path/to/dblp_acm.csv --threshold 0.4
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

def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, set[tuple[str, str]]]:
    """Load DBLP-ACM from a combined CSV with ground-truth cluster_id column.

    Expected CSV columns: id, title, authors, venue, year, source, cluster_id
    - id values are prefixed A_ (DBLP) or B_ (ACM)
    - Records sharing the same cluster_id across sources are true matches
    """
    df = pd.read_csv(data_path)
    required = {"id", "title", "authors", "venue", "year", "source", "cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    dblp = df[df["source"] == "dblp"].set_index("id")
    acm = df[df["source"] == "acm"].set_index("id")

    print(f"  DBLP (tableA): {len(dblp):,} records")
    print(f"  ACM  (tableB): {len(acm):,} records")

    # Build ground truth: for each cluster_id, generate all cross-table (A, B) pairs
    cluster_to_ids: dict[int, list[str]] = {}
    for row in df.itertuples(index=False):
        cluster_to_ids.setdefault(row.cluster_id, []).append(row.id)

    ground_truth: set[tuple[str, str]] = set()
    for ids in cluster_to_ids.values():
        a_ids = [i for i in ids if i.startswith("A_")]
        b_ids = [i for i in ids if i.startswith("B_")]
        for aid in a_ids:
            for bid in b_ids:
                ground_truth.add((min(aid, bid), max(aid, bid)))

    print(f"  True matches : {len(ground_truth):,}")
    return dblp, acm, ground_truth


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
            f"\n--- DBLP-ACM Benchmark Results ---\n"
            f"Records   : {self.n_records:,} ({self.n_records // 2:,} per table approx.)\n"
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1        : {self.f1:.4f}\n"
            f"TP / Pred / True : {self.n_true_positives:,} / "
            f"{self.n_predicted_matches:,} / {self.n_true_matches:,}\n"
            f"Time      : {self.total_time_s:.1f}s"
        )


def evaluate(
    dblp: pd.DataFrame,
    acm: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
    blocking: str = "lsh",
    similarity: str = "jaccard",
    threshold: float = 0.3,
    key_field: str | None = None,
) -> BenchmarkResult:
    feature_cols = ["title", "authors", "venue", "year"]
    a = dblp[feature_cols].copy()
    b = acm[feature_cols].copy()
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
    parser = argparse.ArgumentParser(description="DBLP-ACM ER benchmark for sandx-er")
    parser.add_argument("--data-path", required=True,
                        help="Path to dblp_acm.csv (combined DBLP+ACM with cluster_id column)")
    parser.add_argument("--blocking", default="lsh", choices=["lsh", "snm"],
                        help="Blocking method (default: lsh)")
    parser.add_argument("--similarity", default="jaccard", choices=["jaccard"],
                        help="Similarity scoring (default: jaccard)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Match threshold (default: 0.3)")
    parser.add_argument("--key-field", default=None,
                        help="Key field for SNM blocking (e.g. title)")
    args = parser.parse_args()

    print(f"Loading DBLP-ACM from {args.data_path} ...")
    try:
        dblp, acm, ground_truth = load_data(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nRunning: blocking={args.blocking}, "
        f"similarity={args.similarity}, threshold={args.threshold}"
    )

    bm = evaluate(
        dblp, acm, ground_truth,
        blocking=args.blocking,
        similarity=args.similarity,
        threshold=args.threshold,
        key_field=args.key_field,
    )
    print(bm)


if __name__ == "__main__":
    main()
