"""Fodors-Zagats restaurant benchmark for sandx-er.

Record linkage of restaurant listings across Fodors and Zagats databases.
Evaluates the sandx-er pipeline using an external data file.

Dataset (Fodors-Zagats):
    - tableA: 533 Fodors restaurant records
    - tableB: 331 Zagats restaurant records
    - links:  110 ground-truth matching pairs
    Data: Magellan ER benchmark collection (Köpcke & Rahm, 2010).

Usage:
    python -m benchmarks.restaurant --data-path /path/to/restaurant.csv
    python -m benchmarks.restaurant --data-path /path/to/restaurant.csv --blocking snm --key-field name
    python -m benchmarks.restaurant --data-path /path/to/restaurant.csv --threshold 0.4
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
    """Load Fodors-Zagats from a combined CSV with ground-truth cluster_id column.

    Expected CSV columns: id, name, addr, city, phone, type, source, cluster_id
    - source values: "fodors" or "zagats"
    - id values are prefixed A_ (Fodors) or B_ (Zagats)
    - Records sharing the same cluster_id across sources are true matches
    """
    df = pd.read_csv(data_path)
    required = {"id", "name", "addr", "city", "phone", "type", "source", "cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    fodors = df[df["source"] == "fodors"].set_index("id")
    zagats = df[df["source"] == "zagats"].set_index("id")

    print(f"  Fodors (tableA): {len(fodors):,} records")
    print(f"  Zagats (tableB): {len(zagats):,} records")

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
    return fodors, zagats, ground_truth


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
            f"\n--- Fodors-Zagats Benchmark Results ---\n"
            f"Records   : {self.n_records:,} total\n"
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1        : {self.f1:.4f}\n"
            f"TP / Pred / True : {self.n_true_positives:,} / "
            f"{self.n_predicted_matches:,} / {self.n_true_matches:,}\n"
            f"Time      : {self.total_time_s:.1f}s"
        )


def evaluate(
    fodors: pd.DataFrame,
    zagats: pd.DataFrame,
    ground_truth: set[tuple[str, str]],
    blocking: str = "lsh",
    similarity: str = "jaccard",
    threshold: float = 0.3,
    key_field: str | None = None,
) -> BenchmarkResult:
    feature_cols = ["name", "addr", "city", "phone", "type"]
    a = fodors[[c for c in feature_cols if c in fodors.columns]].copy()
    b = zagats[[c for c in feature_cols if c in zagats.columns]].copy()
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
    parser = argparse.ArgumentParser(description="Fodors-Zagats restaurant ER benchmark for sandx-er")
    parser.add_argument("--data-path", required=True,
                        help="Path to restaurant.csv (combined Fodors+Zagats with cluster_id column)")
    parser.add_argument("--blocking", default="lsh", choices=["lsh", "snm"],
                        help="Blocking method (default: lsh)")
    parser.add_argument("--similarity", default="jaccard", choices=["jaccard"],
                        help="Similarity scoring (default: jaccard)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Match threshold (default: 0.3)")
    parser.add_argument("--key-field", default=None,
                        help="Key field for SNM blocking (e.g. name)")
    args = parser.parse_args()

    print(f"Loading Fodors-Zagats from {args.data_path} ...")
    try:
        fodors, zagats, ground_truth = load_data(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nRunning: blocking={args.blocking}, "
        f"similarity={args.similarity}, threshold={args.threshold}"
    )

    bm = evaluate(
        fodors, zagats, ground_truth,
        blocking=args.blocking,
        similarity=args.similarity,
        threshold=args.threshold,
        key_field=args.key_field,
    )
    print(bm)


if __name__ == "__main__":
    main()
