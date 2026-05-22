"""Abt-Buy benchmark for sandx-er.

Evaluates the entity resolution pipeline on the Abt-Buy product matching dataset.
Downloads data from the DeepMatcher benchmark repository (public, stable).

Dataset:
    - tableA.csv: 1081 Abt.com product records
    - tableB.csv: 1092 Buy.com product records
    - test.csv:   ~300 labeled test pairs (matched / not matched)

Usage:
    python -m benchmarks.abt_buy
    python -m benchmarks.abt_buy --blocking lsh --similarity jaccard --threshold 0.3
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from dataclasses import dataclass

import pandas as pd

from sandx_er import EntityResolver


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_BASE = "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/data/Structured/Abt-Buy"

URLS = {
    "tableA": f"{_BASE}/tableA.csv",
    "tableB": f"{_BASE}/tableb.csv",  # note: lowercase b in repo
    "test":   f"{_BASE}/test.csv",
}


def _fetch(url: str) -> pd.DataFrame:
    print(f"  Downloading {url} ...", end=" ", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "sandx-er-benchmark/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(data))
    print(f"({len(df)} rows)")
    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and return (tableA, tableB, test_pairs)."""
    print("Fetching Abt-Buy dataset...")
    a = _fetch(URLS["tableA"])
    b = _fetch(URLS["tableB"])
    test = _fetch(URLS["test"])
    return a, b, test


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
    blocking_time_s: float
    total_time_s: float

    def __str__(self) -> str:
        return (
            f"Precision : {self.precision:.4f}\n"
            f"Recall    : {self.recall:.4f}\n"
            f"F1        : {self.f1:.4f}\n"
            f"TP / Pred / True : {self.n_true_positives} / "
            f"{self.n_predicted_matches} / {self.n_true_matches}\n"
            f"Total time: {self.total_time_s:.1f}s"
        )


def evaluate(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    test_pairs: pd.DataFrame,
    blocking: str = "lsh",
    similarity: str = "jaccard",
    threshold: float = 0.3,
    key_field: str | None = None,
) -> BenchmarkResult:
    """Run the ER pipeline on Abt-Buy and return precision/recall/F1.

    Strategy: merge tableA and tableB into a single record set, resolve,
    then evaluate against the ground-truth test pair labels.
    """
    # Prepare combined record set with source prefix on IDs
    a = table_a.copy()
    b = table_b.copy()
    a.index = "A_" + a["id"].astype(str)
    b.index = "B_" + b["id"].astype(str)

    # Use name + description as matching fields
    for df in (a, b):
        for col in ["id"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    records = pd.concat([a, b])

    # Ground-truth matched pairs from test split
    ground_truth: set[tuple[str, str]] = set()
    for _, row in test_pairs.iterrows():
        if row.get("label", 1) == 1:  # label=1 means match
            aid = f"A_{int(row['ltable_id'])}"
            bid = f"B_{int(row['rtable_id'])}"
            ground_truth.add((min(aid, bid), max(aid, bid)))

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
        blocking_time_s=0.0,  # not separately tracked
        total_time_s=total_time,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Abt-Buy ER benchmark for sandx-er")
    parser.add_argument("--blocking", default="lsh", choices=["lsh", "snm"],
                        help="Blocking method (default: lsh)")
    parser.add_argument("--similarity", default="jaccard", choices=["jaccard"],
                        help="Similarity scoring method (default: jaccard)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Match confidence threshold (default: 0.3)")
    parser.add_argument("--key-field", default=None,
                        help="Key field for SNM blocking")
    args = parser.parse_args()

    try:
        table_a, table_b, test_pairs = load_data()
    except Exception as exc:
        print(f"Error downloading data: {exc}", file=sys.stderr)
        print("Check your internet connection and try again.", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nRunning benchmark: blocking={args.blocking}, "
        f"similarity={args.similarity}, threshold={args.threshold}"
    )

    bm = evaluate(
        table_a, table_b, test_pairs,
        blocking=args.blocking,
        similarity=args.similarity,
        threshold=args.threshold,
        key_field=args.key_field,
    )

    print("\n--- Abt-Buy Benchmark Results ---")
    print(bm)


if __name__ == "__main__":
    main()
