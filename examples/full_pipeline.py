"""SandX end-to-end pipeline demo.

    sandx-er    -> entity resolution: collapse 15 noisy records -> 5 entities
    sandx-graph -> knowledge graph + consensus scoring over resolved entities

Install:
    pip install sandx-er sandx-graph

Run:
    python -m examples.full_pipeline
"""

from __future__ import annotations

import time

import pandas as pd

from sandx_er import EntityResolver

try:
    from sandx_graph import ConsensusEngine, GraphBuilder
except ImportError as exc:
    raise ImportError(
        "This demo requires sandx-graph.\n"
        "Install with: pip install sandx-graph"
    ) from exc


# ---------------------------------------------------------------------------
# Synthetic dataset
# 5 companies, each recorded 3 times with real-world name variation noise.
# Ground truth: records within each block resolve to one entity.
# ---------------------------------------------------------------------------

RECORDS = [
    {"id": "r01", "name": "Apple Inc.",            "hq": "Cupertino CA"},
    {"id": "r02", "name": "Apple, Inc.",            "hq": "Cupertino, CA"},
    {"id": "r03", "name": "Apple Inc",              "hq": "Cupertino"},
    {"id": "r04", "name": "Microsoft Corp.",        "hq": "Redmond WA"},
    {"id": "r05", "name": "Microsoft Corp",         "hq": "Redmond, WA"},
    {"id": "r06", "name": "Microsoft Corporation",  "hq": "Redmond"},
    {"id": "r07", "name": "Google LLC",             "hq": "Mountain View CA"},
    {"id": "r08", "name": "Google, LLC",            "hq": "Mountain View, CA"},
    {"id": "r09", "name": "Google Inc.",            "hq": "Mountain View"},
    {"id": "r10", "name": "Amazon.com Inc.",        "hq": "Seattle WA"},
    {"id": "r11", "name": "Amazon.com, Inc.",       "hq": "Seattle, WA"},
    {"id": "r12", "name": "Amazon Inc.",            "hq": "Seattle"},
    {"id": "r13", "name": "Meta Platforms Inc.",    "hq": "Menlo Park CA"},
    {"id": "r14", "name": "Meta Platforms, Inc.",   "hq": "Menlo Park, CA"},
    {"id": "r15", "name": "Meta Platforms",         "hq": "Menlo Park"},
]

# Cross-entity relationships (would come from a knowledge base or co-occurrence
# data in production; explicit here for a self-contained demo).
RELATIONS = [
    ("Apple",     "Microsoft", 0.82),
    ("Apple",     "Google",    0.78),
    ("Google",    "Amazon",    0.75),
    ("Amazon",    "Microsoft", 0.71),
    ("Meta",      "Google",    0.68),
]

_KNOWN = ["Apple", "Microsoft", "Google", "Amazon", "Meta"]
_SEP = "-" * 54


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _display_name(record_ids: list[str], df: pd.DataFrame) -> str:
    names = [df.loc[rid, "name"] for rid in record_ids if rid in df.index]
    return min(names, key=len)


def _company_key(record_ids: list[str], df: pd.DataFrame) -> str:
    for rid in record_ids:
        name = df.loc[rid, "name"].lower()
        for k in _KNOWN:
            if k.lower() in name:
                return k
    return "Unknown"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    print(f"\n{'=' * 54}")
    print("  SandX Full Pipeline Demo")
    print(f"{'=' * 54}")
    print("  sandx-er    -> entity resolution")
    print("  sandx-graph -> knowledge graph + consensus")
    print(f"{'=' * 54}\n")
    print(f"  Input: {len(RECORDS)} records  "
          f"({len(RECORDS) // 3} known entities, 3 records each)\n")

    df = pd.DataFrame(RECORDS).set_index("id")

    # ── Stage 1: Entity Resolution ──────────────────────────────────────
    print(_SEP)
    print("  Stage 1 - Entity Resolution  (sandx-er)")
    print(_SEP)
    print("  blocking   : LSH + MinHash")
    print("  matching   : Jaccard similarity")
    print("  clustering : connected components")
    print("  threshold  : 0.35\n")

    er = EntityResolver(
        blocking="lsh",
        similarity="jaccard",
        clustering="connected_components",
        threshold=0.35,
    )

    t0 = time.perf_counter()
    result = er.resolve(df)
    elapsed = time.perf_counter() - t0

    merged = sorted(
        [c for c in result.clusters if c.size > 1],
        key=lambda c: c.size, reverse=True,
    )

    print(f"  Resolved {result.n_records} records -> {len(merged)} entities"
          f"  [{elapsed * 1000:.0f}ms]\n")

    display_names: dict[str, str] = {}
    company_keys: dict[str, str] = {}

    for cluster in merged:
        name = _display_name(cluster.record_ids, df)
        key = _company_key(cluster.record_ids, df)
        display_names[cluster.canonical_id] = name
        company_keys[cluster.canonical_id] = key
        record_names = "  /  ".join(df.loc[r, "name"] for r in cluster.record_ids)
        print(f"  {name:<28}  conf={cluster.confidence:.2f}  size={cluster.size}")
        print(f"    {record_names}")

    # ── Stage 2: Knowledge Graph ─────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  Stage 2 - Knowledge Graph  (sandx-graph)")
    print(_SEP)
    print("  nodes : one per resolved entity")
    print("  edges : sector relationship weights (knowledge base)\n")

    key_to_cid = {v: k for k, v in company_keys.items()}
    rel_edges = [
        (key_to_cid[s], key_to_cid[t], w)
        for s, t, w in RELATIONS
        if key_to_cid.get(s) and key_to_cid.get(t)
    ]

    graph = GraphBuilder().from_resolution(result, edges=rel_edges)
    for cid, name in display_names.items():
        if cid in graph.nodes:
            graph.nodes[cid]["name"] = name
    print(f"  {graph}\n")
    print("  Edges:")
    for src, tgt, w in sorted(rel_edges, key=lambda e: e[2], reverse=True):
        src_n = display_names.get(src, src[:8])
        tgt_n = display_names.get(tgt, tgt[:8])
        bar = "#" * int(w * 30)
        print(f"  {src_n:<22} -- {tgt_n:<22}  {w:.2f}  {bar}")

    # ── Stage 3: Consensus Scores ────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  Stage 3 - Consensus Scores  (sandx-graph)")
    print(_SEP)
    print("  depth            : 1  (direct neighbors)")
    print("  support_threshold: 0.60\n")

    engine = ConsensusEngine(graph, support_threshold=0.60)
    scores = engine.compute_all(depth=1)
    summary = engine.summary(depth=1)

    for s in sorted(scores.values(), key=lambda x: x.score, reverse=True):
        name = display_names.get(s.node_id, s.node_id[:8])
        bar = "#" * int(s.score * 30)
        print(f"  {name:<28}  score={s.score:.3f}  "
              f"support={len(s.supporting_neighbors)}  "
              f"conflict={len(s.conflicting_neighbors)}  {bar}")

    print(f"\n  Summary  "
          f"mean={summary['mean']:.3f}  "
          f"median={summary['median']:.3f}  "
          f"std={summary['std']:.3f}  "
          f"min={summary['min']:.3f}  "
          f"max={summary['max']:.3f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 54}")
    print("  Pipeline complete.")
    print(f"  {result.n_records} raw records -> "
          f"{len(merged)} resolved entities -> "
          f"{graph.n_nodes} nodes / {graph.n_edges} edges")
    print(f"{'=' * 54}\n")
    print("  Upgrade: use semantic (embedding) blocking with sandx-embed:")
    print("    pip install sandx-embed")
    print("    er = EntityResolver(blocking='embedding',")
    print("                        similarity='embedding', threshold=0.85)")
    print()


if __name__ == "__main__":
    run()
