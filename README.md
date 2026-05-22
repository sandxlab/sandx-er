# sandx-er

**Entity Resolution infrastructure for fragmented, noisy, large-scale datasets.**

Part of the [SandX Lab](https://github.com/sandxlab) computational infrastructure ecosystem.

---

## What It Does

`sandx-er` resolves the identity of real-world entities across one or more datasets where the same entity may appear as multiple, inconsistent, or duplicate records. It provides a modular, composable pipeline:

```
Raw records → Blocking → Matching → Clustering → Resolved identity graph
```

Each stage is independently configurable. Outputs carry probabilistic confidence scores, not binary decisions.

## Status

> **Phase 1 — Architecture & Foundations**
> Core engineering begins in Phase 2. This repository establishes the package structure, API contract, and benchmark targets.

| Component | Status |
|-----------|--------|
| `sandx_er.resolver` — EntityResolver API | Skeleton |
| `sandx_er.blocking` — LSH, SNM, embedding-based | Skeleton |
| `sandx_er.matching` — similarity scoring | Skeleton |
| `sandx_er.clustering` — connected components, correlation | Skeleton |
| Python SDK on PyPI | Planned (Phase 2) |
| Benchmarks — Abt-Buy, DBLP-ACM, Cora | Planned (Phase 2) |

## Installation

```bash
# Phase 2 (planned)
pip install sandx-er
```

## Quick Start (planned API)

```python
from sandx_er import EntityResolver

er = EntityResolver(
    blocking="lsh",        # locality-sensitive hashing
    similarity="embedding", # sandx-embed powered matching
    threshold=0.85
)

result = er.resolve(records)

for cluster in result.clusters:
    print(cluster.canonical_id, cluster.size, cluster.confidence)
```

## Architecture

```
sandx_er/
├── resolver.py     # EntityResolver — main pipeline orchestrator
├── blocking.py     # Blocking pipeline: LSH, SNM, embedding ANN
├── matching.py     # Similarity scoring: feature-based, embedding cosine, learned
└── clustering.py   # Cluster reconciliation: connected components, correlation clustering
```

**Depends on:** [`sandx-embed`](https://github.com/sandxlab/sandx-embed) for embedding-based blocking and matching.

## Benchmarks

Target benchmark datasets (Phase 2):

| Dataset | Domain | Records | Pairs |
|---------|--------|---------|-------|
| Abt-Buy | E-commerce | 1,082 / 1,092 | 1,097 matches |
| DBLP-ACM | Academic citations | 2,616 / 2,294 | 2,224 matches |
| DBLP-Scholar | Academic citations | 2,616 / 64,263 | 5,347 matches |
| Cora | Research papers | 1,295 | deduplication |

All benchmark results will be version-tagged and fully reproducible from public data.

## Related

- [`sandx-embed`](https://github.com/sandxlab/sandx-embed) — embedding infrastructure (shared dependency)
- [`sandx-graph`](https://github.com/sandxlab/sandx-graph) — graph intelligence over resolved entities
- [`sandx-compute`](https://github.com/sandxlab/sandx-compute) — distributed compute orchestration

## License

Apache 2.0 — see [LICENSE](LICENSE)
