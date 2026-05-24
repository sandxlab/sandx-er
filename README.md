# sandx-er

**Entity Resolution infrastructure for fragmented, noisy, large-scale datasets.**

[![CI](https://github.com/sandxlab/sandx-er/actions/workflows/ci.yml/badge.svg)](https://github.com/sandxlab/sandx-er/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [SandX Lab](https://github.com/sandxlab) computational infrastructure ecosystem.

---

## What It Does

`sandx-er` resolves the identity of real-world entities across datasets where the same entity appears as multiple, inconsistent, or duplicate records. Pipeline:

```
Raw records  →  Blocking  →  Matching  →  Clustering  →  Resolved identity graph
                 (LSH,          (Jaccard,    (Connected
                  SNM,           cosine)      components,
                  ANN)                        Correlation)
```

Each stage is independently configurable. Every output carries a probabilistic confidence score — not a binary decision.

## Status

> **v0.1 — Phase 2 active development**

| Component | Status |
|-----------|--------|
| `EntityResolver` — pipeline orchestrator | **Working** |
| `LSHBlocking` — MinHash LSH | **Working** |
| `SortedNeighborhoodBlocking` — SNM | **Working** |
| `EmbeddingANNBlocking` — ANN via sandx-embed | **Working** |
| `JaccardScorer` — character shingle Jaccard | **Working** |
| `CosineSimilarityScorer` — embedding cosine | **Working** |
| `ConnectedComponentsClustering` | **Working** |
| `CorrelationClustering` — Kwik-Cluster | **Working** |
| Febrl4 benchmark | **Working** |
| DBLP-ACM benchmark | **Working** |
| PyPI package | **Working** |

## Installation

```bash
pip install sandx-er
```

Or from source:

```bash
git clone https://github.com/sandxlab/sandx-er
cd sandx-er
pip install -e ".[dev]"
```

For embedding-based blocking and matching:

```bash
pip install "sandx-er[embed]"
```

## Quick Start

```python
import pandas as pd
from sandx_er import EntityResolver

records = pd.DataFrame({
    "name":  ["Acme Corp", "Acme Corp.", "GlobalTech Inc", "Global Tech"],
    "city":  ["Boston",    "Boston",     "New York",       "New York"],
})

er = EntityResolver(
    blocking="lsh",       # MinHash LSH candidate generation
    similarity="jaccard", # character Jaccard similarity scoring
    threshold=0.4,
)

result = er.resolve(records)

print(f"Resolved {result.n_records} records → {result.n_clusters} entities")
for cluster in result.clusters:
    print(f"  {cluster.canonical_id[:8]}  size={cluster.size}  conf={cluster.confidence:.2f}")
    print(f"    records: {cluster.record_ids}")
```

Output:
```
Resolved 4 records → 2 entities
  3f2a1b8c  size=2  conf=0.81
    records: ['0', '1']
  7e9d4c2a  size=2  conf=0.76
    records: ['2', '3']
```

## Pipeline Stages

### Blocking

Reduces O(N²) comparisons to a tractable candidate set.

```python
from sandx_er import LSHBlocking, SortedNeighborhoodBlocking, EmbeddingANNBlocking

# MinHash LSH — works on all string fields, no key required
er = EntityResolver(blocking="lsh")

# Sorted Neighborhood Method — fast, requires a sort key
er = EntityResolver(blocking="snm", key_field="name")

# Embedding ANN — semantic similarity (requires sandx-embed)
er = EntityResolver(blocking="embedding", embed_model="sentence-bert")

# Or pass a custom BlockingMethod instance
er = EntityResolver(blocking=LSHBlocking(n_bands=30, n_rows=4))
```

### Matching

Scores each candidate pair.

```python
from sandx_er import JaccardScorer, CosineSimilarityScorer

er = EntityResolver(similarity="jaccard")               # no deps; fast
er = EntityResolver(similarity="embedding")             # requires sandx-embed
er = EntityResolver(similarity=JaccardScorer(shingle_size=2, fields=["name"]))
```

### Clustering

Reconciles pairwise decisions into globally consistent entity clusters.

```python
er = EntityResolver(clustering="connected_components")  # fast; may over-merge
er = EntityResolver(clustering="correlation")           # slower; corrects transitivity errors
```

## Benchmark — Febrl4

```bash
python -m benchmarks.febrl4                                    # LSH + Jaccard, threshold 0.3
python -m benchmarks.febrl4 --blocking snm --key-field surname # SNM + Jaccard
```

Uses the Febrl4 person record linkage dataset (built into `recordlinkage` — no download required).
5,000 records per table · 5,000 true 1:1 matches · synthetic Australian person records with realistic noise.

| Config | Precision | Recall | F1 | Time |
|--------|-----------|--------|-----|------|
| LSH + Jaccard · threshold=0.3 | **1.000** | **0.955** | **0.977** | 1.1s |
| SNM (surname) + Jaccard · threshold=0.3 | 1.000 | 0.384 | 0.555 | 0.4s |

LSH generalizes across all field variations; SNM recall drops when the blocking key (surname) is noisy.
All results are reproducible: `pip install recordlinkage && python -m benchmarks.febrl4`.

## Benchmark — DBLP-ACM

```bash
python -m benchmarks.dblp_acm --data-path /path/to/dblp_acm.csv
python -m benchmarks.dblp_acm --data-path /path/to/dblp_acm.csv --blocking snm --key-field title
```

Academic publication record linkage across DBLP and ACM databases.
2,616 DBLP records · 2,294 ACM records · 2,220 ground-truth matching pairs.
Data: Magellan ER benchmark collection (Köpcke & Rahm, 2010).

| Config | Precision | Recall | F1 | Time |
|--------|-----------|--------|-----|------|
| LSH + Jaccard · threshold=0.5 | 0.697 | 0.925 | 0.795 | 0.9s |
| LSH + Jaccard · threshold=0.7 | 0.900 | 0.653 | 0.757 | 1.1s |
| SNM (title) + Jaccard · threshold=0.5 | **0.899** | **0.957** | **0.927** | 0.3s |

SNM with title blocking outperforms LSH on this academic dataset: paper titles are stable identifiers across DBLP and ACM, so sorted-neighborhood retrieval finds almost all true matches without generating as many false candidates.

## Architecture

```
sandx_er/
├── resolver.py     EntityResolver — pipeline orchestrator
├── blocking.py     LSHBlocking, SortedNeighborhoodBlocking, EmbeddingANNBlocking
├── matching.py     JaccardScorer, CosineSimilarityScorer
└── clustering.py   ConnectedComponentsClustering, CorrelationClustering
```

**Optional dependency:** [`sandx-embed`](https://github.com/sandxlab/sandx-embed) for embedding-based blocking and matching.

## Benchmark Datasets

| Dataset | Domain | Table A | Table B | Matches |
|---------|--------|---------|---------|---------|
| Abt-Buy | E-commerce | 1,081 | 1,092 | ~1,097 |
| DBLP-ACM | Academic | 2,616 | 2,294 | 2,224 |
| DBLP-Scholar | Academic | 2,616 | 64,263 | 5,347 |
| Cora | Citations | 1,879 | — | dedup |

All benchmark runs are version-tagged and fully reproducible from public data.

## Related

- [`sandx-embed`](https://github.com/sandxlab/sandx-embed) — shared embedding infrastructure
- [`sandx-graph`](https://github.com/sandxlab/sandx-graph) — graph intelligence over resolved entities
- [sandx.io](https://sandx.io) — project home

## License

Apache 2.0 — see [LICENSE](LICENSE)
