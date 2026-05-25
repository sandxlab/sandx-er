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
| Fodors-Zagats benchmark | **Working** |
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

## Benchmark — Fodors-Zagats

```bash
python -m benchmarks.restaurant --data-path /path/to/restaurant.csv
python -m benchmarks.restaurant --data-path /path/to/restaurant.csv --blocking snm --key-field name
```

Restaurant record linkage across Fodors and Zagats listings.
533 Fodors records · 331 Zagats records · 110 ground-truth matching pairs.
Data: Magellan ER benchmark collection (Köpcke & Rahm, 2010).

| Config | Precision | Recall | F1 | Time |
|--------|-----------|--------|-----|------|
| LSH + Jaccard · threshold=0.5 | 0.807 | 0.645 | 0.717 | 0.1s |
| SNM (name) + Jaccard · threshold=0.3 | 0.810 | **0.891** | **0.848** | 0.0s |
| SNM (name) + Jaccard · threshold=0.5 | **1.000** | 0.745 | 0.854 | 0.0s |

Restaurant names are a stable-enough identifier despite variations ("art's deli" vs "art's delicatessen"), so SNM on `name` recovers most true matches at low threshold. Setting threshold=0.5 eliminates all false positives (perfect precision) at the cost of recall.

## Raw to Clean Demo

The most common use case: you have a raw DataFrame with duplicate, inconsistent vendor/customer/patient records. You want clean, deduplicated entities.

```bash
pip install sandx-er
python -m examples.raw_to_clean
```

24 vendor records, 6 underlying companies, 7 noise types (punctuation, abbreviations, suffix variation, hyphenation, case differences, word-boundary splits, address shorthand):

```
==============================================================
 SandX Entity Resolution  --  Raw to Clean
==============================================================
 24 raw records  .  6 underlying vendors  .  real-world name/address noise

 RAW INPUT
 --------------------------------------------------------------
 v01    Meridian Health Solutions               Boston, MA
 v02    Meridian Health Solutions Inc.          Boston MA
 v03    Meridian Health Soln. LLC               Boston
 v04    Meridian Health Solution                Boston, MA
 v05    BioCore Analytics Inc.                  San Diego, CA
 v06    Bio-Core Analytics                      San Diego CA
 ...

 RESOLVED ENTITIES
 --------------------------------------------------------------
 ENTITY                               CONF  SIZE  RECORDS
 Meridian Health Solution             0.69     4  [v03  v04  v02  v01]
 Biocore Analytics                    0.67     4  [v08  v05  v07  v06]
 DataVault Sys.                       0.58     4  [v09  v11  v12  v10]
 Cloudpeak Infra.                     0.61     4  [v16  v13  v15  v14]
 Nexus Financial Group                0.71     2  [v17  v19]
 Vertex Res. Labs                     0.61     4  [v24  v22  v23  v21]

 Unresolved singletons: 2

==============================================================
 24 raw records  ->  6 resolved entities  [5 ms]
==============================================================
```

The 2 singletons ("Nexus Financial Grp." and "Nexus Fin. Group") are too heavily abbreviated for character Jaccard at threshold 0.30. Switching to embedding-based matching resolves them:

```python
er = EntityResolver(blocking="embedding", similarity="embedding", threshold=0.85)
```

Three lines of code drove the entire resolution:

```python
from sandx_er import EntityResolver

er     = EntityResolver(blocking="lsh", similarity="jaccard", threshold=0.30)
result = er.resolve(df)        # df: pandas DataFrame of raw records
for c in result.clusters:
    print(c.canonical_id[:8], c.size, c.confidence)
```

See [`examples/raw_to_clean.py`](examples/raw_to_clean.py) for the full source with annotated noise types.

## Full Pipeline Demo

Run the end-to-end demo (sandx-er + sandx-graph):

```bash
pip install sandx-er sandx-graph
python -m examples.full_pipeline
```

Resolves 15 noisy company records into 5 entities, builds a knowledge graph, and computes consensus scores:

```
Resolved 15 records -> 5 entities  [2ms]

  Apple Inc               conf=0.59  size=3
  Microsoft Corp          conf=0.58  size=3
  Google LLC              conf=0.57  size=3
  Amazon Inc.             conf=0.62  size=3
  Meta Platforms          conf=0.67  size=3

KnowledgeGraph(n_nodes=5, n_edges=5)

  Apple Inc    -- Microsoft Corp   0.82  ########################
  Apple Inc    -- Google LLC       0.78  #######################
  Google LLC   -- Amazon Inc.      0.75  ######################

  Google LLC   score=0.737  support=3  conflict=0
  Apple Inc    score=0.800  support=2  conflict=0
```

See [`examples/full_pipeline.py`](examples/full_pipeline.py) for the full source.

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
| Fodors-Zagats | Restaurants | 533 | 331 | 110 |
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
