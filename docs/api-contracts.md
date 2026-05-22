# SandX API Contract Sketches — Phase 1

**Status:** Informal design sketches. These define intended interfaces for Phase 2 implementation. Not binding; subject to revision before v0.1 release.

---

## sandx-er — Entity Resolution Engine

### `EntityResolver`

```python
EntityResolver(
    blocking: str | BlockingMethod = "lsh",     # "lsh" | "snm" | "embedding" | BlockingMethod instance
    similarity: str | SimilarityMethod = "embedding",  # "embedding" | "jaccard" | "edit" | custom
    clustering: str | ClusteringMethod = "connected_components",  # "connected_components" | "correlation"
    threshold: float = 0.85,                    # match confidence threshold [0.0, 1.0]
    embed_model: str | None = None,             # sandx-embed model name, if similarity="embedding"
    n_jobs: int = -1,                           # parallelism (-1 = all cores)
)
```

**`.resolve(records) → ResolutionResult`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `records` | `pd.DataFrame` | Input records. Each row is one entity observation. Must have a unique row ID. |

| Output field | Type | Description |
|-------------|------|-------------|
| `result.clusters` | `list[EntityCluster]` | List of resolved entity clusters |
| `result.n_records` | `int` | Total input records processed |
| `result.n_clusters` | `int` | Distinct resolved entities |
| `result.to_dataframe()` | `pd.DataFrame` | Records with `cluster_id` and `confidence` columns appended |

**`EntityCluster`**

| Field | Type | Description |
|-------|------|-------------|
| `canonical_id` | `str` | Stable cluster identifier (UUID) |
| `record_ids` | `list` | Row IDs of all records in this cluster |
| `confidence` | `float` | Mean pairwise match confidence within cluster |
| `size` | `int` | Number of records in cluster |

**Error conditions:**

| Condition | Behavior |
|-----------|----------|
| Empty DataFrame | Raises `ValueError` |
| Missing required index | Raises `ValueError` with field name |
| Blocking returns 0 candidates | Returns `ResolutionResult` with N singleton clusters |
| Threshold out of [0, 1] | Raises `ValueError` |

---

## sandx-embed — Embedding Infrastructure

### `Encoder`

```python
Encoder(model: str)  # e.g. "sbert", "e5-base", "bge-large", or registered custom name
```

**`.encode(inputs) → np.ndarray`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `inputs` | `list[str] | pd.Series | pd.DataFrame` | Text strings or tabular records to encode |
| `batch_size` | `int` | Default 64. Batch size for model inference. |
| `normalize` | `bool` | Default True. L2-normalize output vectors. |

Returns: `np.ndarray` of shape `(n_inputs, embedding_dim)`

### `VectorIndex`

```python
VectorIndex(method: str = "hnsw", metric: str = "cosine")
# method: "hnsw" | "faiss" | "annoy" | "exact"
# metric: "cosine" | "euclidean" | "dot"
```

**`.build(vectors: np.ndarray) → None`**

**`.query(query: np.ndarray, k: int = 10) → SearchResult`**

| Output field | Type | Description |
|-------------|------|-------------|
| `result.ids` | `np.ndarray[int]` | Indices of k nearest neighbors |
| `result.distances` | `np.ndarray[float]` | Distances to k nearest neighbors |

**`.save(path: str) → None`** / **`.load(path: str) → VectorIndex`**

---

## sandx-graph — Graph Intelligence Engine

### `GraphBuilder`

```python
GraphBuilder.from_clusters(result: ResolutionResult) → KnowledgeGraph
GraphBuilder.from_dataframe(df: pd.DataFrame, node_col: str, edge_col: str, weight_col: str | None) → KnowledgeGraph
```

### `KnowledgeGraph`

```python
graph = KnowledgeGraph(nodes: list[str], edges: list[tuple[str, str, float]])
```

| Method | Returns | Description |
|--------|---------|-------------|
| `graph.nodes` | `list[str]` | All node identifiers |
| `graph.edges` | `list[tuple]` | `(source, target, weight)` triples |
| `graph.neighbors(node_id)` | `list[str]` | Adjacent node IDs |
| `graph.to_networkx()` | `nx.Graph` | Export to NetworkX for interop |
| `graph.to_dataframe()` | `pd.DataFrame` | Edge list as DataFrame |

### `ConsensusEngine`

```python
ConsensusEngine(graph: KnowledgeGraph)
```

**`.compute(node_id: str, depth: int = 2) → ConsensusScore`**

| Output field | Type | Description |
|-------------|------|-------------|
| `score.node_id` | `str` | Target node |
| `score.consensus` | `float` | Neighborhood consensus score [0.0, 1.0] |
| `score.support` | `int` | Number of neighbors contributing to score |
| `score.depth` | `int` | Depth used in computation |

**`.compute_all(depth: int = 2) → dict[str, ConsensusScore]`**

---

## sandx-compute — Distributed Compute Orchestration

### `ResourceRegistry`

```python
registry = ResourceRegistry()
```

| Method | Returns | Description |
|--------|---------|-------------|
| `registry.register(node: ComputeNode)` | `None` | Add a compute node to the registry |
| `registry.available(requirements: dict)` | `list[ComputeNode]` | Nodes meeting resource requirements |
| `registry.update_status(node_id, status)` | `None` | Update node availability |
| `registry.all()` | `list[ComputeNode]` | All registered nodes |

**`ComputeNode`**

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | `str` | Unique node identifier |
| `gpu` | `str | None` | GPU model name (e.g. "A100", "RTX 4090") |
| `vram_gb` | `float` | GPU VRAM in gigabytes |
| `cpu_cores` | `int` | Available CPU cores |
| `ram_gb` | `float` | Available system RAM |
| `status` | `str` | "available" \| "busy" \| "offline" |
| `tags` | `list[str]` | Optional labels (e.g. "gpu", "high-memory") |

### `Scheduler`

```python
scheduler = Scheduler(registry: ResourceRegistry)
```

**`.submit(task: dict, requirements: dict) → Job`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | `dict` | Workload descriptor (type, payload, priority) |
| `requirements` | `dict` | Resource requirements (min_vram_gb, min_cpu_cores, tags) |

| Output field | Type | Description |
|-------------|------|-------------|
| `job.job_id` | `str` | UUID for tracking |
| `job.status` | `str` | "queued" \| "running" \| "done" \| "failed" |
| `job.node_id` | `str | None` | Assigned node, once scheduled |

---

## Cross-Engine Data Flow

```
records: pd.DataFrame
    │
    ├─ Encoder.encode() → vectors: np.ndarray
    │       │
    │       └─ VectorIndex.build(vectors) → index: VectorIndex
    │               │
    ├───────────────┘
    │
    EntityResolver.resolve(records) → result: ResolutionResult
    │       (uses sandx-embed internally for blocking + matching)
    │
    GraphBuilder.from_clusters(result) → graph: KnowledgeGraph
    │
    ConsensusEngine(graph).compute_all() → scores: dict[str, ConsensusScore]
```

---

*Last updated: 2026-05-22 — Phase 1 design sketches. Implementation begins Phase 2.*
