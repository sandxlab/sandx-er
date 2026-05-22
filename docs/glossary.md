# SandX Glossary

Core terminology used across SandX documentation, research notes, and engineering specifications.

---

## A

**Approximate Nearest Neighbor (ANN)**
A class of algorithms that find the k vectors in an index most similar to a query vector, without exhaustively comparing all pairs. Approximate (rather than exact) search trades a small accuracy cost for dramatically lower latency and memory use. Core component of `sandx-embed`. Key methods: HNSW, FAISS, Annoy, ScaNN.

**Attention (Graph Attention)**
A mechanism in graph neural networks where a node's representation is computed as a weighted sum of its neighbors' representations, with weights learned by an attention function. Used in Graph Attention Networks (GATs). See also: *Message Passing*, *GNN*.

---

## B

**Blocking**
The candidate generation step in entity resolution. Instead of comparing all O(N²) record pairs, blocking groups records by a coarse key so that only records within the same block are compared. Reduces comparison space from O(N²) to O(N·B) where B is the mean block size. Quality measured by *reduction ratio* and *pair completeness*. See also: *LSH*, *Sorted Neighborhood Method*, *Embedding ANN Blocking*.

**Blocking Key**
An attribute or attribute combination used to partition records into blocks. Records sharing the same blocking key are considered candidates for comparison. Example: first 3 characters of surname + year of birth.

---

## C

**Canonical Entity**
The single authoritative record representing a resolved entity cluster — the "best" or merged representation of all records that resolve to the same real-world entity. See also: *Entity Cluster*, *Canonical Merging*.

**Canonical Merging**
The process of combining all records in a resolved entity cluster into a single canonical record. Requires a merge policy: use the most frequent value, the most recent value, or a weighted vote.

**Candidate Pairs**
The set of record pairs produced by the blocking step that are considered for pairwise similarity computation. A candidate pair (A, B) will be evaluated; non-candidate pairs are skipped.

**Clustering (Entity Resolution)**
The step that groups pairwise match decisions into globally consistent entity sets (clusters). Because pairwise matching is noisy and may produce intransitive decisions (A~B, B~C, but A≁C), clustering must reconcile these into coherent groups. Methods: *Connected Components*, *Correlation Clustering*, min-cost flow.

**Confidence Score**
A probability or normalized score in [0, 1] attached to a match decision, cluster assignment, or graph consensus computation. SandX outputs carry confidence scores rather than binary decisions. A score of 1.0 indicates certainty; 0.0 indicates no evidence.

**Connected Components (Clustering)**
The simplest ER clustering method: treat all above-threshold pairs as edges in a graph; each connected component is a resolved entity cluster. Fast and exact but cannot resolve conflicting transitive matches.

**Consensus (Graph)**
A measure of agreement among a node's neighborhood in a knowledge graph. When multiple observers or sources submit claims about an entity's relationships, consensus computation reconciles these into a single high-confidence graph state. Related: belief propagation, collective classification.

**Correlation Clustering**
An ER clustering algorithm that finds the partition of records that best agrees with pairwise match/no-match decisions, allowing corrections to intransitive decisions at a cost. Formulated as an optimization problem; NP-hard in general, approximated in practice.

---

## D

**Deduplication**
The ER sub-task of identifying duplicate records within a *single* dataset. Contrast with *Record Linkage*, which operates across two or more datasets.

---

## E

**Embedding**
A dense, fixed-length vector in a continuous latent space representing an object (record, word, graph node, image). Objects that are semantically similar have geometrically proximate embeddings. Produced by encoder models. Core primitive of `sandx-embed`.

**Embedding ANN Blocking**
A blocking method where records are encoded as embeddings and a vector index (ANN) retrieves the k nearest neighbors for each record as candidate pairs. More flexible than key-based blocking because it captures semantic similarity rather than exact key matches.

**Entity**
A real-world object or concept represented by one or more records in a dataset. Examples: a person, an organization, a product, a publication, a location.

**Entity Cluster**
A set of records that entity resolution has determined refer to the same real-world entity. Each cluster has a *canonical ID* and a *confidence score*.

**Entity Resolution (ER)**
The computational task of determining which records across one or more datasets refer to the same real-world entity. Encompasses *deduplication*, *record linkage*, *identity graph construction*, and *canonical entity merging*. Also known as entity matching, record deduplication, reference reconciliation. See: `sandx-er`.

---

## F

**F1 Score**
Harmonic mean of precision and recall: F1 = 2 · (P · R) / (P + R). The primary evaluation metric for ER systems. A score of 1.0 is perfect.

**Fellegi-Sunter Model**
The foundational probabilistic framework for record linkage (1969). Models each field comparison as evidence for or against a match, combining evidence using log-likelihood ratios derived from match and non-match populations. Basis for modern probabilistic ER systems including Splink.

---

## G

**Graph Neural Network (GNN)**
A class of neural networks that operate directly on graph-structured data. Nodes aggregate information from their neighbors through message passing, producing node representations that encode both local attributes and neighborhood structure. Used in `sandx-graph` for node classification, link prediction, and entity matching.

**GPU Orchestration**
The scheduling and allocation of GPU compute resources across one or more nodes or clusters. In `sandx-compute`, orchestration includes resource registration, workload scheduling, consensus over resource state, and fault-tolerant job execution.

---

## H

**HNSW (Hierarchical Navigable Small World)**
A graph-based ANN index structure. Builds a multi-layer proximity graph; search traverses from coarse to fine layers. Achieves high recall at low latency; preferred for production deployments. Implemented in `hnswlib` and `faiss`.

---

## I

**Identity Graph**
A graph where nodes represent input records and edges represent same-entity match decisions (with weights = confidence scores). The connected components of a high-confidence identity graph define entity clusters.

---

## K

**Knowledge Graph**
A structured graph representation of entities and their relationships. Nodes are entities; edges are typed relationships with attributes. Built from resolved entity clusters by `sandx-graph`.

---

## L

**Locality-Sensitive Hashing (LSH)**
A family of hashing methods where similar items hash to the same bucket with high probability. Used in blocking: records with similar field values are placed in the same LSH bucket and compared as candidates. Commonly based on MinHash (for Jaccard similarity) or random projections (for cosine similarity).

**Latent Space**
The continuous vector space in which embeddings live. "Latent" refers to the fact that the dimensions are learned, not hand-engineered — they encode meaningful structure implicitly.

---

## M

**Matching**
The step in the ER pipeline that computes a pairwise similarity score for each candidate pair produced by blocking. Matching outputs a probability or score indicating whether the two records refer to the same entity. Methods range from rule-based field comparison to deep learning matchers.

**Message Passing**
The computational pattern underlying GNNs: each node sends a message to its neighbors; each node aggregates received messages to update its representation. Multiple rounds of message passing allow information to propagate through the graph.

---

## P

**Pair Completeness (PC)**
The fraction of true matching pairs that survive blocking: PC = (matching pairs in candidates) / (all matching pairs). A blocking method with high PC misses few true matches. Ideal: 1.0. Must be balanced against reduction ratio.

**Probabilistic Output**
A core SandX design principle: every output carries a confidence score or probability, not a binary decision. Enables downstream systems to apply their own thresholds and propagate uncertainty.

---

## R

**Record Linkage**
The ER sub-task of matching records *across* two or more datasets. Contrast with *Deduplication*, which operates within a single dataset.

**Reduction Ratio (RR)**
The fraction of pairs eliminated by blocking: RR = 1 − (candidate pairs) / (all pairs). A blocking method with high RR eliminates most of the O(N²) comparison space. Ideal: close to 1.0. Must be balanced against pair completeness.

**Resolution Infrastructure**
The generalized SandX concept: infrastructure for resolving ambiguity in complex systems — fragmented identity, noisy observations, uncertain relationships, probabilistic truth. ER is the concrete instantiation; the concept extends to AI reasoning and computational consensus.

---

## S

**Similarity Score**
A numerical value in [0, 1] indicating how similar two records or embeddings are. High similarity → likely same entity. Threshold applied to similarity scores produces binary match decisions.

**Sorted Neighborhood Method (SNM)**
A blocking method that sorts records by a key and compares records within a sliding window. Simple, interpretable, effective for key-based similarity. Linear in N for fixed window size.

---

## V

**Vector Index**
A data structure optimized for approximate nearest neighbor search over a set of dense vectors. Given a query vector, returns the k most similar vectors from the index. See: *HNSW*, *FAISS*, *ANN*. Implemented in `sandx-embed`.

**Vector Database**
A system that stores, indexes, and queries dense vectors at scale with additional metadata support and distributed storage. Examples: Pinecone, Weaviate, Qdrant, Chroma. `sandx-embed`'s `VectorIndex` provides the core primitive; full vector database features are out of scope for Phase 2.

---

*Last updated: 2026-05-22*
