# Entity Resolution — Domain Overview

**Domain:** Entity Resolution (ER); also known as record linkage, deduplication, entity matching, reference reconciliation
**SandX engine:** `sandx-er`
**Phase 2 priority:** #1 — flagship engine

---

## What Is Entity Resolution?

Entity Resolution is the problem of determining which records across one or more datasets refer to the same real-world entity. In production databases, the same entity is routinely represented in fragmented, inconsistent, or duplicate forms due to data entry errors, schema mismatches, system migrations, or integration across heterogeneous data sources.

**Core tasks:**

| Task | Definition |
|------|-----------|
| Deduplication | Identify duplicate records within a single dataset |
| Record linkage | Match records across two or more datasets |
| Identity graph construction | Build a graph where edges represent same-entity decisions |
| Canonical entity merging | Merge a resolved cluster into a single authoritative record |

---

## Why It Is Hard

### 1. Scale — the O(N²) problem
Naïve pairwise comparison of N records requires O(N²) comparisons. At N = 10 million, that is 50 trillion comparisons. **Blocking** (candidate generation) is the technique that reduces the comparison space to a tractable subset of likely-match pairs. It is the central engineering challenge in scalable ER.

Blocking methods:
- Sorted Neighborhood Method (SNM)
- Standard Blocking on key attributes
- Locality-Sensitive Hashing (LSH)
- Embedding-based Approximate Nearest Neighbor (ANN) search

### 2. Accuracy — noise and variation
Field values representing the same entity differ due to typos, abbreviations, transliterations, missing data, and format variation. Matching requires similarity measures that are robust to these variations: edit distance, Jaro-Winkler, TF-IDF, and learned embeddings.

### 3. Transitivity — the clustering problem
Pairwise matching decisions are noisy and may be inconsistent: if A matches B and B matches C, it does not follow that A matches C. Clustering reconciles pairwise decisions into globally consistent entity sets. Methods: connected components, correlation clustering, min-cost flow optimization.

### 4. Ground truth scarcity
Labeled ER datasets are expensive to produce — they require human annotation at scale. This motivates active learning, weak supervision (programmatic labeling), and unsupervised approaches.

---

## State of the Art

| Approach | Key Methods |
|----------|------------|
| Classical rule-based | Deterministic field matching, business rules |
| Probabilistic | Fellegi-Sunter model, FastLink, BayesMatch |
| Feature-based ML | Logistic regression, gradient boosting on field similarity features |
| Deep learning | DeepMatcher, DITTO (pre-trained transformers), BERT-based matchers |
| Embedding-based | EmbDI, MCAN, GNN-based entity matching |
| Scalable systems | Zingg (Spark-based), Dedupe.io, Splink |

Notable benchmarks: Abt-Buy, DBLP-ACM, DBLP-Scholar, Amazon-Google, Walmart-Amazon, WDC product corpus, Cora citation dataset.

---

## SandX-ER Positioning

Existing tools fall into two categories:
- **Research-grade:** High-accuracy, not production-deployable (DeepMatcher, DITTO)
- **Domain-specific:** Production-ready but not generalizable (Zingg, Dedupe.io)

SandX-ER targets the gap: a **composable, production-ready ER engine** with a clean programmatic interface, modular pipeline stages, and embedding-aware similarity (via `sandx-embed`).

**Target differentiators:**
1. Modular pipeline — swap blocking, matching, and clustering components independently
2. Embedding-native — `sandx-embed` as first-class blocking and matching primitive
3. Probabilistic outputs — every match decision carries a confidence score
4. Reproducible benchmarks — all results re-runnable from public datasets
5. Python-first, clean API — production integrable without research expertise

---

## Key References

- Christen, P. (2012). *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection.* Springer.
- Fellegi, I. P., & Sunter, A. B. (1969). A Theory for Record Linkage. *JASA.*
- Mudgal, S. et al. (2018). Deep Learning for Entity Matching: A Design Space Exploration. *SIGMOD.*
- Li, Y. et al. (2020). Deep Entity Matching with Pre-Trained Language Models (DITTO). *VLDB.*
- Papadakis, G. et al. (2021). Blocking and Filtering Techniques for Entity Resolution: A Survey. *ACM CSUR.*
- Köpcke, H., Thor, A., & Rahm, E. (2010). Evaluation of Entity Resolution Approaches on Real-World Match Problems. *VLDB.*
- Christophides, V. et al. (2020). End-to-End Entity Resolution for Big Data: A Survey. *ACM CSUR.* — Comprehensive pipeline survey covering blocking, matching, clustering, and scalability; frames ER as an end-to-end systems problem.
- Peeters, R., & Bizer, C. (2023). Using ChatGPT for Entity Matching. *VLDB.* — Evaluates LLM-based zero-shot entity matching; establishes when LLMs substitute for trained matchers and where they fall short.
- Mudgal, S. et al. (2022). Generalized Sparse Matrix-Vector Products for Blocking in Entity Matching. *SIGMOD.* — Advances in sparse ANN blocking; directly applicable to `sandx-er` blocking layer design.
