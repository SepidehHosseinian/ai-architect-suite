# 🏛️ Architecture Spec 03: Dense/Sparse Vector Search & Payload Isolation

## 1. Executive Summary
To achieve precision across specialized technical terminology and broad semantic queries, this spec details a multi-tenant vector engine combining dense neural embeddings with sparse BM25 token vectors, constrained by real-time payload filtering.

---

## 2. Ingestion & Retrieval Pipeline

```text
 [ Unstructured Document ]
            │
            ▼
 ┌──────────────────────┐
 │ Chunking & Metadata  │
 └──────────┬───────────┘
            │
            ├───────────────────────────────┐
            │                               │
            ▼                               ▼
 ┌──────────────────────┐        ┌──────────────────────┐
 │ Dense Embedder       │        │ Sparse Embedder      │
 │ (e.g., bge-large-en) │        │ (e.g., BM25 / SPLADE)│
 └──────────┬───────────┘        └──────────┬───────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │ Vector DB Indexing          │
             │ - HNSW Graph (Dense)        │
             │ - Inverted Index (Sparse)   │
             │ - Inverted Payload Index    │
             └──────────────┬──────────────┘
                            │
                            ▼
             ┌─────────────────────────────┐
             │ Filtered Hybrid Search      │
             │ Payload: tenant_id == 'A'   │
             └─────────────────────────────┘
```             
## 3. Hybrid Scoring MechanicsFinal
 relevance score $S_{\text{hybrid}}$ balances dense semantic distance with sparse exact-match lexical weight:$$S_{\text{hybrid}}(q, d) = \alpha \cdot S_{\text{dense}}(q, d) + (1 - \alpha) \cdot S_{\text{sparse}}(q, d)$$Where:$S_{\text{dense}}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\Vert{}\mathbf{v}_q\Vert{} \Vert{}\mathbf{v}_d\Vert{}}$$S_{\text{sparse}}(q, d) = \sum_{t \in q \cap d} \text{IDF}(t) \cdot \frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}d\vert{}}{\text{avgdl}}\right)}$$\alpha \in [0, 1]$ represents the tunable fusion weighting hyperparameter (default $\alpha = 0.7$).