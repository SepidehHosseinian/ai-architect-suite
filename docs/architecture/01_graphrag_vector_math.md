# 🏛️ Architecture Spec 01: Hybrid GraphRAG & Vector Retrieval Topology

## 1. Executive Summary
Standard flat vector search struggles with multi-hop reasoning, broad contextual synthesis, and global dataset comprehension. This spec outlines a production **Hybrid GraphRAG System** that unifies community-level knowledge graph traversals with dense vector similarity search to deliver sub-second multi-hop retrieval.

---

## 2. System Architecture Diagram

```text
                                 [ User Query ]
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │  Intent & Query Router   │
                         └─────────────┬────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ (Global / Relational)                       │ (Local / Semantic)
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │ Community Summary Search  │                 │  Dense Vector Retrieval   │
  │  (Leiden Graph Hierarchies)│                 │   (HNSW Cosine Similarity)│
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │                                             │
                │ Extract Entities & Subgraphs                │ Top-K Chunks
                ▼                                             ▼
  ┌───────────────────────────┐                 ┌───────────────────────────┐
  │ Knowledge Graph Search    │                 │ Payload Criteria Filter   │
  │   (Cypher / Neo4j Engine) │                 │   (Tenant & RBAC Isolation│
  └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │ Reciprocal Rank Fusion   │
                         │      (RRF Reranker)      │
                         └─────────────┬────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │ Context Assembler & LLM  │
                         └──────────────────────────┘
```
## 3. Mathematical Foundations: Vector Cosine Similarity
Cosine similarity measures the orientation between query vector $\mathbf{q}$ and document vector $\mathbf{d}$ in $D$-dimensional space:$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\Vert{}\mathbf{q}\Vert{}_2 \Vert{}\mathbf{d}\Vert{}_2} = \frac{\sum_{i=1}^{D} q_i d_i}{\sqrt{\sum_{i=1}^{D} q_i^2} \cdot \sqrt{\sum_{i=1}^{D} d_i^2}}$$When vectors are unit-normalized ($\Vert{}\mathbf{q}\Vert{}_2 = \Vert{}\mathbf{d}\Vert{}_2 = 1$), cosine similarity simplifies directly to the dot product:$$\text{Cosine Similarity}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d} = \sum_{i=1}^{D} q_i d_i$$

## 4. Architectural Trade-Off Matrix
Architectural Pattern,Latency (p95),Memory Overhead,Multi-Hop Accuracy,Best Use Case
Flat Vector Search (HNSW),<15ms,Low (O(N⋅D)),Low (42%),"Factoid lookup, single-doc extraction"
Pure Graph Traversal (Cypher),45ms−120ms,High (O(V+E)),High (88%),"Entity relationship tracking, lineage"
Hybrid GraphRAG + RRF,85ms−180ms,Medium-High,Very High (94%),"Enterprise co-pilots, complex synthesis"