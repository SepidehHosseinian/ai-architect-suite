"""
Module 3: In-Memory Filtered Vector Indexer
Lightweight in-memory vector indexing datastructure with namespace / payload isolation.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class InMemoryVectorIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: Optional[np.ndarray] = None
        self.payloads: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def upsert(self, doc_id: str, vector: List[float], payload: Dict[str, Any]):
        """Upsert vector and associated metadata payload into index."""
        vec = np.array(vector, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)

        if doc_id in self.ids:
            idx = self.ids.index(doc_id)
            self.vectors[idx] = vec
            self.payloads[idx] = payload
        else:
            self.ids.append(doc_id)
            self.payloads.append(payload)
            if self.vectors is None:
                self.vectors = np.array([vec], dtype=np.float32)
            else:
                self.vectors = np.vstack([self.vectors, vec])

    def query(
        self, 
        query_vector: List[float], 
        top_k: int = 5, 
        payload_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_vec = q_vec / np.linalg.norm(q_vec)

        # Dot product for cosine similarity
        scores = np.dot(self.vectors, q_vec)

        matched_results = []
        for i, score in enumerate(scores):
            payload = self.payloads[i]
            
            # Apply Filter
            if payload_filter:
                match = all(payload.get(k) == v for k, v in payload_filter.items())
                if not match:
                    continue

            matched_results.append({
                "id": self.ids[i],
                "score": float(score),
                "payload": payload
            })

        matched_results.sort(key=lambda x: x["score"], reverse=True)
        return matched_results[:top_k]


if __name__ == "__main__":
    idx = InMemoryVectorIndex(dimension=3)
    idx.upsert("doc1", [1.0, 0.0, 0.0], {"category": "tech"})
    idx.upsert("doc2", [0.0, 1.0, 0.0], {"category": "finance"})
    
    print("Tech search:", idx.query([0.9, 0.1, 0.0], top_k=1, payload_filter={"category": "tech"}))