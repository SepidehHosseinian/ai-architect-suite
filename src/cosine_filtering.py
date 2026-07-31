"""
Module 1: Vector Similarity & Metadata Payload Filter Engine
High-throughput vector search core with strict payload filtering.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class VectorFilterEngine:
    def __init__(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """
        Initialize index with normalized embeddings and corresponding metadata.
        embeddings shape: (N, D)
        """
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.metadata = metadata

    def _matches_filter(self, payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches exact key-value criteria."""
        for key, value in filters.items():
            if payload.get(key) != value:
                return False
        return True

    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 5, 
        filters: Dict[str, Any] = None
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Executes cosine similarity search with optional payload pre-filtering.
        Returns: List of (index, similarity_score, metadata)
        """
        # Normalize query vector
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        # Calculate raw cosine similarities: shape (N,)
        similarities = np.dot(self.embeddings, query_norm)
        
        results = []
        for idx, score in enumerate(similarities):
            payload = self.metadata[idx]
            if filters and not self._matches_filter(payload, filters):
                continue
            results.append((idx, float(score), payload))
            
        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


if __name__ == "__main__":
    # Smoke Test
    np.random.seed(42)
    vectors = np.random.randn(100, 128)
    meta = [{"tenant_id": "A" if i % 2 == 0 else "B", "doc_type": "pdf"} for i in range(100)]
    
    engine = VectorFilterEngine(vectors, meta)
    q = np.random.randn(128)
    matches = engine.search(q, top_k=3, filters={"tenant_id": "A"})
    print("Filtered Results:", matches)