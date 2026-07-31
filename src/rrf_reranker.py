"""
Module 6: Reciprocal Rank Fusion (RRF) Hybrid Search Reranker
Merges dense vector and sparse keyword ranking streams using standard RRF formulas.
"""

from typing import List, Dict, Any


class ReciprocalRankFusionReranker:
    def __init__(self, k: int = 60):
        """
        k: Smoothing constant parameter controlling high-rank weighting (default 60).
        """
        self.k = k

    def rerank(self, ranking_lists: List[List[Dict[str, Any]]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Merges list of ordered result lists.
        Each item dict must contain a unique 'id' key.
        RRF Score = sum(1 / (k + rank_i))
        """
        rrf_scores: Dict[str, float] = {}
        item_store: Dict[str, Dict[str, Any]] = {}

        for ranking in ranking_lists:
            for rank, item in enumerate(ranking, start=1):
                doc_id = item["id"]
                item_store[doc_id] = item

                score = 1.0 / (self.k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        # Sort documents by accumulated RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda doc_id: rrf_scores[doc_id], reverse=True)

        reranked_results = []
        for doc_id in sorted_ids[:top_n]:
            entry = item_store[doc_id].copy()
            entry["rrf_score"] = float(rrf_scores[doc_id])
            reranked_results.append(entry)

        return reranked_results


if __name__ == "__main__":
    dense_results = [
        {"id": "doc_A", "text": "Dense result 1"},
        {"id": "doc_B", "text": "Dense result 2"}
    ]
    sparse_results = [
        {"id": "doc_B", "text": "Sparse result 1"},
        {"id": "doc_C", "text": "Sparse result 2"}
    ]

    reranker = ReciprocalRankFusionReranker(k=60)
    merged = reranker.rerank([dense_results, sparse_results], top_n=3)
    
    print("Merged RRF Rankings:")
    for item in merged:
        print(f"ID: {item['id']} | RRF Score: {item['rrf_score']:.5f}")