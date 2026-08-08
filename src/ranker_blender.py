import numpy as np
from typing import List, Dict, Any, Optional

class HybridReRanker:
    """
    High-throughput hybrid re-ranker implementing Reciprocal Rank Fusion (RRF)
    and Cross-Encoder score blending under multi-tenant constraints.
    """
    def __init__(self, rrf_k: int = 60, alpha: float = 0.5):
        """
        :param rrf_k: Smoothing constant for RRF (default 60 per Cormack et al.).
        :param alpha: Weight balancing RRF rank score vs normalized Cross-Encoder score.
                      alpha=1.0 relies strictly on RRF; alpha=0.0 on Cross-Encoder.
        """
        self.rrf_k = rrf_k
        self.alpha = alpha

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max normalization with epsilon guard for zero-variance outputs."""
        min_val, max_val = np.min(scores), np.max(scores)
        if max_val - min_val < 1e-6:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def compute_rrf(self, rankings: List[List[str]]) -> Dict[str, float]:
        """
        Computes RRF score across multiple candidate rank lists (e.g., Sparse + Dense).
        RRF(d) = sum(1 / (k + r_m(d)))
        """
        rrf_scores: Dict[str, float] = {}
        
        for system_rankings in rankings:
            for rank, doc_id in enumerate(system_rankings, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))
                
        return rrf_scores

    def re_rank_and_blend(
        self,
        tenant_id: str,
        sparse_results: List[Dict[str, Any]],  # Expects [{'id': str, 'score': float}, ...]
        dense_results: List[Dict[str, Any]],   # Expects [{'id': str, 'score': float}, ...]
        cross_encoder_scores: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Blends sparse and dense streams, applies tenant validation, and optional
        Cross-Encoder fusion.
        """
        # Step 1: Multi-tenant payload filtering & rank extraction
        sparse_ranked = [
            doc['id'] for doc in sorted(sparse_results, key=lambda x: x['score'], reverse=True)
            if doc.get('tenant_id') == tenant_id
        ]
        dense_ranked = [
            doc['id'] for doc in sorted(dense_results, key=lambda x: x['score'], reverse=True)
            if doc.get('tenant_id') == tenant_id
        ]

        # Union of candidate IDs for the targeted tenant
        candidate_ids = list(set(sparse_ranked) | set(dense_ranked))
        if not candidate_ids:
            return []

        # Step 2: Compute baseline RRF across sparse and dense rank streams
        rrf_map = self.compute_rrf([sparse_ranked, dense_ranked])
        
        # Extract RRF scores in deterministic candidate order
        rrf_vector = np.array([rrf_map[doc_id] for doc_id in candidate_ids])
        
        # Step 3: Blend with Cross-Encoder scores if available
        if cross_encoder_scores:
            # Normalize RRF to [0, 1] range to make scale comparable
            norm_rrf = self._min_max_normalize(rrf_vector)
            
            # Gather and normalize cross-encoder raw logits/probabilities
            ce_vector = np.array([cross_encoder_scores.get(doc_id, -100.0) for doc_id in candidate_ids])
            norm_ce = self._min_max_normalize(ce_vector)
            
            # Convex combination: Score = alpha * Norm_RRF + (1 - alpha) * Norm_CE
            final_scores = (self.alpha * norm_rrf) + ((1.0 - self.alpha) * norm_ce)
        else:
            final_scores = rrf_vector

        # Step 4: Sort and construct output payload
        top_indices = np.argsort(-final_scores)[:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = candidate_ids[idx]
            results.append({
                "id": doc_id,
                "tenant_id": tenant_id,
                "score": float(final_scores[idx]),
                "rrf_score": float(rrf_map[doc_id])
            })
            
        return results