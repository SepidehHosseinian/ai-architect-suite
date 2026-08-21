import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("RRFRankerEngine")


# ---------------------------------------------------------------------------
# Domain Models & Contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RankedCandidate:
    """
    Immutable domain contract representing a document candidate returned 
    by a specific retrieval engine stream (Dense or Sparse).
    """
    id: str
    tenant_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedSearchResult:
    """Output contract containing unified rank positions and score components."""
    document_id: str
    tenant_id: str
    rrf_score: float
    rank_positions: Dict[str, int]  # Maps retrieval_stream_name -> 1-based rank
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core Hybrid Ranker Engine
# ---------------------------------------------------------------------------

class ReciprocalRankFusionEngine:
    """
    High-throughput Hybrid Search Ranker Engine using Reciprocal Rank Fusion (RRF).
    Fuses dense vector and sparse keyword streams under multi-tenant isolation.
    """

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError(f"RRF smoothing constant k must be > 0. Received: {k}")
        self._k = k

    def rank(
        self,
        tenant_id: str,
        retrieval_streams: Dict[str, List[RankedCandidate]],
        top_k: int = 10
    ) -> List[FusedSearchResult]:
        """
        Executes multi-stream reciprocal rank fusion with tenant payload isolation.

        :param tenant_id: Multi-tenant boundary key for hard security filtering.
        :param retrieval_streams: Keyed dictionary of retrieval candidate streams 
                                  e.g., {'dense_vector': [...], 'bm25_sparse': [...]}
        :param top_k: Maximum number of merged candidates to return.
        :return: Rank-ordered list of FusedSearchResult objects.
        """
        if not retrieval_streams:
            return []

        # Step 1: Pre-Filter & Normalize Streams (Enforce Tenant Boundaries & Rank Order)
        cleaned_streams: Dict[str, List[RankedCandidate]] = {}
        for stream_name, candidates in retrieval_streams.items():
            # Apply Tenant Isolation Security Guard
            tenant_valid = [c for c in candidates if c.tenant_id == tenant_id]
            # Ensure strict ordering by raw score descending to establish ordinal rank
            sorted_candidates = sorted(tenant_valid, key=lambda x: x.score, reverse=True)
            cleaned_streams[stream_name] = sorted_candidates

        # Step 2: Accumulate RRF Scores and Track Positional Lineage
        rrf_accumulator: Dict[str, float] = {}
        document_metadata: Dict[str, Dict[str, Any]] = {}
        rank_tracker: Dict[str, Dict[str, int]] = {}

        for stream_name, candidates in cleaned_streams.items():
            for rank_idx, candidate in enumerate(candidates, start=1):
                doc_id = candidate.id

                # Calculate RRF reciprocal score step: 1 / (k + r_m(d))
                reciprocal_rank_delta = 1.0 / (self._k + rank_idx)
                rrf_accumulator[doc_id] = rrf_accumulator.get(doc_id, 0.0) + reciprocal_rank_delta

                # Capture positional lineage per channel
                if doc_id not in rank_tracker:
                    rank_tracker[doc_id] = {}
                rank_tracker[doc_id][stream_name] = rank_idx

                # Preserve document metadata from candidate payloads
                if doc_id not in document_metadata and candidate.metadata:
                    document_metadata[doc_id] = candidate.metadata

        if not rrf_accumulator:
            return []

        # Step 3: Sort Aggregated Candidates by Fused Score
        sorted_doc_ids = sorted(
            rrf_accumulator.keys(),
            key=lambda doc_id: rrf_accumulator[doc_id],
            reverse=True
        )[:top_k]

        # Step 4: Construct Immutable Response Objects
        return [
            FusedSearchResult(
                document_id=doc_id,
                tenant_id=tenant_id,
                rrf_score=float(rrf_accumulator[doc_id]),
                rank_positions=rank_tracker[doc_id],
                metadata=document_metadata.get(doc_id, {})
            )
            for doc_id in sorted_doc_ids
        ]