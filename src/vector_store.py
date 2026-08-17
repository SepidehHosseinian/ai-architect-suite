import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np


# ---------------------------------------------------------------------------
# Domain Contracts & Domain Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SecurityContext:
    """Immutable RBAC and Tenant security isolation context."""
    tenant_id: str
    user_roles: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class VectorRecord:
    """Represents an indexed document vector with metadata payload."""
    id: str
    vector: np.ndarray
    tenant_id: str
    allowed_roles: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Output container for rank-ordered query results."""
    id: str
    score: float
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# High-Throughput In-Memory Vector Index
# ---------------------------------------------------------------------------

class InMemoryVectorStore:
    """
    Zero-dependency, SIMD-accelerated In-Memory Vector Store supporting 
    Cosine Similarity and Hard Pre-Filtering for RBAC & Multi-Tenancy.
    """

    def __init__(self, dimension: int):
        self._dim = dimension
        
        # Internal Storage Structures
        self._records: List[VectorRecord] = []
        self._id_to_idx: Dict[str, int] = {}
        
        # Dense Contiguous Matrix for Fast Array Vectorization (N x D)
        self._matrix: Optional[np.ndarray] = None
        self._dirty = False  # Flag indicating matrix re-sync requirement

    def __len__(self) -> int:
        return len(self._records)

    def upsert(self, record: VectorRecord) -> None:
        """Upserts a record into the vector store with unit-normalization."""
        if record.vector.shape != (self._dim,):
            raise ValueError(f"Dimensionality mismatch: Expected ({self._dim},), got {record.vector.shape}")

        # L2-normalize vector up-front to transform cosine distance to dot product
        norm = np.linalg.norm(record.vector)
        normalized_vec = record.vector / norm if norm > 0 else record.vector

        normalized_record = VectorRecord(
            id=record.id,
            vector=normalized_vec.astype(np.float32),
            tenant_id=record.tenant_id,
            allowed_roles=record.allowed_roles,
            metadata=record.metadata
        )

        if record.id in self._id_to_idx:
            idx = self._id_to_idx[record.id]
            self._records[idx] = normalized_record
        else:
            self._id_to_idx[record.id] = len(self._records)
            self._records.append(normalized_record)

        self._dirty = True

    def _rebuild_matrix_if_dirty(self) -> None:
        """Consolidates vectors into a dense, C-contiguous NumPy array."""
        if self._dirty and self._records:
            vectors = [r.vector for r in self._records]
            # Stack into contiguous 2D array for BLAS SIMD optimization
            self._matrix = np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
            self._dirty = False

    def search(
        self,
        query_vector: np.ndarray,
        sec_context: SecurityContext,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Executes pre-filtered cosine similarity search matching tenant 
        and RBAC authorizations.
        """
        if not self._records:
            return []

        self._rebuild_matrix_if_dirty()

        # Step 1: Pre-Filtering Boundary (Tenant Lock + RBAC Isolation)
        candidate_indices: List[int] = []
        for idx, rec in enumerate(self._records):
            # Hard Tenant Isolation
            if rec.tenant_id != sec_context.tenant_id:
                continue
            
            # Role-Based Access Control (Set Intersection)
            if rec.allowed_roles and not (rec.allowed_roles & sec_context.user_roles):
                continue

            candidate_indices.append(idx)

        if not candidate_indices:
            return []

        # Step 2: Vector Normalization for Dot-Product Distance
        q_norm = np.linalg.norm(query_vector)
        norm_q = (query_vector / q_norm if q_norm > 0 else query_vector).astype(np.float32)

        # Step 3: Compute Vectorized Similarity over Filtered Matrix Subset
        sub_matrix = self._matrix[candidate_indices]  # Sub-slice C-contiguous array
        sim_scores = np.dot(sub_matrix, norm_q)       # Fast BLAS GEMV product

        # Step 4: Top-K Argpartition Heap Extraction
        k_actual = min(top_k, len(candidate_indices))
        if k_actual < len(candidate_indices):
            # Partition indices to locate top-k without sorting whole array: O(N)
            top_k_sub_idx = np.argpartition(-sim_scores, k_actual - 1)[:k_actual]
            # Sort only the top k subset: O(K log K)
            sorted_top_k = top_k_sub_idx[np.argsort(-sim_scores[top_k_sub_idx])]
        else:
            sorted_top_k = np.argsort(-sim_scores)

        # Step 5: Map back to SearchResult domain objects
        results = []
        for sub_idx in sorted_top_k:
            orig_idx = candidate_indices[sub_idx]
            rec = self._records[orig_idx]
            results.append(
                SearchResult(
                    id=rec.id,
                    score=float(sim_scores[sub_idx]),
                    metadata=rec.metadata
                )
            )

        return results