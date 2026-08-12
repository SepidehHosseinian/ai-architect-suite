import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("AgenticSearchFSM")


# ---------------------------------------------------------------------------
# State & Domain Models
# ---------------------------------------------------------------------------

class SearchState(Enum):
    DECOMPOSE = auto()
    FETCH_ASYNC = auto()
    EVALUATE = auto()
    REFINE = auto()
    SYNTHESIZE = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class SearchContext:
    """Immutable state container tracking loop memory across transitions."""
    user_query: str
    max_loops: int = 3
    sufficiency_threshold: float = 0.85
    
    current_loop: int = 0
    sub_queries: List[str] = field(default_factory=list)
    fetched_documents: List[Dict[str, Any]] = field(default_factory=list)
    sufficiency_score: float = 0.0
    final_synthesis: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider Abstraction Interface (Search & Intelligence Engine)
# ---------------------------------------------------------------------------

class ISearchEngineProvider(ABC):
    """Abstract interface for parallel non-blocking vector/web searches."""
    @abstractmethod
    async def fetch_documents(self, query: str) -> List[Dict[str, Any]]:
        pass


class MockAsyncSearchEngine(ISearchEngineProvider):
    """Simulates multi-source non-blocking retrieval with latency overhead."""
    async def fetch_documents(self, query: str) -> List[Dict[str, Any]]:
        await asyncio.sleep(0.1)  # Simulate network/vector DB latency
        return [
            {"query": query, "content": f"Retrieved snippet for: {query}", "relevance": 0.8}
        ]


# ---------------------------------------------------------------------------
# Core Agentic FSM Engine
# ---------------------------------------------------------------------------

class AgenticSearchEngineFSM:
    """
    Asynchronous State Machine orchestrating adaptive, bounded agentic search.
    """
    def __init__(
        self, 
        search_provider: ISearchEngineProvider,
        sufficiency_threshold: float = 0.85,
        max_loops: int = 3
    ):
        self._provider = search_provider
        self._threshold = sufficiency_threshold
        self._max_loops = max_loops

    async def run(self, user_query: str) -> SearchContext:
        ctx = SearchContext(
            user_query=user_query,
            max_loops=self._max_loops,
            sufficiency_threshold=self._threshold
        )
        
        current_state = SearchState.DECOMPOSE
        
        while current_state not in (SearchState.COMPLETED, SearchState.FAILED):
            logger.info(f"FSM State Transition -> {current_state.name} | Loop: {ctx.current_loop}")
            
            try:
                if current_state == SearchState.DECOMPOSE:
                    current_state = await self._state_decompose(ctx)
                elif current_state == SearchState.FETCH_ASYNC:
                    current_state = await self._state_fetch_async(ctx)
                elif current_state == SearchState.EVALUATE:
                    current_state = await self._state_evaluate(ctx)
                elif current_state == SearchState.REFINE:
                    current_state = await self._state_refine(ctx)
                elif current_state == SearchState.SYNTHESIZE:
                    current_state = await self._state_synthesize(ctx)
            except Exception as exc:
                logger.error(f"FSM Processing Failure in {current_state.name}: {exc}")
                current_state = SearchState.FAILED
                
        return ctx

    # -----------------------------------------------------------------------
    # State Handlers
    # -----------------------------------------------------------------------

    async def _state_decompose(self, ctx: SearchContext) -> SearchState:
        """Deconstructs complex input query into focused sub-queries."""
        # Simulated LLM Query Decomposition
        ctx.sub_queries = [
            f"{ctx.user_query} - primary concepts",
            f"{ctx.user_query} - technical specs"
        ]
        return SearchState.FETCH_ASYNC

    async def _state_fetch_async(self, ctx: SearchContext) -> SearchState:
        """Executes non-blocking parallel fetches for all current sub-queries."""
        tasks = [self._provider.fetch_documents(q) for q in ctx.sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for res in results:
            if isinstance(res, list):
                ctx.fetched_documents.extend(res)
                
        return SearchState.EVALUATE

    async def _state_evaluate(self, ctx: SearchContext) -> SearchState:
        """Evaluates overall context sufficiency to determine loop exit or continuation."""
        ctx.current_loop += 1
        
        # Calculate dynamic sufficiency score based on document count and coverage
        doc_count = len(ctx.fetched_documents)
        ctx.sufficiency_score = min(1.0, doc_count * 0.3)  # Evaluator metric logic
        
        logger.info(f"Context Sufficiency Score: {ctx.sufficiency_score:.2f} (Target: {self._threshold})")
        
        # Early Exit / Termination Check
        if (ctx.sufficiency_score >= self._threshold) or (ctx.current_loop >= ctx.max_loops):
            return SearchState.SYNTHESIZE
        
        return SearchState.REFINE

    async def _state_refine(self, ctx: SearchContext) -> SearchState:
        """Generates targeted follow-up sub-queries based on missing information gaps."""
        # Focus sub-queries on specific gaps identified during evaluation
        ctx.sub_queries = [
            f"{ctx.user_query} - gap analysis iteration {ctx.current_loop}"
        ]
        return SearchState.FETCH_ASYNC

    async def _state_synthesize(self, ctx: SearchContext) -> SearchState:
        """Consolidates gathered documents into final structured response."""
        summary_snippets = [doc['content'] for doc in ctx.fetched_documents]
        ctx.final_synthesis = (
            f"Synthesized response for '{ctx.user_query}' from {len(summary_snippets)} context sources. "
            f"Sufficiency reached: {ctx.sufficiency_score:.2f} in {ctx.current_loop} loop(s)."
        )
        return SearchState.COMPLETED