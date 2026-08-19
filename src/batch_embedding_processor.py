import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("BatchEmbeddingProcessor")


# ---------------------------------------------------------------------------
# Domain Contracts & Data Transfer Objects
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingPayload:
    """Immutable domain contract representing an embedding task."""
    id: str
    tenant_id: str
    text: str
    retry_count: int = 0
    last_error: Optional[str] = None


@dataclass
class BatchEmbeddingResult:
    """Output contract coupling the input payload ID with its vector embedding."""
    id: str
    tenant_id: str
    embedding: np.ndarray


@dataclass(frozen=True)
class ProcessorConfig:
    """Configuration driving dynamic batching and retry behaviors."""
    max_batch_size: int = 64
    max_wait_time_sec: float = 0.02  # 20ms maximum batch accumulation window
    max_retries: int = 3
    base_backoff_sec: float = 0.5
    max_backoff_sec: float = 5.0


# ---------------------------------------------------------------------------
# Infrastructure Interfaces & Mock Implementation
# ---------------------------------------------------------------------------

class IEmbeddingProvider(ABC):
    """Abstract interface for remote or local embedding inference engines."""
    
    @abstractmethod
    async def generate_embeddings(
        self, texts: List[str]
    ) -> List[np.ndarray]:
        pass


class MockEmbeddingProvider(IEmbeddingProvider):
    """Simulates a remote embedding API with transient failure rates."""
    
    def __init__(self, dimension: int = 1536, failure_rate: float = 0.1):
        self._dim = dimension
        self._failure_rate = failure_rate

    async def generate_embeddings(
        self, texts: List[str]
    ) -> List[np.ndarray]:
        await asyncio.sleep(0.05)  # Simulate network and GPU execution latency
        
        # Simulate transient remote gateway errors
        if random.random() < self._failure_rate:
            raise RuntimeError("HTTP 429: Rate Limit Exceeded or Provider Timeout")

        return [
            np.random.randn(self._dim).astype(np.float32) for _ in texts
        ]


# ---------------------------------------------------------------------------
# Core Resilience Engine
# ---------------------------------------------------------------------------

class DeadLetterQueue:
    """Isolated storage layer for unprocessable or repeatedly failing payloads."""
    
    def __init__(self):
        self._dlq_store: List[EmbeddingPayload] = []
        self._lock = asyncio.Lock()

    async def push(self, payload: EmbeddingPayload) -> None:
        async with self._lock:
            self._dlq_store.append(payload)
            logger.error(
                f"DLQ_ISOLATION: Payload [{payload.id}] for Tenant [{payload.tenant_id}] "
                f"isolated after {payload.retry_count} retries. Error: {payload.last_error}"
            )

    @property
    def size(self) -> int:
        return len(self._dlq_store)


class ResilientBatchEmbeddingProcessor:
    """
    High-throughput async processor executing dynamic micro-batching, 
    jittered retries, and DLQ handling for vector embeddings.
    """

    def __init__(
        self,
        provider: IEmbeddingProvider,
        config: ProcessorConfig = ProcessorConfig()
    ):
        self._provider = provider
        self._config = config
        
        # Concurrency & Work Queues
        self._ingestion_queue: asyncio.Queue[EmbeddingPayload] = asyncio.Queue()
        self._results_store: Dict[str, BatchEmbeddingResult] = {}
        self.dlq = DeadLetterQueue()
        
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self) -> None:
        """Boots background batch execution worker loop."""
        self._is_running = True
        self._worker_task = asyncio.create_task(self._batch_processing_loop())
        logger.info("Batch Embedding Engine initialized successfully.")

    async def stop(self) -> None:
        """Gracefully drains active ingestion queues and terminates worker processes."""
        self._is_running = False
        if self._worker_task:
            await self._ingestion_queue.join()
            self._worker_task.cancel()
            self._worker_task = None
        logger.info("Batch Embedding Engine safely shutdown.")

    async def submit(self, payload: EmbeddingPayload) -> None:
        """Non-blocking producer boundary for ingesting document payloads."""
        await self._ingestion_queue.put(payload)

    def _compute_jittered_backoff(self, retry_count: int) -> float:
        """Calculates Exponential Backoff with Full Jitter."""
        backoff = min(
            self._config.max_backoff_sec, 
            self._config.base_backoff_sec * (2 ** retry_count)
        )
        return random.uniform(0, backoff)

    async def _batch_processing_loop(self) -> None:
        """Main dynamic sliding-window batching loop."""
        while self._is_running or not self._ingestion_queue.empty():
            batch: List[EmbeddingPayload] = []
            start_time = time.monotonic()

            # Dynamic accumulation loop: collect until max_batch_size OR max_wait_time_sec hit
            while len(batch) < self._config.max_batch_size:
                elapsed = time.monotonic() - start_time
                remaining_time = self._config.max_wait_time_sec - elapsed

                if remaining_time <= 0:
                    break  # Window wait time threshold reached

                try:
                    payload = await asyncio.wait_for(
                        self._ingestion_queue.get(), 
                        timeout=remaining_time
                    )
                    batch.append(payload)
                except asyncio.TimeoutError:
                    break  # Window expired without reaching max_batch_size

            if not batch:
                await asyncio.sleep(0.005)  # Yield CPU during low-traffic periods
                continue

            # Process aggregated micro-batch
            await self._process_batch_with_retry(batch)

            # Mark processed items as task-done for queue join semantics
            for _ in batch:
                self._ingestion_queue.task_done()

    async def _process_batch_with_retry(self, batch: List[EmbeddingPayload]) -> None:
        """Executes inference for a micro-batch with fault-handling and re-queuing logic."""
        texts = [item.text for item in batch]
        
        try:
            # Batch inference execution
            vectors = await self._provider.generate_embeddings(texts)
            
            # Map results upon successful batch execution
            for item, vector in zip(batch, vectors):
                self._results_store[item.id] = BatchEmbeddingResult(
                    id=item.id,
                    tenant_id=item.tenant_id,
                    embedding=vector
                )

        except Exception as exc:
            logger.warning(
                f"Batch execution failed for {len(batch)} items: {exc}. "
                "Evaluating retry/DLQ routing..."
            )
            
            # Handle individual items within the failed batch
            for item in batch:
                item.retry_count += 1
                item.last_error = str(exc)

                if item.retry_count > self._config.max_retries:
                    # Retry exhaustion: Isolate to Dead Letter Queue
                    await self.dlq.push(item)
                else:
                    # Re-queue payload with jittered exponential delay
                    asyncio.create_task(self._requeue_with_delay(item))

    async def _requeue_with_delay(self, payload: EmbeddingPayload) -> None:
        """Applies jittered backoff delay before re-injecting payload into ingestion pipeline."""
        delay = self._compute_jittered_backoff(payload.retry_count)
        logger.info(
            f"Re-queuing Payload [{payload.id}] (Attempt {payload.retry_count}/"
            f"{self._config.max_retries}) in {delay:.2f}s..."
        )
        await asyncio.sleep(delay)
        await self._ingestion_queue.put(payload)