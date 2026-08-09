import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

# Configure standard logger to output structured JSON
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain Models & Interfaces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TokenMetricsEvent:
    """Immutable event capturing a single LLM generation lifecycle."""
    tenant_id: str
    model_variant: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def throughput_tps(self) -> float:
        """Calculates generation throughput (Tokens Per Second)."""
        generation_time_s = (self.total_latency_ms - self.ttft_ms) / 1000.0
        if generation_time_s <= 0:
            return 0.0
        return self.completion_tokens / generation_time_s


class ITelemetryExporter(ABC):
    """Abstract interface for metrics exportation (OpenTelemetry, JSON, etc.)"""
    @abstractmethod
    async def export(self, payload: Dict) -> None:
        pass


class StructuredJSONExporter(ITelemetryExporter):
    """Emits structured JSON compatible with FluentBit / OTel Collectors."""
    async def export(self, payload: Dict) -> None:
        # In a real environment, this might write to stdout for a DaemonSet log 
        # collector, or push to an OTel gRPC endpoint.
        logger.info(json.dumps(payload))


# ---------------------------------------------------------------------------
# Core Telemetry Monitor
# ---------------------------------------------------------------------------

class AsyncTelemetryMonitor:
    """
    Non-blocking telemetry monitor. Ingests events asynchronously and computes
    rolling p50/p99 metrics per tenant and model variant.
    """
    def __init__(
        self, 
        exporter: ITelemetryExporter, 
        queue_size: int = 5000,
        window_size: int = 1000
    ):
        self._exporter = exporter
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._window_size = window_size
        
        # State: composite key (tenant_id, model) -> bounded deque of latencies
        self._ttft_history: Dict[Tuple[str, str], deque] = {}
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Starts the background processing worker."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())

    async def stop(self) -> None:
        """Gracefully drains the queue and stops the worker."""
        if self._worker_task:
            await self._queue.join()
            self._worker_task.cancel()
            self._worker_task = None

    def record_event(self, event: TokenMetricsEvent) -> None:
        """
        Fire-and-forget ingestion. Called from the critical path.
        Will drop events rather than block if the system is overwhelmed.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Backpressure strategy: drop telemetry, protect the LLM request
            pass

    def _update_and_calculate_percentiles(self, event: TokenMetricsEvent) -> Dict[str, float]:
        """Maintains rolling windows and computes p50 / p99."""
        key = (event.tenant_id, event.model_variant)
        
        if key not in self._ttft_history:
            self._ttft_history[key] = deque(maxlen=self._window_size)
            
        self._ttft_history[key].append(event.ttft_ms)
        
        # Compute percentiles efficiently using numpy
        latencies = list(self._ttft_history[key])
        p50, p99 = np.percentile(latencies, [50, 99])
        
        return {"ttft_p50_ms": round(p50, 2), "ttft_p99_ms": round(p99, 2)}

    async def _process_queue(self) -> None:
        """Background worker loop consuming events."""
        while True:
            try:
                event: TokenMetricsEvent = await self._queue.get()
                
                percentiles = self._update_and_calculate_percentiles(event)
                
                # Construct OTel-compatible structured payload
                payload = {
                    "event_type": "llm_generation_metrics",
                    "timestamp": event.timestamp,
                    "tenant_id": event.tenant_id,
                    "model_variant": event.model_variant,
                    "metrics": {
                        "prompt_tokens": event.prompt_tokens,
                        "completion_tokens": event.completion_tokens,
                        "ttft_ms": event.ttft_ms,
                        "total_latency_ms": event.total_latency_ms,
                        "throughput_tps": round(event.throughput_tps, 2),
                        **percentiles
                    }
                }
                
                await self._exporter.export(payload)
                self._queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Catch-all to prevent the background worker from dying silently
                logger.error(json.dumps({"error": "Telemetry processing failed", "details": str(e)}))