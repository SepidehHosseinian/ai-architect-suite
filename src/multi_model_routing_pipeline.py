import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MultiModelRouter")


# ---------------------------------------------------------------------------
# Domain Models & Enums
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = auto()     # Normal operations: Traffic flows through
    OPEN = auto()       # Tripped: Fast-fail or redirect to fallback
    HALF_OPEN = auto()  # Trial state: Testing provider recovery


class ModelProviderType(Enum):
    VLLM_LOCAL = "vllm_local"
    SLA_SAAS_PRIMARY = "saas_primary"
    SLA_SAAS_FALLBACK = "saas_fallback"


@dataclass(frozen=True)
class ModelRequest:
    """Immutable model invocation contract."""
    request_id: str
    tenant_id: str
    prompt: str
    required_capability: str
    max_tokens: int = 512


@dataclass
class ModelResponse:
    """Standardized response payload wrapper."""
    request_id: str
    provider_used: str
    content: str
    latency_ms: float
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Resiliency: Circuit Breaker FSM
# ---------------------------------------------------------------------------

class ModelCircuitBreaker:
    """
    Non-blocking, concurrency-safe Circuit Breaker guarding an individual 
    model provider endpoint.
    """

    def __init__(
        self,
        provider_id: str,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 30.0
    ):
        self.provider_id = provider_id
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_sec

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_state_change = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def allow_request(self) -> bool:
        """Determines if a request can proceed based on the current state."""
        async with self._lock:
            now = time.monotonic()
            if self._state == CircuitState.OPEN:
                # Evaluate transition from OPEN to HALF_OPEN
                if now - self._last_state_change > self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._last_state_change = now
                    logger.warning(
                        f"Circuit [{self.provider_id}] entering HALF-OPEN probe state."
                    )
                    return True
                return False
            return True

    async def record_result(self, success: bool) -> None:
        """Updates internal FSM based on request outcome."""
        async with self._lock:
            now = time.monotonic()
            if success:
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._last_state_change = now
                    logger.info(f"Circuit [{self.provider_id}] recovered -> CLOSED.")
                else:
                    self._failure_count = 0
            else:
                self._failure_count += 1
                if (
                    self._failure_count >= self._failure_threshold 
                    and self._state != CircuitState.OPEN
                ):
                    self._state = CircuitState.OPEN
                    self._last_state_change = now
                    logger.error(
                        f"Circuit [{self.provider_id}] TRIPPED -> OPEN. "
                        f"Failures: {self._failure_count}"
                    )


# ---------------------------------------------------------------------------
# Provider Abstraction & Dynamic Router
# ---------------------------------------------------------------------------

class IModelProvider(ABC):
    @property
    @abstractmethod
    def provider_id(self) -> str:
        pass

    @abstractmethod
    async def invoke(self, request: ModelRequest) -> str:
        pass


class DynamicModelRouter:
    """Selects target model execution routes based on request requirements."""

    def __init__(self, provider_map: Dict[str, IModelProvider]):
        self._providers = provider_map

    def resolve_route(self, request: ModelRequest) -> List[IModelProvider]:
        """
        Returns an ordered primary-to-fallback strategy list 
        based on system capabilities.
        """
        # Example routing policy: Route to high-throughput local engine first,
        # with fallback to hosted SaaS endpoints.
        if request.required_capability == "code_gen":
            primary = self._providers.get(ModelProviderType.VLLM_LOCAL.value)
            fallback = self._providers.get(ModelProviderType.SLA_SAAS_PRIMARY.value)
        else:
            primary = self._providers.get(ModelProviderType.SLA_SAAS_PRIMARY.value)
            fallback = self._providers.get(ModelProviderType.SLA_SAAS_FALLBACK.value)

        return [p for p in [primary, fallback] if p is not None]


# ---------------------------------------------------------------------------
# Core Multi-Model Orchestration Pipeline
# ---------------------------------------------------------------------------

class MultiModelRoutingPipeline:
    """
    Async orchestrator combining dynamic routing, circuit breaking, 
    and fault-tolerant execution.
    """

    def __init__(self, router: DynamicModelRouter):
        self._router = router
        self._breakers: Dict[str, ModelCircuitBreaker] = {}

    def register_provider_breaker(
        self, provider_id: str, breaker: ModelCircuitBreaker
    ) -> None:
        self._breakers[provider_id] = breaker

    async def execute(self, request: ModelRequest) -> ModelResponse:
        route_chain = self._router.resolve_route(request)
        if not route_chain:
            raise RuntimeError(f"No active routes configured for: {request}")

        last_exception: Optional[Exception] = None

        for idx, provider in enumerate(route_chain):
            provider_id = provider.provider_id
            breaker = self._breakers.get(
                provider_id, ModelCircuitBreaker(provider_id)
            )

            # Check Circuit State
            if not await breaker.allow_request():
                logger.warning(
                    f"Skipping route [{provider_id}]: Circuit Breaker OPEN."
                )
                continue

            start_time = time.monotonic()
            try:
                # Execute request against target provider
                content = await provider.invoke(request)
                latency_ms = (time.monotonic() - start_time) * 1000.0

                await breaker.record_result(success=True)

                return ModelResponse(
                    request_id=request.request_id,
                    provider_used=provider_id,
                    content=content,
                    latency_ms=round(latency_ms, 2),
                    is_fallback=(idx > 0)
                )

            except Exception as exc:
                latency_ms = (time.monotonic() - start_time) * 1000.0
                await breaker.record_result(success=False)
                logger.error(
                    f"Provider [{provider_id}] failed after {latency_ms:.2f}ms: {exc}"
                )
                last_exception = exc
                # Fall through to the next candidate in the routing chain

        raise RuntimeError(
            f"All model routes exhausted for request {request.request_id}. "
            f"Last error: {last_exception}"
        )