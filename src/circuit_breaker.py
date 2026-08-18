import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("CircuitBreakerLLM")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# State & Domain Contracts
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = auto()     # Healthy: Requests pass through
    OPEN = auto()       # Unhealthy: Requests fast-fail immediately
    HALF_OPEN = auto()  # Recovery Probe: Testing backend viability


class CircuitBreakerOpenException(Exception):
    """Raised when an execution request is blocked by an OPEN circuit."""
    pass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Immutable configuration setting system resilience boundaries."""
    failure_threshold: int = 5
    recovery_timeout_sec: float = 30.0
    max_retries: int = 3
    base_backoff_sec: float = 0.5
    max_backoff_sec: float = 8.0


# ---------------------------------------------------------------------------
# Core Circuit Breaker Implementation
# ---------------------------------------------------------------------------

class ResilientLLMCircuitBreaker:
    """
    Production-grade Async Circuit Breaker enforcing state transitions,
    jittered exponential retries, and fallback handling for remote LLM APIs.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig = CircuitBreakerConfig()
    ):
        self.name = name
        self.config = config

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_state_change = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def _allow_request(self) -> bool:
        """Determines if a request can proceed based on current FSM state."""
        async with self._lock:
            now = time.monotonic()

            if self._state == CircuitState.OPEN:
                if now - self._last_state_change >= self.config.recovery_timeout_sec:
                    self._state = CircuitState.HALF_OPEN
                    self._last_state_change = now
                    logger.warning(
                        f"Circuit [{self.name}] entering HALF-OPEN probe state."
                    )
                    return True
                return False

            return True

    async def _record_success(self) -> None:
        """Handles successful execution outcomes."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._last_state_change = time.monotonic()
                logger.info(f"Circuit [{self.name}] successfully recovered -> CLOSED.")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _record_failure(self) -> None:
        """Handles execution failure outcomes and evaluates state transitions."""
        async with self._lock:
            self._failure_count += 1
            now = time.monotonic()

            if (
                self._state == CircuitState.HALF_OPEN 
                or self._failure_count >= self.config.failure_threshold
            ):
                if self._state != CircuitState.OPEN:
                    self._state = CircuitState.OPEN
                    self._last_state_change = now
                    logger.error(
                        f"Circuit [{self.name}] TRIPPED -> OPEN. "
                        f"Consecutive Failures: {self._failure_count}"
                    )

    def _calculate_jittered_backoff(self, attempt: int) -> float:
        """Computes Exponential Backoff with Full Jitter."""
        calculated_backoff = self.config.base_backoff_sec * (2 ** attempt)
        capped_backoff = min(self.config.max_backoff_sec, calculated_backoff)
        return random.uniform(0, capped_backoff)

    async def call(
        self,
        func: Callable[[], Awaitable[T]],
        fallback: Optional[Callable[[], Awaitable[T]]] = None
    ) -> T:
        """
        Executes target LLM request within the circuit boundary.
        Enforces retry backoff logic and executes fallback on failure.
        """
        if not await self._allow_request():
            logger.warning(f"Circuit [{self.name}] is OPEN. Bypassing API call.")
            if fallback:
                return await fallback()
            raise CircuitBreakerOpenException(
                f"Circuit [{self.name}] is OPEN. Fast-failing request."
            )

        last_exception: Optional[Exception] = None

        # Exponential Backoff Retry Loop
        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func()
                await self._record_success()
                return result

            except Exception as exc:
                last_exception = exc
                logger.warning(
                    f"Circuit [{self.name}] Request Attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1} failed: {exc}"
                )

                if attempt < self.config.max_retries:
                    sleep_time = self._calculate_jittered_backoff(attempt)
                    await asyncio.sleep(sleep_time)

        # Retry exhaustion triggers circuit failure recording
        await self._record_failure()

        if fallback:
            logger.info(f"Circuit [{self.name}] invoking fallback strategy.")
            return await fallback()

        raise RuntimeError(
            f"Circuit [{self.name}] request failed after {self.config.max_retries + 1} attempts."
        ) from last_exception