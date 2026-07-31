"""
Module 5: Resilient State Machine Engine with Exponential Retries & Circuit Breaking
Fault-tolerant execution loop for multi-step agent pipelines.
"""

import asyncio
import time
from typing import Callable, Awaitable, Dict, Any, Optional


class CircuitBreakerOpenException(Exception):
    pass


class ResilientStateMachine:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5, failure_threshold: int = 3):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.failure_threshold = failure_threshold
        
        self.consecutive_failures = 0
        self.circuit_open = False
        self.circuit_open_until = 0.0

    async def execute_step(
        self, 
        step_fn: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]], 
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes state transaction with circuit check and retries."""
        
        # Check circuit state
        if self.circuit_open:
            if time.time() < self.circuit_open_until:
                raise CircuitBreakerOpenException("Circuit breaker OPEN. Request rejected.")
            else:
                self.circuit_open = False  # Reset to half-open state

        attempt = 0
        delay = 0.5

        while attempt < self.max_retries:
            try:
                result_state = await step_fn(state)
                self.consecutive_failures = 0  # Success reset
                return result_state
            except Exception as e:
                attempt += 1
                self.consecutive_failures += 1
                print(f"[Warning] Step failed (Attempt {attempt}/{self.max_retries}): {e}")

                if self.consecutive_failures >= self.failure_threshold:
                    self.circuit_open = True
                    self.circuit_open_until = time.time() + 10.0  # Trip for 10s
                    print("[ALERT] Circuit breaker TRIPPED!")
                    raise e

                if attempt >= self.max_retries:
                    raise e

                await asyncio.sleep(delay)
                delay *= self.backoff_factor


if __name__ == "__main__":
    async def flaky_step(state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("count", 0) < 2:
            state["count"] = state.get("count", 0) + 1
            raise ValueError("Transient API Error")
        state["status"] = "SUCCESS"
        return state

    async def main():
        sm = ResilientStateMachine(max_retries=3)
        res = await sm.execute_step(flaky_step, {"count": 0})
        print("Final State:", res)

    asyncio.run(main())