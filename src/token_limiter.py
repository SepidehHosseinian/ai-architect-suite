"""
Module 4: Token-Bucket Rate Limiter Middleware
Asynchronous rate limiting middleware enforcing dynamic token/request capacity limits.
"""

import asyncio
import time


class TokenBucketRateLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        """
        capacity: Maximum token capacity of bucket.
        refill_rate: Tokens added per second.
        """
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = float(refill_rate)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self):
        now = time.monotonic()
        delta = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
        self.last_refill = now

    async def acquire(self, tokens_requested: int = 1) -> bool:
        """Attempt to consume tokens immediately without waiting."""
        async with self._lock:
            self._refill()
            if self.tokens >= tokens_requested:
                self.tokens -= tokens_requested
                return True
            return False

    async def wait_for_tokens(self, tokens_requested: int = 1):
        """Block asynchronously until required tokens are replenished."""
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= tokens_requested:
                    self.tokens -= tokens_requested
                    return
                needed = tokens_requested - self.tokens
                wait_time = needed / self.refill_rate

            await asyncio.sleep(wait_time)


if __name__ == "__main__":
    async def main():
        limiter = TokenBucketRateLimiter(capacity=2, refill_rate=1.0)
        
        print("Acquire 1:", await limiter.acquire(1))
        print("Acquire 1:", await limiter.acquire(1))
        print("Acquire 1 (expect False):", await limiter.acquire(1))
        
        print("Waiting for tokens...")
        await limiter.wait_for_tokens(1)
        print("Acquired after waiting!")

    asyncio.run(main())