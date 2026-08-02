import time
import threading
from typing import Dict, Tuple, Optional


class TokenBucket:
    """
    Thread-safe individual Token Bucket using Lazy Evaluation.
    """
    __slots__ = ('capacity', 'fill_rate', 'tokens', 'last_refill', '_lock')

    def __init__(self, capacity: float, fill_rate: float):
        """
        :param capacity: Maximum bucket size (burst capacity).
        :param fill_rate: Tokens added per second.
        """
        self.capacity: float = float(capacity)
        self.fill_rate: float = float(fill_rate)
        self.tokens: float = float(capacity)
        self.last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens_requested: float = 1.0) -> bool:
        """
        Consumes requested tokens if available after refilling via lazy evaluation.
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        with self._lock:
            now = time.monotonic()
            
            # 1. Lazy Refill Calculation: Δt * fill_rate
            elapsed = now - self.last_refill
            if elapsed > 0:
                refill_amount = elapsed * self.fill_rate
                self.tokens = min(self.capacity, self.tokens + refill_amount)
                self.last_refill = now

            # 2. Token Consumption Check
            if self.tokens >= tokens_requested:
                self.tokens -= tokens_requested
                return True
            
            return False


class RateLimiterManager:
    """
    Extensible client manager for API Proxy Middleware (FastAPI / vLLM routing).
    Manages per-client/per-API-key bucket instances dynamically.
    """
    def __init__(self, default_capacity: float = 100.0, default_fill_rate: float = 10.0):
        self.default_capacity = default_capacity
        self.default_fill_rate = default_fill_rate
        self._buckets: Dict[str, TokenBucket] = {}
        self._manager_lock = threading.Lock()

    def get_bucket(self, client_id: str, custom_limits: Optional[Tuple[float, float]] = None) -> TokenBucket:
        """
        Retrieves or initializes a client's bucket in a thread-safe manner.
        """
        if client_id not in self._buckets:
            with self._manager_lock:
                # Double-checked locking pattern
                if client_id not in self._buckets:
                    cap, rate = custom_limits if custom_limits else (self.default_capacity, self.default_fill_rate)
                    self._buckets[client_id] = TokenBucket(capacity=cap, fill_rate=rate)
        
        return self._buckets[client_id]

    def allow_request(self, client_id: str, tokens_required: float = 1.0) -> bool:
        """
        Primary middleware interface for API proxy routing.
        """
        bucket = self.get_bucket(client_id)
        return bucket.consume(tokens_required)
    
from fastapi import FastAPI, Request, HTTPException, status

app = FastAPI()
limiter = RateLimiterManager(default_capacity=50, default_fill_rate=10)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Extract API Key or IP address
    client_id = request.headers.get("X-API-Key") or request.client.host

    # 1 token per HTTP request (or dynamically scale based on estimated payload size)
    if not limiter.allow_request(client_id=client_id, tokens_required=1.0):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later."
        )

    response = await call_next(request)
    return response    
    