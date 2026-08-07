"""
Async Resilient Retry Module using Tenacity.

Key Mechanics:
- Uses Tenacity for declarative, decorated retry behavior.
- Exponential backoff with Full Jitter to avoid thundering-herd issues.
- Catches transient rate limits (HTTP 429) and network failures.
"""

import asyncio
import logging
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# Set up logging to observe retry behavior
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom exceptions for simulated failures
class TransientRateLimitError(Exception):
    """Simulates HTTP 429 Too Many Requests."""
    pass


class PermanentAPIError(Exception):
    """Simulates non-retryable errors (e.g., HTTP 400 Bad Request)."""
    pass


# Target async inference function decorated with Tenacity retry logic
@retry(
    # Stop after 5 failed attempts
    stop=stop_after_attempt(5),
    # Exponential backoff with Full Jitter (multiplier=1s, max=10s)
    wait=wait_random_exponential(multiplier=1, max=10),
    # Only retry on transient rate limit or network errors
    retry=retry_if_exception_type(TransientRateLimitError),
    # Log details before sleeping between retries
    before_sleep=before_sleep_log(logger, logging.WARNING),
    # Reraise the exception if all retry attempts are exhausted
    reraise=True,
)
async def call_llm_inference_api(prompt: str) -> str:
    """Simulates calling a remote LLM endpoint subject to transient rate limiting."""
    # Simulate random failure behavior
    outcome = random.random()

    if outcome < 0.6:
        logger.info("Executing API request... -> [FAILED: 429 Rate Limit]")
        raise TransientRateLimitError("429 Too Many Requests: Rate limit exceeded.")
    elif outcome < 0.7:
        logger.info("Executing API request... -> [FAILED: 400 Bad Request]")
        raise PermanentAPIError("400 Bad Request: Invalid prompt schema.")

    logger.info("Executing API request... -> [SUCCESS: 200 OK]")
    return f"Response for prompt: '{prompt}'"


# --- Quick Test Verification Routine ---
async def main():
    print("=== Testing Tenacity Resilient Async Endpoint ===")
    try:
        response = await call_llm_inference_api("Explain p99 latency in vector search.")
        print(f"\nFinal Result: {response}")
    except TransientRateLimitError:
        print("\nFailed: Max retry attempts exhausted.")
    except PermanentAPIError as e:
        print(f"\nFailed: Encountered non-retryable exception ({e})")


if __name__ == "__main__":
    asyncio.run(main())