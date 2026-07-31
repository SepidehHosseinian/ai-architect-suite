"""
Module 2: Async Dynamic Micro-Batching Request Manager
Buffers incoming LLM request tasks into optimal dynamic batches to maximize throughput.
"""

import asyncio
import time
from typing import List, Any, Callable, Awaitable


class AsyncDynamicBatcher:
    def __init__(
        self, 
        batch_handler: Callable[[List[Any]], Awaitable[List[Any]]], 
        max_batch_size: int = 16, 
        max_wait_time: float = 0.05
    ):
        self.batch_handler = batch_handler
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task = None

    async def start(self):
        """Start background batch processor."""
        self.worker_task = asyncio.create_task(self._process_batches())

    async def stop(self):
        """Gracefully drain and stop worker."""
        if self.worker_task:
            self.worker_task.cancel()

    async def enqueue(self, item: Any) -> Any:
        """Enqueue single request item and await batched processing result."""
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((item, future))
        return await future

    async def _process_batches(self):
        while True:
            batch = []
            futures = []
            start_time = time.time()

            while len(batch) < self.max_batch_size:
                elapsed = time.time() - start_time
                timeout = max(0.0, self.max_wait_time - elapsed)

                try:
                    item, future = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break  # Batch window expired, flush available requests

            if batch:
                try:
                    results = await self.batch_handler(batch)
                    for fut, res in zip(futures, results):
                        if not fut.done():
                            fut.set_result(res)
                except Exception as exc:
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(exc)


if __name__ == "__main__":
    async def dummy_inference(inputs: List[str]) -> List[str]:
        print(f"--- Processing Batch of Size {len(inputs)} ---")
        return [f"Output for: {x}" for x in inputs]

    async def main():
        batcher = AsyncDynamicBatcher(dummy_inference, max_batch_size=4, max_wait_time=0.1)
        await batcher.start()

        # Fire 6 parallel requests
        tasks = [batcher.enqueue(f"req_{i}") for i in range(6)]
        results = await asyncio.gather(*tasks)
        print("Results:", results)

        await batcher.stop()

    asyncio.run(main())