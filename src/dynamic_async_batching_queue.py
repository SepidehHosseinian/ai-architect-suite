import asyncio
import time
from typing import Generic, List, Optional, Tuple, TypeVar

ReqT = TypeVar("ReqT")
ResT = TypeVar("ResT")


class DynamicAsyncBatcher(Generic[ReqT, ResT]):
    """
    An asynchronous queue that dynamically aggregates individual client requests 
    into batches based on target capacity caps or maximum latency bounds.
    """

    def __init__(self, max_batch_size: int = 4, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        # Holds tuples of (request_payload, response_future)
        self._queue: asyncio.Queue[Tuple[ReqT, asyncio.Future[ResT]]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Starts the background worker loop."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self) -> None:
        """Gracefully shuts down the worker process."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, request_payload: ReqT) -> ResT:
        """
        Public endpoint for client calls. Enqueues payload with a future
        and awaits completion asynchronously.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ResT] = loop.create_future()

        await self._queue.put((request_payload, future))
        return await future

    async def _mock_batch_inference(self, requests: List[ReqT]) -> List[ResT]:
        """Simulates downstream GPU batch inference pass."""
        await asyncio.sleep(0.05)  # Simulate execution latency
        return [f"Processed: '{req}'" for req in requests]

    async def _batch_worker(self) -> None:
        """Continuous collection loop executing dynamic batch aggregation."""
        while True:
            # Step 1: Idle state — block until the first item arrives (0% CPU overhead)
            first_req, first_fut = await self._queue.get()
            batch: List[Tuple[ReqT, asyncio.Future[ResT]]] = [(first_req, first_fut)]
            self._queue.task_done()

            start_time = asyncio.get_running_loop().time()

            # Step 2: Accumulate additional requests up to capacity or dynamic SLA timeout
            while len(batch) < self.max_batch_size:
                elapsed = asyncio.get_running_loop().time() - start_time
                remaining_time = self.max_wait_time - elapsed

                if remaining_time <= 0:
                    break

                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining_time
                    )
                    batch.append(item)
                    self._queue.task_done()
                except asyncio.TimeoutError:
                    # Timeout limit reached; proceed to process current batch
                    break

            # Step 3: Extract payloads, run batched inference, and fulfill caller futures
            requests = [item[0] for item in batch]
            futures = [item[1] for item in batch]

            try:
                results = await self._mock_batch_inference(requests)
                for fut, res in zip(futures, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as exc:
                # Error Isolation: Fulfill futures with exception on model failure
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)