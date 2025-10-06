"""
Batch processor for optimizing similar LLM requests

This module implements request batching to reduce redundant LLM calls
by detecting and grouping similar requests within a time window.
"""

import asyncio
import hashlib
import time
import logging
from typing import Dict, List, Optional, Callable, Awaitable, Set
from dataclasses import dataclass, field
from uuid import uuid4

from ..config import settings
from .models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class BatchedRequest:
    """Represents a request waiting to be batched"""
    id: str = field(default_factory=lambda: str(uuid4()))
    request: LLMRequest = None
    similarity_hash: str = ""
    created_at: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)
    timeout: float = 30.0


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance"""
    total_requests: int = 0
    batched_requests: int = 0
    saved_llm_calls: int = 0
    batch_hits: int = 0
    batch_misses: int = 0
    average_batch_size: float = 0.0
    total_batches_processed: int = 0


class RequestBatchProcessor:
    """
    Batch processor for similar LLM requests

    Detects similar requests within a time window and processes them together,
    sharing the result across all similar requests.
    """

    def __init__(
        self,
        processor: Callable[[LLMRequest], Awaitable[LLMResponse]],
        enabled: bool = True,
        batch_window_ms: int = 100,
        max_batch_size: int = 5,
        similarity_threshold: float = 0.9
    ):
        self.processor = processor
        self.enabled = enabled
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.similarity_threshold = similarity_threshold

        # Pending requests grouped by similarity hash
        self.pending: Dict[str, List[BatchedRequest]] = {}

        # Currently processing batches
        self.processing: Set[str] = set()

        # Metrics
        self.metrics = BatchMetrics()

        # Background task for batch processing
        self._batch_task: Optional[asyncio.Task] = None
        self._running = False

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the batch processor"""
        if self._running:
            logger.warning("Batch processor already running")
            return

        self._running = True
        self._batch_task = asyncio.create_task(self._batch_worker())
        logger.info(
            f"Batch processor started (window={self.batch_window_ms}ms, "
            f"max_size={self.max_batch_size}, enabled={self.enabled})"
        )

    async def stop(self):
        """Stop the batch processor"""
        if not self._running:
            return

        self._running = False

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Cancel any pending requests
        async with self._lock:
            for batch in self.pending.values():
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            asyncio.CancelledError("Batch processor stopped")
                        )

            self.pending.clear()

        logger.info("Batch processor stopped")

    async def process_request(
        self,
        request: LLMRequest,
        timeout: float = 30.0
    ) -> LLMResponse:
        """
        Process a request with batching optimization

        If batching is disabled or request is unique, processes immediately.
        Otherwise, waits for similar requests and batches them together.
        """
        if not self.enabled:
            # Batching disabled, process directly
            return await self.processor(request)

        self.metrics.total_requests += 1

        # Generate similarity hash
        similarity_hash = self._generate_similarity_hash(request)

        # Check if already processing this hash
        async with self._lock:
            if similarity_hash in self.processing:
                # Find the existing batch and join it
                if similarity_hash in self.pending:
                    for batched_req in self.pending[similarity_hash]:
                        if not batched_req.future.done():
                            # Share the result from processing batch
                            self.metrics.batch_hits += 1
                            return await batched_req.future

        # Create batched request
        batched_request = BatchedRequest(
            request=request,
            similarity_hash=similarity_hash,
            timeout=timeout
        )

        # Add to pending batch
        async with self._lock:
            if similarity_hash not in self.pending:
                self.pending[similarity_hash] = []

            self.pending[similarity_hash].append(batched_request)

            # Check if batch is ready to process immediately
            batch = self.pending[similarity_hash]
            if len(batch) >= self.max_batch_size:
                # Process batch immediately (it's full)
                asyncio.create_task(self._process_batch(similarity_hash))

        # Wait for result
        try:
            response = await asyncio.wait_for(
                batched_request.future,
                timeout=timeout
            )
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Batch request {batched_request.id[:8]} timed out")
            # Timeout, process directly
            async with self._lock:
                if similarity_hash in self.pending:
                    if batched_request in self.pending[similarity_hash]:
                        self.pending[similarity_hash].remove(batched_request)

            # Fall back to direct processing
            return await self.processor(request)

    async def _batch_worker(self):
        """Background worker that processes batches"""
        logger.debug("Batch worker started")

        while self._running:
            try:
                # Wait for batch window
                await asyncio.sleep(self.batch_window_ms / 1000.0)

                # Process ready batches
                await self._process_ready_batches()

            except asyncio.CancelledError:
                logger.debug("Batch worker cancelled")
                break
            except Exception as e:
                logger.error(f"Batch worker error: {e}")
                await asyncio.sleep(0.1)

        logger.debug("Batch worker stopped")

    async def _process_ready_batches(self):
        """Process all batches that are ready"""
        ready_hashes = []

        async with self._lock:
            current_time = time.time()

            # Find batches ready to process
            for similarity_hash, batch in list(self.pending.items()):
                if not batch:
                    continue

                # Check if batch window has elapsed
                oldest_request = min(batch, key=lambda r: r.created_at)
                elapsed_ms = (current_time - oldest_request.created_at) * 1000

                if elapsed_ms >= self.batch_window_ms:
                    ready_hashes.append(similarity_hash)

        # Process ready batches (outside lock to avoid blocking)
        for similarity_hash in ready_hashes:
            asyncio.create_task(self._process_batch(similarity_hash))

    async def _process_batch(self, similarity_hash: str):
        """Process a batch of similar requests"""
        # Mark as processing
        async with self._lock:
            if similarity_hash in self.processing:
                # Already processing
                return

            if similarity_hash not in self.pending:
                return

            batch = self.pending.pop(similarity_hash)
            if not batch:
                return

            self.processing.add(similarity_hash)

        try:
            # Process the first request (they're all similar)
            representative_request = batch[0].request

            logger.debug(
                f"Processing batch of {len(batch)} similar requests "
                f"(hash={similarity_hash[:8]})"
            )

            # Execute the request
            start_time = time.time()
            response = await self.processor(representative_request)
            processing_time = (time.time() - start_time) * 1000

            # Share result with all requests in batch
            for batched_req in batch:
                if not batched_req.future.done():
                    batched_req.future.set_result(response)

            # Update metrics
            batch_size = len(batch)
            self.metrics.batched_requests += batch_size
            self.metrics.saved_llm_calls += (batch_size - 1)  # Saved calls
            self.metrics.total_batches_processed += 1

            # Update average batch size
            total_batched = self.metrics.batched_requests
            self.metrics.average_batch_size = (
                total_batched / self.metrics.total_batches_processed
            )

            logger.info(
                f"Batch processed: {batch_size} requests, "
                f"{processing_time:.0f}ms, saved {batch_size - 1} LLM calls"
            )

        except Exception as e:
            logger.error(f"Batch processing error: {e}")

            # Fail all requests in batch
            for batched_req in batch:
                if not batched_req.future.done():
                    batched_req.future.set_exception(e)

        finally:
            # Remove from processing
            async with self._lock:
                self.processing.discard(similarity_hash)

    def _generate_similarity_hash(self, request: LLMRequest) -> str:
        """
        Generate a hash to identify similar requests

        Considers:
        - System prompt
        - First few messages (ignoring exact wording variations)
        - Request parameters (temperature, structured output, etc.)
        """
        hash_components = []

        # Include system prompt
        if request.system_prompt:
            hash_components.append(request.system_prompt)

        # Include first 3 messages (or all if less than 3)
        num_messages = min(3, len(request.messages))
        for i in range(num_messages):
            msg = request.messages[i]
            hash_components.append(f"{msg.role.value}:{msg.content}")

        # Include key parameters that affect response
        hash_components.append(f"temp:{request.temperature:.2f}")
        hash_components.append(f"structured:{request.structured_output}")
        hash_components.append(f"max_tokens:{request.max_tokens}")

        # Generate hash
        hash_input = "|".join(hash_components)
        similarity_hash = hashlib.md5(hash_input.encode()).hexdigest()

        return similarity_hash

    async def get_metrics(self) -> Dict[str, any]:
        """Get batch processing metrics"""
        async with self._lock:
            pending_count = sum(len(batch) for batch in self.pending.values())
            processing_count = len(self.processing)

        saved_percentage = 0.0
        if self.metrics.total_requests > 0:
            saved_percentage = (
                self.metrics.saved_llm_calls / self.metrics.total_requests * 100
            )

        return {
            "enabled": self.enabled,
            "total_requests": self.metrics.total_requests,
            "batched_requests": self.metrics.batched_requests,
            "saved_llm_calls": self.metrics.saved_llm_calls,
            "saved_percentage": saved_percentage,
            "batch_hits": self.metrics.batch_hits,
            "batch_misses": self.metrics.batch_misses,
            "total_batches_processed": self.metrics.total_batches_processed,
            "average_batch_size": self.metrics.average_batch_size,
            "pending_batches": len(self.pending),
            "pending_requests": pending_count,
            "processing_batches": processing_count
        }

    async def health_check(self) -> Dict[str, any]:
        """Check batch processor health"""
        metrics = await self.get_metrics()

        status = "healthy"
        issues = []

        # Check if too many pending requests
        if metrics["pending_requests"] > 50:
            status = "degraded"
            issues.append(f"High pending requests: {metrics['pending_requests']}")

        # Check if processing is stuck
        if metrics["processing_batches"] > 10:
            status = "degraded"
            issues.append(f"Many processing batches: {metrics['processing_batches']}")

        return {
            "status": status,
            "issues": issues,
            "running": self._running,
            **metrics
        }
