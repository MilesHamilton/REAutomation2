import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from .models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class RequestPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QueuedRequest:
    id: str = field(default_factory=lambda: str(uuid4()))
    request: LLMRequest = None
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    timeout: float = 30.0
    retries_remaining: int = 3
    callback: Optional[Callable] = None
    future: Optional[asyncio.Future] = field(default_factory=asyncio.Future)
    agent_type: Optional[str] = None


@dataclass
class QueueMetrics:
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timed_out_requests: int = 0
    retried_requests: int = 0
    current_queue_size: int = 0
    peak_queue_size: int = 0
    average_wait_time: float = 0.0
    average_processing_time: float = 0.0


class LLMRequestQueue:
    """
    Priority queue for managing concurrent LLM requests
    """

    def __init__(self, max_concurrent: int = 5, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size

        # Priority queues for different priorities
        self.queues = {
            RequestPriority.CRITICAL: asyncio.Queue(),
            RequestPriority.HIGH: asyncio.Queue(),
            RequestPriority.NORMAL: asyncio.Queue(),
            RequestPriority.LOW: asyncio.Queue()
        }

        # Currently processing requests
        self.processing: Dict[str, QueuedRequest] = {}

        # Request metrics
        self.metrics = QueueMetrics()

        # Processing workers
        self.workers: List[asyncio.Task] = []
        self.is_running = False

        # LLM processor function
        self.processor: Optional[Callable[[LLMRequest], Awaitable[LLMResponse]]] = None

    async def start(self, processor: Callable[[LLMRequest], Awaitable[LLMResponse]]):
        """Start the queue processing workers"""
        if self.is_running:
            logger.warning("Queue manager is already running")
            return

        self.processor = processor
        self.is_running = True

        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

        logger.info(f"Started {len(self.workers)} queue processing workers")

    async def stop(self):
        """Stop the queue processing workers"""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        self.workers.clear()
        logger.info("Stopped all queue processing workers")

    async def enqueue(
        self,
        request: LLMRequest,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 30.0,
        agent_type: str = None
    ) -> asyncio.Future:
        """
        Enqueue an LLM request for processing

        Returns a Future that will contain the LLMResponse
        """
        # Check queue capacity
        total_queued = sum(queue.qsize() for queue in self.queues.values())
        if total_queued >= self.max_queue_size:
            raise asyncio.QueueFull(f"Queue is full (max size: {self.max_queue_size})")

        # Create queued request
        queued_request = QueuedRequest(
            request=request,
            priority=priority,
            timeout=timeout,
            agent_type=agent_type
        )

        # Add to appropriate priority queue
        await self.queues[priority].put(queued_request)

        # Update metrics
        self.metrics.total_requests += 1
        current_size = total_queued + 1
        self.metrics.current_queue_size = current_size
        self.metrics.peak_queue_size = max(self.metrics.peak_queue_size, current_size)

        logger.debug(f"Enqueued request {queued_request.id[:8]} with priority {priority}")

        return queued_request.future

    async def _worker(self, worker_id: str):
        """Worker task for processing queued requests"""
        logger.debug(f"Worker {worker_id} started")

        while self.is_running:
            try:
                # Get next request from priority queues
                queued_request = await self._get_next_request()

                if queued_request is None:
                    continue

                # Check if request has timed out while waiting
                wait_time = time.time() - queued_request.created_at
                if wait_time > queued_request.timeout:
                    self._handle_timeout(queued_request)
                    continue

                # Process the request
                self.processing[queued_request.id] = queued_request
                await self._process_request(queued_request, worker_id)

            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)  # Brief pause before retrying

        logger.debug(f"Worker {worker_id} stopped")

    async def _get_next_request(self) -> Optional[QueuedRequest]:
        """Get next request from priority queues"""
        # Check queues in priority order
        for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH,
                        RequestPriority.NORMAL, RequestPriority.LOW]:
            queue = self.queues[priority]

            try:
                # Try to get request with short timeout
                queued_request = await asyncio.wait_for(queue.get(), timeout=0.1)
                return queued_request
            except asyncio.TimeoutError:
                continue

        return None

    async def _process_request(self, queued_request: QueuedRequest, worker_id: str):
        """Process a single request"""
        request_id = queued_request.id[:8]
        start_time = time.time()

        try:
            logger.debug(f"Worker {worker_id} processing request {request_id}")

            # Process the request using the configured processor
            if not self.processor:
                raise RuntimeError("No processor configured")

            # Process with timeout
            response = await asyncio.wait_for(
                self.processor(queued_request.request),
                timeout=queued_request.timeout
            )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Update response with processing metadata
            response.processing_time_ms = processing_time

            # Complete the future
            if not queued_request.future.done():
                queued_request.future.set_result(response)

            # Update metrics
            self.metrics.completed_requests += 1
            self._update_processing_time_metrics(processing_time)

            logger.debug(f"Worker {worker_id} completed request {request_id} in {processing_time:.1f}ms")

        except asyncio.TimeoutError:
            self._handle_timeout(queued_request)
            logger.warning(f"Request {request_id} timed out after {queued_request.timeout}s")

        except Exception as e:
            await self._handle_error(queued_request, e, worker_id)
            logger.error(f"Worker {worker_id} failed to process request {request_id}: {e}")

        finally:
            # Remove from processing dict
            if queued_request.id in self.processing:
                del self.processing[queued_request.id]

            # Update queue size metric
            total_queued = sum(queue.qsize() for queue in self.queues.values())
            self.metrics.current_queue_size = total_queued

    async def _handle_error(self, queued_request: QueuedRequest, error: Exception, worker_id: str):
        """Handle request processing error"""
        request_id = queued_request.id[:8]

        if queued_request.retries_remaining > 0:
            # Retry the request
            queued_request.retries_remaining -= 1
            queued_request.created_at = time.time()  # Reset timing

            # Re-queue with same priority
            await self.queues[queued_request.priority].put(queued_request)

            self.metrics.retried_requests += 1
            logger.info(f"Retrying request {request_id}, {queued_request.retries_remaining} retries left")

        else:
            # No more retries, fail the request
            if not queued_request.future.done():
                queued_request.future.set_exception(error)

            self.metrics.failed_requests += 1
            logger.error(f"Request {request_id} failed after all retries: {error}")

    def _handle_timeout(self, queued_request: QueuedRequest):
        """Handle request timeout"""
        if not queued_request.future.done():
            timeout_error = asyncio.TimeoutError(
                f"Request timed out after {queued_request.timeout}s"
            )
            queued_request.future.set_exception(timeout_error)

        self.metrics.timed_out_requests += 1

    def _update_processing_time_metrics(self, processing_time: float):
        """Update average processing time metrics"""
        if self.metrics.completed_requests == 1:
            self.metrics.average_processing_time = processing_time
        else:
            # Calculate running average
            current_avg = self.metrics.average_processing_time
            new_avg = (current_avg * (self.metrics.completed_requests - 1) + processing_time) / self.metrics.completed_requests
            self.metrics.average_processing_time = new_avg

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        queue_sizes = {
            priority.value: queue.qsize()
            for priority, queue in self.queues.items()
        }

        processing_requests = [
            {
                "id": req.id[:8],
                "priority": req.priority.value,
                "agent_type": req.agent_type,
                "processing_time": time.time() - req.created_at
            }
            for req in self.processing.values()
        ]

        return {
            "is_running": self.is_running,
            "workers": len(self.workers),
            "max_concurrent": self.max_concurrent,
            "queue_sizes": queue_sizes,
            "total_queued": sum(queue_sizes.values()),
            "processing": processing_requests,
            "processing_count": len(processing_requests),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "completed_requests": self.metrics.completed_requests,
                "failed_requests": self.metrics.failed_requests,
                "timed_out_requests": self.metrics.timed_out_requests,
                "retried_requests": self.metrics.retried_requests,
                "current_queue_size": self.metrics.current_queue_size,
                "peak_queue_size": self.metrics.peak_queue_size,
                "average_processing_time": self.metrics.average_processing_time,
                "success_rate": self.metrics.completed_requests / max(1, self.metrics.total_requests),
                "retry_rate": self.metrics.retried_requests / max(1, self.metrics.total_requests),
                "timeout_rate": self.metrics.timed_out_requests / max(1, self.metrics.total_requests)
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check queue manager health"""
        status = await self.get_queue_status()

        # Determine health status
        health = "healthy"
        issues = []

        # Check if queue is backing up
        if status["total_queued"] > self.max_queue_size * 0.8:
            health = "degraded"
            issues.append("Queue is nearly full")

        # Check if too many workers are failing
        if status["metrics"]["success_rate"] < 0.8:
            health = "degraded"
            issues.append("High failure rate")

        # Check if processing times are too high
        if status["metrics"]["average_processing_time"] > 10000:  # 10 seconds
            health = "degraded"
            issues.append("High processing times")

        return {
            "status": health,
            "issues": issues,
            **status
        }

    async def clear_queues(self):
        """Clear all pending requests from queues"""
        cleared_count = 0

        for queue in self.queues.values():
            while not queue.empty():
                try:
                    queued_request = queue.get_nowait()
                    if not queued_request.future.done():
                        queued_request.future.set_exception(
                            asyncio.CancelledError("Request cancelled due to queue clear")
                        )
                    cleared_count += 1
                except asyncio.QueueEmpty:
                    break

        logger.info(f"Cleared {cleared_count} pending requests from queues")
        return cleared_count


# Global queue manager instance
request_queue = LLMRequestQueue()