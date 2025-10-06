"""
Streaming Handler for LLM Responses

This module provides utilities for handling streaming LLM responses,
including buffering, error recovery, and integration with voice pipelines.
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class StreamState(str, Enum):
    """State of streaming response."""
    IDLE = "idle"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StreamChunk:
    """A chunk of streamed content."""
    content: str
    chunk_index: int
    timestamp: float
    tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamMetrics:
    """Metrics for streaming response."""
    total_chunks: int = 0
    total_tokens: int = 0
    total_bytes: int = 0
    time_to_first_chunk_ms: float = 0.0
    total_duration_ms: float = 0.0
    average_chunk_time_ms: float = 0.0
    chunks_per_second: float = 0.0
    throughput_tokens_per_second: float = 0.0


class StreamingHandler:
    """
    Handles streaming LLM responses with buffering and error recovery.

    Features:
    - Chunk buffering and aggregation
    - Error detection and recovery
    - Performance metrics tracking
    - Sentence-boundary detection for voice synthesis
    - Backpressure handling
    """

    def __init__(
        self,
        buffer_size: int = 10,
        sentence_detection: bool = True,
        chunk_callback: Optional[Callable[[StreamChunk], None]] = None
    ):
        """
        Initialize streaming handler.

        Args:
            buffer_size: Number of chunks to buffer
            sentence_detection: Enable sentence boundary detection
            chunk_callback: Optional callback for each chunk
        """
        self.buffer_size = buffer_size
        self.sentence_detection = sentence_detection
        self.chunk_callback = chunk_callback

        self.state = StreamState.IDLE
        self.buffer: List[StreamChunk] = []
        self.metrics = StreamMetrics()

        self._start_time: Optional[float] = None
        self._first_chunk_time: Optional[float] = None
        self._chunk_times: List[float] = []

    async def stream_with_buffer(
        self,
        stream_generator: AsyncGenerator[str, None],
        aggregate_sentences: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        Stream content with buffering and optional sentence aggregation.

        Args:
            stream_generator: Async generator of string chunks
            aggregate_sentences: Aggregate chunks until sentence boundary

        Yields:
            Processed string chunks
        """
        self.state = StreamState.STREAMING
        self._start_time = time.time()
        chunk_index = 0
        buffer_content = ""

        try:
            async for chunk_content in stream_generator:
                chunk_time = time.time()

                # Track first chunk time
                if self._first_chunk_time is None:
                    self._first_chunk_time = chunk_time
                    self.metrics.time_to_first_chunk_ms = (
                        (chunk_time - self._start_time) * 1000
                    )

                # Create chunk object
                chunk = StreamChunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    timestamp=chunk_time,
                    tokens=len(chunk_content.split())  # Rough estimate
                )

                # Update metrics
                self.metrics.total_chunks += 1
                self.metrics.total_tokens += chunk.tokens
                self.metrics.total_bytes += len(chunk_content.encode('utf-8'))
                self._chunk_times.append(chunk_time)

                # Call chunk callback if provided
                if self.chunk_callback:
                    try:
                        self.chunk_callback(chunk)
                    except Exception as e:
                        logger.error(f"Chunk callback error: {e}")

                # Handle sentence aggregation
                if aggregate_sentences and self.sentence_detection:
                    buffer_content += chunk_content

                    # Check for sentence boundary
                    if self._has_sentence_boundary(buffer_content):
                        yield buffer_content
                        buffer_content = ""
                else:
                    yield chunk_content

                chunk_index += 1

            # Yield any remaining buffered content
            if buffer_content:
                yield buffer_content

            self.state = StreamState.COMPLETED
            self._finalize_metrics()

        except asyncio.CancelledError:
            logger.info("Streaming cancelled")
            self.state = StreamState.PAUSED
            raise

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.state = StreamState.ERROR
            raise

    def _has_sentence_boundary(self, text: str) -> bool:
        """
        Detect if text ends with a sentence boundary.

        Args:
            text: Text to check

        Returns:
            True if sentence boundary detected
        """
        # Check for sentence-ending punctuation
        sentence_enders = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        return any(text.endswith(ender) for ender in sentence_enders)

    def _finalize_metrics(self):
        """Finalize metrics after streaming completes."""
        if self._start_time:
            self.metrics.total_duration_ms = (time.time() - self._start_time) * 1000

        if self.metrics.total_chunks > 0:
            self.metrics.average_chunk_time_ms = (
                self.metrics.total_duration_ms / self.metrics.total_chunks
            )

        if self.metrics.total_duration_ms > 0:
            duration_seconds = self.metrics.total_duration_ms / 1000
            self.metrics.chunks_per_second = self.metrics.total_chunks / duration_seconds
            self.metrics.throughput_tokens_per_second = self.metrics.total_tokens / duration_seconds

    async def stream_with_retry(
        self,
        stream_generator: AsyncGenerator[str, None],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> AsyncGenerator[str, None]:
        """
        Stream with automatic retry on failure.

        Args:
            stream_generator: Async generator of string chunks
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Yields:
            String chunks
        """
        retries = 0

        while retries <= max_retries:
            try:
                async for chunk in self.stream_with_buffer(stream_generator):
                    yield chunk
                break  # Success, exit retry loop

            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Streaming failed after {max_retries} retries: {e}")
                    raise

                logger.warning(f"Streaming error, retry {retries}/{max_retries}: {e}")
                await asyncio.sleep(retry_delay * retries)  # Exponential backoff

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get streaming metrics.

        Returns:
            Dictionary with metrics
        """
        return {
            "state": self.state.value,
            "total_chunks": self.metrics.total_chunks,
            "total_tokens": self.metrics.total_tokens,
            "total_bytes": self.metrics.total_bytes,
            "time_to_first_chunk_ms": self.metrics.time_to_first_chunk_ms,
            "total_duration_ms": self.metrics.total_duration_ms,
            "average_chunk_time_ms": self.metrics.average_chunk_time_ms,
            "chunks_per_second": self.metrics.chunks_per_second,
            "throughput_tokens_per_second": self.metrics.throughput_tokens_per_second
        }

    def reset(self):
        """Reset handler state and metrics."""
        self.state = StreamState.IDLE
        self.buffer.clear()
        self.metrics = StreamMetrics()
        self._start_time = None
        self._first_chunk_time = None
        self._chunk_times.clear()


class VoiceStreamAdapter:
    """
    Adapts LLM streaming output for voice synthesis pipeline.

    Features:
    - Sentence-based chunking for natural speech
    - Pause insertion at punctuation
    - Buffer management for smooth audio
    """

    def __init__(
        self,
        min_chunk_length: int = 10,
        max_chunk_length: int = 200
    ):
        """
        Initialize voice stream adapter.

        Args:
            min_chunk_length: Minimum characters per chunk
            max_chunk_length: Maximum characters per chunk
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length

    async def adapt_for_voice(
        self,
        stream_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        """
        Adapt streaming text for voice synthesis.

        Args:
            stream_generator: Raw text stream

        Yields:
            Voice-optimized text chunks
        """
        buffer = ""

        async for chunk in stream_generator:
            buffer += chunk

            # Check if we have enough content for a voice chunk
            while len(buffer) >= self.min_chunk_length:
                # Find sentence boundary
                boundary_pos = self._find_voice_boundary(buffer)

                if boundary_pos > 0:
                    # Yield chunk up to boundary
                    voice_chunk = buffer[:boundary_pos].strip()
                    if voice_chunk:
                        yield voice_chunk

                    buffer = buffer[boundary_pos:].strip()
                else:
                    # No boundary found, but buffer is too long
                    if len(buffer) >= self.max_chunk_length:
                        # Force split at max length
                        yield buffer[:self.max_chunk_length].strip()
                        buffer = buffer[self.max_chunk_length:].strip()
                    else:
                        # Wait for more content
                        break

        # Yield any remaining content
        if buffer.strip():
            yield buffer.strip()

    def _find_voice_boundary(self, text: str) -> int:
        """
        Find optimal boundary for voice synthesis.

        Priorities:
        1. Sentence endings (. ! ?)
        2. Clause boundaries (, ; :)
        3. Word boundaries

        Args:
            text: Text to analyze

        Returns:
            Position of boundary, or -1 if none found
        """
        # Look for sentence endings
        for punct in ['. ', '! ', '? ']:
            pos = text.find(punct)
            if pos > self.min_chunk_length and pos < self.max_chunk_length:
                return pos + len(punct)

        # Look for clause boundaries
        for punct in [', ', '; ', ': ']:
            pos = text.find(punct)
            if pos > self.min_chunk_length and pos < self.max_chunk_length:
                return pos + len(punct)

        # Look for last word boundary within limit
        if len(text) > self.max_chunk_length:
            chunk = text[:self.max_chunk_length]
            last_space = chunk.rfind(' ')
            if last_space > self.min_chunk_length:
                return last_space + 1

        return -1


class StreamingMetricsCollector:
    """Collects and aggregates streaming metrics across multiple streams."""

    def __init__(self):
        self.stream_metrics: List[Dict[str, Any]] = []
        self.total_streams = 0

    def record_stream(self, metrics: Dict[str, Any]):
        """Record metrics from a completed stream."""
        self.stream_metrics.append(metrics)
        self.total_streams += 1

        # Keep only last 100 streams
        if len(self.stream_metrics) > 100:
            self.stream_metrics.pop(0)

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all streams."""
        if not self.stream_metrics:
            return {
                "total_streams": 0,
                "avg_time_to_first_chunk_ms": 0.0,
                "avg_total_duration_ms": 0.0,
                "avg_chunks_per_stream": 0.0,
                "avg_tokens_per_stream": 0.0,
                "avg_throughput_tokens_per_second": 0.0
            }

        total_ttfc = sum(m.get("time_to_first_chunk_ms", 0) for m in self.stream_metrics)
        total_duration = sum(m.get("total_duration_ms", 0) for m in self.stream_metrics)
        total_chunks = sum(m.get("total_chunks", 0) for m in self.stream_metrics)
        total_tokens = sum(m.get("total_tokens", 0) for m in self.stream_metrics)
        total_throughput = sum(m.get("throughput_tokens_per_second", 0) for m in self.stream_metrics)

        count = len(self.stream_metrics)

        return {
            "total_streams": self.total_streams,
            "avg_time_to_first_chunk_ms": total_ttfc / count,
            "avg_total_duration_ms": total_duration / count,
            "avg_chunks_per_stream": total_chunks / count,
            "avg_tokens_per_stream": total_tokens / count,
            "avg_throughput_tokens_per_second": total_throughput / count
        }
