"""
Audio Buffer Manager for Real-Time Voice Processing

Provides efficient circular buffer management for 20ms audio chunks
with latency monitoring and overflow/underflow handling.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, List, Tuple
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BufferMetrics:
    """Metrics for audio buffer performance"""
    buffer_fill_percentage: float = 0.0
    underruns: int = 0
    overruns: int = 0
    total_chunks_processed: int = 0
    average_latency_ms: float = 0.0
    current_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AudioBufferManager:
    """
    Circular buffer manager for real-time audio processing

    Features:
    - 20ms chunk normalization (320 samples @ 16kHz)
    - Circular buffer with configurable size
    - Zero-copy operations where possible
    - Latency monitoring (<200ms target)
    - Overflow/underflow detection
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        buffer_duration_ms: int = 500,
        channels: int = 1,
        dtype: str = "int16"
    ):
        """
        Initialize audio buffer manager

        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            chunk_duration_ms: Target chunk duration in ms (default: 20)
            buffer_duration_ms: Total buffer duration in ms (default: 500)
            channels: Number of audio channels (default: 1 for mono)
            dtype: Audio data type (default: int16)
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.buffer_duration_ms = buffer_duration_ms
        self.channels = channels
        self.dtype = np.dtype(dtype)

        # Calculate chunk size in samples
        self.chunk_size = int((chunk_duration_ms / 1000.0) * sample_rate)

        # Calculate max number of chunks in buffer
        self.max_chunks = int(buffer_duration_ms / chunk_duration_ms)

        # Circular buffer (deque for O(1) append/pop)
        self.buffer: deque = deque(maxlen=self.max_chunks)

        # Metrics
        self.metrics = BufferMetrics()
        self._latency_samples: List[float] = []
        self._max_latency_samples = 100

        # State
        self._lock = asyncio.Lock()
        self._write_index = 0
        self._read_index = 0

        logger.info(
            f"AudioBufferManager initialized: "
            f"{sample_rate}Hz, {chunk_duration_ms}ms chunks, "
            f"{self.chunk_size} samples/chunk, "
            f"max {self.max_chunks} chunks ({buffer_duration_ms}ms)"
        )

    async def write_chunk(
        self,
        audio_data: np.ndarray,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Write audio chunk to buffer

        Args:
            audio_data: Audio data as numpy array
            timestamp: Optional timestamp for latency tracking

        Returns:
            True if successful, False if overflow
        """
        try:
            async with self._lock:
                # Normalize chunk size
                normalized_chunk = self._normalize_chunk(audio_data)
                if normalized_chunk is None:
                    return False

                # Check for overflow
                if len(self.buffer) >= self.max_chunks:
                    self.metrics.overruns += 1
                    logger.warning(f"Buffer overflow! Dropping oldest chunk. Overruns: {self.metrics.overruns}")
                    # Buffer will auto-drop oldest due to maxlen

                # Add chunk with metadata
                chunk_metadata = {
                    "data": normalized_chunk,
                    "timestamp": timestamp or time.time(),
                    "write_index": self._write_index
                }

                self.buffer.append(chunk_metadata)
                self._write_index += 1
                self.metrics.total_chunks_processed += 1

                # Update metrics
                self._update_metrics()

                return True

        except Exception as e:
            logger.error(f"Error writing audio chunk: {e}")
            return False

    async def read_chunk(
        self,
        timeout: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Read audio chunk from buffer

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Audio chunk as numpy array, or None if buffer empty
        """
        try:
            start_time = time.time()

            while True:
                async with self._lock:
                    if len(self.buffer) > 0:
                        chunk_metadata = self.buffer.popleft()
                        self._read_index += 1

                        # Calculate latency
                        latency_ms = (time.time() - chunk_metadata["timestamp"]) * 1000
                        self._record_latency(latency_ms)

                        return chunk_metadata["data"]

                    # Buffer underrun
                    self.metrics.underruns += 1

                # Check timeout
                if timeout is not None and (time.time() - start_time) > timeout:
                    logger.debug("Read timeout, buffer empty")
                    return None

                # Wait a bit before retrying
                await asyncio.sleep(self.chunk_duration_ms / 2000.0)  # Half chunk duration

        except Exception as e:
            logger.error(f"Error reading audio chunk: {e}")
            return None

    def read_chunk_nowait(self) -> Optional[np.ndarray]:
        """
        Read audio chunk without waiting (synchronous)

        Returns:
            Audio chunk or None if buffer empty
        """
        try:
            if len(self.buffer) > 0:
                chunk_metadata = self.buffer.popleft()
                self._read_index += 1

                # Calculate latency
                latency_ms = (time.time() - chunk_metadata["timestamp"]) * 1000
                self._record_latency(latency_ms)

                return chunk_metadata["data"]

            self.metrics.underruns += 1
            return None

        except Exception as e:
            logger.error(f"Error reading audio chunk (nowait): {e}")
            return None

    def _normalize_chunk(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize audio chunk to target size

        Args:
            audio_data: Input audio data

        Returns:
            Normalized chunk or None if error
        """
        try:
            # Ensure numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.frombuffer(audio_data, dtype=self.dtype)

            # Ensure correct dtype
            if audio_data.dtype != self.dtype:
                audio_data = audio_data.astype(self.dtype)

            # Handle different chunk sizes
            if len(audio_data) == self.chunk_size:
                # Perfect size, return as-is (zero-copy)
                return audio_data

            elif len(audio_data) < self.chunk_size:
                # Pad with zeros
                padding = self.chunk_size - len(audio_data)
                return np.pad(audio_data, (0, padding), mode='constant')

            else:
                # Truncate or split
                logger.warning(f"Audio chunk too large: {len(audio_data)} samples, expected {self.chunk_size}")
                return audio_data[:self.chunk_size]

        except Exception as e:
            logger.error(f"Error normalizing audio chunk: {e}")
            return None

    def _update_metrics(self):
        """Update buffer metrics"""
        self.metrics.buffer_fill_percentage = (len(self.buffer) / self.max_chunks) * 100.0
        self.metrics.timestamp = time.time()

        # Update current latency estimate
        if len(self.buffer) > 0:
            # Latency = number of chunks * chunk duration
            self.metrics.current_latency_ms = len(self.buffer) * self.chunk_duration_ms

    def _record_latency(self, latency_ms: float):
        """Record latency sample for averaging"""
        self._latency_samples.append(latency_ms)

        # Keep only recent samples
        if len(self._latency_samples) > self._max_latency_samples:
            self._latency_samples.pop(0)

        # Update average
        if self._latency_samples:
            self.metrics.average_latency_ms = np.mean(self._latency_samples)

    def get_metrics(self) -> BufferMetrics:
        """Get current buffer metrics"""
        self._update_metrics()
        return self.metrics

    def get_buffer_level(self) -> Tuple[int, int, float]:
        """
        Get buffer fill level

        Returns:
            Tuple of (current_chunks, max_chunks, fill_percentage)
        """
        current = len(self.buffer)
        return (current, self.max_chunks, (current / self.max_chunks) * 100.0)

    def is_healthy(self, min_fill: float = 10.0, max_fill: float = 90.0) -> bool:
        """
        Check if buffer is in healthy state

        Args:
            min_fill: Minimum healthy fill percentage (default: 10%)
            max_fill: Maximum healthy fill percentage (default: 90%)

        Returns:
            True if buffer is healthy
        """
        fill_pct = self.metrics.buffer_fill_percentage
        return min_fill <= fill_pct <= max_fill

    async def clear(self):
        """Clear all data from buffer"""
        async with self._lock:
            self.buffer.clear()
            logger.info("Audio buffer cleared")

    def get_latency_ms(self) -> float:
        """Get current buffer latency in milliseconds"""
        return len(self.buffer) * self.chunk_duration_ms

    def __len__(self) -> int:
        """Return number of chunks in buffer"""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation"""
        current, max_chunks, fill_pct = self.get_buffer_level()
        return (
            f"AudioBufferManager("
            f"level={current}/{max_chunks} ({fill_pct:.1f}%), "
            f"latency={self.get_latency_ms():.1f}ms, "
            f"underruns={self.metrics.underruns}, "
            f"overruns={self.metrics.overruns})"
        )
