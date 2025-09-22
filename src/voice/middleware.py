import logging
import time
import asyncio
from typing import Optional, Callable, Dict, Any
from contextlib import asynccontextmanager

from .models import AudioChunk, VoiceMetrics, VoiceCallState

logger = logging.getLogger(__name__)


class VoiceProcessingMiddleware:
    """
    Middleware for voice processing pipeline that handles audio buffering,
    quality monitoring, and performance optimization.
    """

    def __init__(self, buffer_size: int = 1024, max_latency_ms: float = 200):
        self.buffer_size = buffer_size
        self.max_latency_ms = max_latency_ms
        self.audio_buffers: Dict[str, list] = {}
        self.processing_metrics: Dict[str, VoiceMetrics] = {}

    async def process_audio_chunk(
        self,
        call_id: str,
        audio_chunk: AudioChunk,
        processor: Callable
    ) -> Optional[Any]:
        """Process audio chunk with buffering and metrics collection"""
        try:
            start_time = time.time()

            # Add to buffer
            if call_id not in self.audio_buffers:
                self.audio_buffers[call_id] = []

            self.audio_buffers[call_id].append(audio_chunk)

            # Check if buffer is ready for processing
            if self._should_process_buffer(call_id, audio_chunk):
                # Get buffered chunks
                buffered_chunks = self.audio_buffers[call_id]
                self.audio_buffers[call_id] = []

                # Combine chunks if needed
                combined_chunk = self._combine_audio_chunks(buffered_chunks)

                # Process with the provided processor
                result = await processor(combined_chunk)

                # Update metrics
                processing_time = (time.time() - start_time) * 1000
                self._update_metrics(call_id, processing_time, len(buffered_chunks))

                return result

            return None

        except Exception as e:
            logger.error(f"Audio processing middleware error for call {call_id}: {e}")
            return None

    def _should_process_buffer(self, call_id: str, latest_chunk: AudioChunk) -> bool:
        """Determine if buffer should be processed"""
        buffer = self.audio_buffers.get(call_id, [])

        # Process if buffer is full
        if len(buffer) >= self.buffer_size:
            return True

        # Process if silence detected (end of speech)
        if latest_chunk.silence_duration > 1.0:
            return len(buffer) > 0

        # Process if buffer has been waiting too long
        if buffer:
            oldest_chunk = buffer[0]
            wait_time = (time.time() - oldest_chunk.timestamp) * 1000
            if wait_time > self.max_latency_ms:
                return True

        return False

    def _combine_audio_chunks(self, chunks: list[AudioChunk]) -> AudioChunk:
        """Combine multiple audio chunks into one"""
        if len(chunks) == 1:
            return chunks[0]

        # Combine audio data
        combined_data = b''.join(chunk.data for chunk in chunks)

        # Use properties from first chunk
        first_chunk = chunks[0]

        return AudioChunk(
            data=combined_data,
            timestamp=first_chunk.timestamp,
            chunk_id=first_chunk.chunk_id,
            sample_rate=first_chunk.sample_rate,
            channels=first_chunk.channels,
            is_speech=any(chunk.is_speech for chunk in chunks),
            silence_duration=chunks[-1].silence_duration
        )

    def _update_metrics(self, call_id: str, processing_time: float, chunks_processed: int):
        """Update processing metrics"""
        if call_id not in self.processing_metrics:
            self.processing_metrics[call_id] = VoiceMetrics(call_id=call_id)

        metrics = self.processing_metrics[call_id]
        metrics.total_processing_latency_ms += processing_time
        metrics.timestamp = time.time()

        logger.debug(f"Call {call_id}: Processed {chunks_processed} chunks in {processing_time:.1f}ms")

    def get_metrics(self, call_id: str) -> Optional[VoiceMetrics]:
        """Get processing metrics for a call"""
        return self.processing_metrics.get(call_id)

    def cleanup_call(self, call_id: str):
        """Clean up resources for a completed call"""
        if call_id in self.audio_buffers:
            del self.audio_buffers[call_id]
        if call_id in self.processing_metrics:
            del self.processing_metrics[call_id]


class AudioQualityMonitor:
    """
    Monitor audio quality metrics and detect issues
    """

    def __init__(self):
        self.quality_thresholds = {
            "min_volume": 0.01,
            "max_volume": 0.95,
            "min_frequency": 80,  # Hz
            "max_frequency": 8000,  # Hz
            "max_distortion": 0.05
        }

    async def analyze_audio_quality(self, audio_chunk: AudioChunk) -> Dict[str, Any]:
        """Analyze audio quality and return metrics"""
        try:
            import numpy as np

            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_chunk.data, dtype=np.int16).astype(np.float32) / 32768.0

            if len(audio_array) == 0:
                return {"quality_score": 0.0, "issues": ["empty_audio"]}

            # Basic quality metrics
            volume_level = np.abs(audio_array).mean()
            peak_level = np.abs(audio_array).max()

            # Detect clipping
            clipping_ratio = np.sum(np.abs(audio_array) > 0.95) / len(audio_array)

            # Simple noise estimation (high frequency content)
            if len(audio_array) > 1024:
                fft = np.fft.fft(audio_array[:1024])
                high_freq_energy = np.sum(np.abs(fft[512:])) / np.sum(np.abs(fft))
            else:
                high_freq_energy = 0.0

            # Calculate quality score
            quality_score = 1.0
            issues = []

            # Volume checks
            if volume_level < self.quality_thresholds["min_volume"]:
                quality_score *= 0.7
                issues.append("low_volume")
            elif peak_level > self.quality_thresholds["max_volume"]:
                quality_score *= 0.8
                issues.append("high_volume")

            # Clipping check
            if clipping_ratio > 0.01:
                quality_score *= (1.0 - clipping_ratio)
                issues.append("clipping")

            # Noise check
            if high_freq_energy > 0.3:
                quality_score *= 0.9
                issues.append("high_noise")

            return {
                "quality_score": quality_score,
                "volume_level": volume_level,
                "peak_level": peak_level,
                "clipping_ratio": clipping_ratio,
                "noise_level": high_freq_energy,
                "issues": issues
            }

        except Exception as e:
            logger.error(f"Audio quality analysis error: {e}")
            return {"quality_score": 0.5, "issues": ["analysis_error"]}


class LatencyOptimizer:
    """
    Optimize pipeline for low latency processing
    """

    def __init__(self):
        self.target_latency_ms = 200
        self.adaptive_buffer_size = 512
        self.performance_history: Dict[str, list] = {}

    async def optimize_for_call(self, call_id: str, current_latency: float) -> Dict[str, Any]:
        """Optimize processing parameters based on current performance"""
        if call_id not in self.performance_history:
            self.performance_history[call_id] = []

        self.performance_history[call_id].append(current_latency)

        # Keep only recent history
        if len(self.performance_history[call_id]) > 10:
            self.performance_history[call_id] = self.performance_history[call_id][-10:]

        avg_latency = sum(self.performance_history[call_id]) / len(self.performance_history[call_id])

        recommendations = {}

        # Adjust buffer size based on latency
        if avg_latency > self.target_latency_ms * 1.2:
            # Latency too high, reduce buffer size
            recommendations["buffer_size"] = max(256, self.adaptive_buffer_size - 128)
            recommendations["processing_priority"] = "high"
        elif avg_latency < self.target_latency_ms * 0.8:
            # Good latency, can increase buffer size for quality
            recommendations["buffer_size"] = min(1024, self.adaptive_buffer_size + 128)
            recommendations["processing_priority"] = "normal"

        # Recommend model adjustments
        if avg_latency > self.target_latency_ms * 1.5:
            recommendations["stt_model"] = "whisper-base"  # Faster model
            recommendations["tts_quality"] = "fast"
        else:
            recommendations["stt_model"] = "whisper-small"  # Better quality
            recommendations["tts_quality"] = "normal"

        return {
            "current_latency": current_latency,
            "average_latency": avg_latency,
            "target_latency": self.target_latency_ms,
            "recommendations": recommendations
        }

    def cleanup_call(self, call_id: str):
        """Clean up performance history for completed call"""
        if call_id in self.performance_history:
            del self.performance_history[call_id]


class VoiceMiddlewareStack:
    """
    Combined middleware stack for voice processing
    """

    def __init__(self):
        self.processing_middleware = VoiceProcessingMiddleware()
        self.quality_monitor = AudioQualityMonitor()
        self.latency_optimizer = LatencyOptimizer()

    @asynccontextmanager
    async def process_call(self, call_id: str):
        """Context manager for call processing with all middleware"""
        try:
            logger.info(f"Starting voice middleware for call {call_id}")
            yield self
        finally:
            # Cleanup all middleware for this call
            self.processing_middleware.cleanup_call(call_id)
            self.latency_optimizer.cleanup_call(call_id)
            logger.info(f"Voice middleware cleanup complete for call {call_id}")

    async def process_audio(
        self,
        call_id: str,
        audio_chunk: AudioChunk,
        processor: Callable
    ) -> Optional[Any]:
        """Process audio through the middleware stack"""
        # Quality monitoring
        quality_metrics = await self.quality_monitor.analyze_audio_quality(audio_chunk)

        # Skip processing if quality is too poor
        if quality_metrics["quality_score"] < 0.3:
            logger.warning(f"Call {call_id}: Poor audio quality detected, skipping processing")
            return None

        # Process through buffering middleware
        result = await self.processing_middleware.process_audio_chunk(
            call_id, audio_chunk, processor
        )

        # Latency optimization
        if result is not None:
            processing_metrics = self.processing_middleware.get_metrics(call_id)
            if processing_metrics:
                optimization_suggestions = await self.latency_optimizer.optimize_for_call(
                    call_id, processing_metrics.total_processing_latency_ms
                )
                logger.debug(f"Call {call_id}: Latency optimization suggestions: {optimization_suggestions}")

        return result

    def get_call_metrics(self, call_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a call"""
        processing_metrics = self.processing_middleware.get_metrics(call_id)

        return {
            "processing_metrics": processing_metrics.dict() if processing_metrics else None,
            "call_id": call_id,
            "timestamp": time.time()
        }