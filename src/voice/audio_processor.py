"""
Unified Audio Processor for Real-Time VoIP

Integrates all audio processing components into a single pipeline:
- Audio buffer management
- Jitter buffer
- Echo cancellation
- Noise reduction
- Packet loss concealment
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from .audio_buffer import AudioBufferManager, BufferMetrics
from .jitter_buffer import AdaptiveJitterBuffer, JitterMetrics
from .echo_cancellation import EchoCancellationProcessor
from .noise_reduction import NoiseReductionProcessor
from .packet_loss_concealment import PacketLossConcealment

logger = logging.getLogger(__name__)


@dataclass
class AudioProcessorMetrics:
    """Combined metrics for audio processor"""
    buffer_metrics: BufferMetrics
    jitter_metrics: JitterMetrics
    total_latency_ms: float = 0.0
    processing_time_ms: float = 0.0
    echo_cancelled_frames: int = 0
    noise_reduced_frames: int = 0
    timestamp: float = field(default_factory=time.time)


class AudioProcessor:
    """
    Unified audio processor for real-time VoIP

    Provides complete audio processing pipeline with configurable components:
    - Input: Raw audio from network → Jitter buffer → Echo cancel → Noise reduce → Output buffer
    - Output: Raw audio from TTS → Output buffer → Network

    Handles:
    - Network jitter and packet reordering
    - Echo cancellation (AEC)
    - Noise reduction
    - Buffer management
    - Latency optimization (<200ms target)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        enable_jitter_buffer: bool = True,
        enable_echo_cancellation: bool = True,
        enable_noise_reduction: bool = True,
        jitter_target_ms: int = 80,
        noise_reduction_strength: float = 0.8,
        echo_filter_length: int = 1024
    ):
        """
        Initialize audio processor

        Args:
            sample_rate: Audio sample rate (default: 16000)
            chunk_duration_ms: Audio chunk duration (default: 20ms)
            enable_jitter_buffer: Enable jitter buffering (default: True)
            enable_echo_cancellation: Enable AEC (default: True)
            enable_noise_reduction: Enable noise reduction (default: True)
            jitter_target_ms: Target jitter buffer delay (default: 80ms)
            noise_reduction_strength: Noise reduction strength (default: 0.8)
            echo_filter_length: Echo cancellation filter length (default: 1024)
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int((chunk_duration_ms / 1000.0) * sample_rate)

        # Configuration
        self.enable_jitter_buffer = enable_jitter_buffer
        self.enable_echo_cancellation = enable_echo_cancellation
        self.enable_noise_reduction = enable_noise_reduction

        # Initialize components
        self.input_buffer = AudioBufferManager(
            sample_rate=sample_rate,
            chunk_duration_ms=chunk_duration_ms,
            buffer_duration_ms=500
        )

        self.output_buffer = AudioBufferManager(
            sample_rate=sample_rate,
            chunk_duration_ms=chunk_duration_ms,
            buffer_duration_ms=300  # Smaller buffer for output
        )

        self.jitter_buffer: Optional[AdaptiveJitterBuffer] = None
        if enable_jitter_buffer:
            self.jitter_buffer = AdaptiveJitterBuffer(
                target_delay_ms=jitter_target_ms,
                sample_rate=sample_rate,
                chunk_duration_ms=chunk_duration_ms
            )

        self.echo_canceller: Optional[EchoCancellationProcessor] = None
        if enable_echo_cancellation:
            self.echo_canceller = EchoCancellationProcessor(
                sample_rate=sample_rate,
                frame_size=self.chunk_size,
                filter_length=echo_filter_length
            )

        self.noise_reducer: Optional[NoiseReductionProcessor] = None
        if enable_noise_reduction:
            self.noise_reducer = NoiseReductionProcessor(
                sample_rate=sample_rate,
                frame_size=self.chunk_size,
                reduction_strength=noise_reduction_strength
            )

        # State
        self.is_initialized = True
        self.far_end_audio: Optional[np.ndarray] = None  # For echo cancellation reference

        # Metrics
        self.echo_cancelled_frames = 0
        self.noise_reduced_frames = 0
        self._last_processing_time_ms = 0.0

        logger.info(
            f"AudioProcessor initialized: "
            f"{sample_rate}Hz, {chunk_duration_ms}ms chunks, "
            f"jitter_buffer={'ON' if enable_jitter_buffer else 'OFF'}, "
            f"echo_cancel={'ON' if enable_echo_cancellation else 'OFF'}, "
            f"noise_reduction={'ON' if enable_noise_reduction else 'OFF'}"
        )

    async def process_input_audio(
        self,
        audio_data: np.ndarray,
        sequence_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        is_speech: bool = False
    ) -> Optional[np.ndarray]:
        """
        Process incoming audio (from network/microphone)

        Pipeline: Jitter Buffer → Echo Cancel → Noise Reduce → Input Buffer

        Args:
            audio_data: Raw audio data
            sequence_number: Packet sequence number (for jitter buffer)
            timestamp: Packet timestamp
            is_speech: Whether audio contains speech

        Returns:
            Processed audio or None if not ready
        """
        try:
            start_time = time.time()

            # Step 1: Jitter buffer (if enabled and sequence number provided)
            if self.enable_jitter_buffer and self.jitter_buffer and sequence_number is not None:
                # Add to jitter buffer
                await self.jitter_buffer.add_packet(audio_data, sequence_number, timestamp)

                # Get packet from jitter buffer
                audio_data = await self.jitter_buffer.get_packet(timeout=0.05)
                if audio_data is None:
                    return None  # Not enough data yet

            # Step 2: Echo cancellation (if enabled and reference available)
            if self.enable_echo_cancellation and self.echo_canceller and self.far_end_audio is not None:
                audio_data = self.echo_canceller.process(self.far_end_audio, audio_data)
                self.echo_cancelled_frames += 1

            # Step 3: Noise reduction (if enabled)
            if self.enable_noise_reduction and self.noise_reducer:
                audio_data = self.noise_reducer.process(audio_data, is_speech)
                self.noise_reduced_frames += 1

            # Step 4: Add to input buffer
            success = await self.input_buffer.write_chunk(audio_data, timestamp)
            if not success:
                logger.warning("Failed to write to input buffer")

            # Track processing time
            self._last_processing_time_ms = (time.time() - start_time) * 1000

            return audio_data

        except Exception as e:
            logger.error(f"Error processing input audio: {e}")
            return None

    async def process_output_audio(
        self,
        audio_data: np.ndarray,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Process outgoing audio (to network/speaker)

        Pipeline: TTS → Output Buffer → Network

        Also stores as far-end reference for echo cancellation

        Args:
            audio_data: Audio data from TTS
            timestamp: Optional timestamp

        Returns:
            True if successful
        """
        try:
            # Store as far-end reference for echo cancellation
            self.far_end_audio = audio_data.copy()

            # Add to output buffer
            success = await self.output_buffer.write_chunk(audio_data, timestamp)

            return success

        except Exception as e:
            logger.error(f"Error processing output audio: {e}")
            return False

    async def get_input_audio(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get processed input audio

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Processed audio chunk or None
        """
        try:
            return await self.input_buffer.read_chunk(timeout)
        except Exception as e:
            logger.error(f"Error getting input audio: {e}")
            return None

    async def get_output_audio(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get output audio for transmission

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Output audio chunk or None
        """
        try:
            return await self.output_buffer.read_chunk(timeout)
        except Exception as e:
            logger.error(f"Error getting output audio: {e}")
            return None

    def get_metrics(self) -> AudioProcessorMetrics:
        """Get combined processor metrics"""
        try:
            buffer_metrics = self.input_buffer.get_metrics()
            jitter_metrics = self.jitter_buffer.get_metrics() if self.jitter_buffer else JitterMetrics()

            # Calculate total latency
            total_latency = buffer_metrics.current_latency_ms
            if self.jitter_buffer:
                total_latency += jitter_metrics.buffer_delay_ms

            return AudioProcessorMetrics(
                buffer_metrics=buffer_metrics,
                jitter_metrics=jitter_metrics,
                total_latency_ms=total_latency,
                processing_time_ms=self._last_processing_time_ms,
                echo_cancelled_frames=self.echo_cancelled_frames,
                noise_reduced_frames=self.noise_reduced_frames
            )

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return AudioProcessorMetrics(
                buffer_metrics=BufferMetrics(),
                jitter_metrics=JitterMetrics()
            )

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of audio processor"""
        try:
            metrics = self.get_metrics()

            return {
                "status": "healthy" if metrics.total_latency_ms < 200 else "degraded",
                "total_latency_ms": metrics.total_latency_ms,
                "processing_time_ms": metrics.processing_time_ms,
                "input_buffer_fill_pct": metrics.buffer_metrics.buffer_fill_percentage,
                "jitter_ms": metrics.jitter_metrics.jitter_ms if self.jitter_buffer else 0.0,
                "packet_loss_rate": metrics.jitter_metrics.packet_loss_rate if self.jitter_buffer else 0.0,
                "echo_cancelled_frames": self.echo_cancelled_frames,
                "noise_reduced_frames": self.noise_reduced_frames,
                "components": {
                    "jitter_buffer": "enabled" if self.enable_jitter_buffer else "disabled",
                    "echo_cancellation": "enabled" if self.enable_echo_cancellation else "disabled",
                    "noise_reduction": "enabled" if self.enable_noise_reduction else "disabled"
                }
            }

        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {"status": "error", "error": str(e)}

    async def reset(self):
        """Reset all processor state"""
        try:
            await self.input_buffer.clear()
            await self.output_buffer.clear()

            if self.jitter_buffer:
                await self.jitter_buffer.reset()

            if self.echo_canceller:
                self.echo_canceller.reset()

            if self.noise_reducer:
                self.noise_reducer.reset()

            self.far_end_audio = None
            self.echo_cancelled_frames = 0
            self.noise_reduced_frames = 0

            logger.info("Audio processor reset")

        except Exception as e:
            logger.error(f"Error resetting audio processor: {e}")

    def __repr__(self) -> str:
        """String representation"""
        metrics = self.get_metrics()
        return (
            f"AudioProcessor("
            f"latency={metrics.total_latency_ms:.1f}ms, "
            f"jitter={metrics.jitter_metrics.jitter_ms:.1f}ms, "
            f"echo_cancel={self.echo_cancelled_frames}, "
            f"noise_reduce={self.noise_reduced_frames})"
        )
