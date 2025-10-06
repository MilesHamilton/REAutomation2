"""
Echo Cancellation for VoIP Audio

Provides acoustic echo cancellation (AEC) to remove echo from
audio calls, using either speexdsp or adaptive filter fallback.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import speexdsp (optional but recommended)
try:
    from speexdsp import EchoCanceller as SpeexEchoCanceller
    SPEEXDSP_AVAILABLE = True
    logger.info("speexdsp-python is available for echo cancellation")
except ImportError:
    SPEEXDSP_AVAILABLE = False
    SpeexEchoCanceller = None
    logger.warning("speexdsp-python not available, using fallback AEC")


class EchoCancellationProcessor:
    """
    Echo Cancellation Processor

    Cancels acoustic echo using adaptive filtering to remove
    far-end signal (TTS output) from near-end signal (microphone input).

    Supports:
    - speexdsp (Speex AEC) - Recommended, production quality
    - Adaptive LMS filter - Fallback when speexdsp not available
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,  # 20ms @ 16kHz
        filter_length: int = 1024,
        use_speex: bool = True
    ):
        """
        Initialize echo cancellation processor

        Args:
            sample_rate: Audio sample rate (default: 16000)
            frame_size: Frame size in samples (default: 320 = 20ms)
            filter_length: Adaptive filter length (default: 1024)
            use_speex: Use speexdsp if available (default: True)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.use_speex = use_speex and SPEEXDSP_AVAILABLE

        self.aec_engine: Optional[object] = None
        self.is_initialized = False

        # Fallback adaptive filter state
        self.filter_weights: Optional[np.ndarray] = None
        self.far_end_buffer: Optional[np.ndarray] = None

        self._initialize_aec()

    def _initialize_aec(self):
        """Initialize AEC engine"""
        try:
            if self.use_speex and SPEEXDSP_AVAILABLE:
                # Initialize Speex AEC
                self.aec_engine = SpeexEchoCanceller(
                    frame_size=self.frame_size,
                    filter_length=self.filter_length
                )
                self.is_initialized = True
                logger.info(f"Speex AEC initialized: frame_size={self.frame_size}, filter_length={self.filter_length}")

            else:
                # Initialize fallback adaptive filter (NLMS)
                self.filter_weights = np.zeros(self.filter_length, dtype=np.float32)
                self.far_end_buffer = np.zeros(self.filter_length, dtype=np.float32)
                self.is_initialized = True
                logger.info(f"Fallback NLMS AEC initialized: filter_length={self.filter_length}")

        except Exception as e:
            logger.error(f"Failed to initialize AEC: {e}")
            self.is_initialized = False

    def process(
        self,
        far_end_audio: np.ndarray,
        near_end_audio: np.ndarray
    ) -> np.ndarray:
        """
        Process audio to cancel echo

        Args:
            far_end_audio: Far-end signal (TTS output/reference)
            near_end_audio: Near-end signal (microphone input)

        Returns:
            Echo-cancelled audio
        """
        try:
            if not self.is_initialized:
                logger.warning("AEC not initialized, returning near-end audio unchanged")
                return near_end_audio

            # Ensure correct size
            far_end = self._normalize_frame(far_end_audio)
            near_end = self._normalize_frame(near_end_audio)

            if far_end is None or near_end is None:
                return near_end_audio

            # Process with appropriate engine
            if self.use_speex and self.aec_engine is not None:
                return self._process_speex(far_end, near_end)
            else:
                return self._process_adaptive_filter(far_end, near_end)

        except Exception as e:
            logger.error(f"Error in echo cancellation: {e}")
            return near_end_audio

    def _process_speex(self, far_end: np.ndarray, near_end: np.ndarray) -> np.ndarray:
        """
        Process using Speex AEC

        Args:
            far_end: Far-end audio (reference)
            near_end: Near-end audio (input with echo)

        Returns:
            Echo-cancelled audio
        """
        try:
            # Speex expects int16
            if far_end.dtype != np.int16:
                far_end = (far_end * 32767).astype(np.int16)
            if near_end.dtype != np.int16:
                near_end = (near_end * 32767).astype(np.int16)

            # Process with Speex AEC
            output = self.aec_engine.process(far_end, near_end)

            return output

        except Exception as e:
            logger.error(f"Speex AEC processing error: {e}")
            return near_end

    def _process_adaptive_filter(self, far_end: np.ndarray, near_end: np.ndarray) -> np.ndarray:
        """
        Process using NLMS adaptive filter fallback

        Args:
            far_end: Far-end audio (reference)
            near_end: Near-end audio (input with echo)

        Returns:
            Echo-cancelled audio
        """
        try:
            # Ensure float32
            if far_end.dtype != np.float32:
                far_end = far_end.astype(np.float32) / 32768.0 if far_end.dtype == np.int16 else far_end.astype(np.float32)
            if near_end.dtype != np.float32:
                near_end = near_end.astype(np.float32) / 32768.0 if near_end.dtype == np.int16 else near_end.astype(np.float32)

            # NLMS parameters
            mu = 0.1  # Step size
            epsilon = 1e-6  # Regularization

            output = near_end.copy()

            # Process sample by sample
            for i in range(len(far_end)):
                # Update far-end buffer
                self.far_end_buffer = np.roll(self.far_end_buffer, 1)
                self.far_end_buffer[0] = far_end[i]

                # Estimate echo
                estimated_echo = np.dot(self.filter_weights, self.far_end_buffer)

                # Error signal (desired output)
                error = near_end[i] - estimated_echo
                output[i] = error

                # Update filter weights (NLMS)
                norm_factor = np.dot(self.far_end_buffer, self.far_end_buffer) + epsilon
                self.filter_weights += (mu / norm_factor) * error * self.far_end_buffer

            # Convert back to original dtype if needed
            if near_end_audio.dtype == np.int16:
                output = (output * 32767).astype(np.int16)

            return output

        except Exception as e:
            logger.error(f"Adaptive filter processing error: {e}")
            return near_end

    def _normalize_frame(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize audio frame to expected size

        Args:
            audio_data: Input audio

        Returns:
            Normalized audio or None if error
        """
        try:
            # Convert to numpy array if needed
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)

            # Check size
            if len(audio_data) == self.frame_size:
                return audio_data

            elif len(audio_data) < self.frame_size:
                # Pad with zeros
                padding = self.frame_size - len(audio_data)
                return np.pad(audio_data, (0, padding), mode='constant')

            else:
                # Truncate
                logger.warning(f"Audio frame too large: {len(audio_data)} samples, expected {self.frame_size}")
                return audio_data[:self.frame_size]

        except Exception as e:
            logger.error(f"Error normalizing frame: {e}")
            return None

    def reset(self):
        """Reset AEC state"""
        try:
            if self.use_speex and self.aec_engine is not None:
                # Speex reset (if available)
                self.aec_engine.reset()

            else:
                # Reset adaptive filter
                if self.filter_weights is not None:
                    self.filter_weights.fill(0)
                if self.far_end_buffer is not None:
                    self.far_end_buffer.fill(0)

            logger.info("AEC state reset")

        except Exception as e:
            logger.error(f"Error resetting AEC: {e}")

    def get_stats(self) -> dict:
        """Get AEC statistics"""
        return {
            "engine": "speex" if self.use_speex else "nlms",
            "initialized": self.is_initialized,
            "frame_size": self.frame_size,
            "filter_length": self.filter_length,
            "sample_rate": self.sample_rate
        }

    def __repr__(self) -> str:
        """String representation"""
        engine = "Speex" if self.use_speex else "NLMS"
        return (
            f"EchoCancellationProcessor("
            f"engine={engine}, "
            f"frame_size={self.frame_size}, "
            f"filter_length={self.filter_length}, "
            f"initialized={self.is_initialized})"
        )
