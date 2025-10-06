"""
Noise Reduction for VoIP Audio

Provides real-time noise reduction to improve speech clarity
using spectral subtraction or the noisereduce library.
"""

import logging
import numpy as np
from typing import Optional
from collections import deque

logger = logging.getLogger(__name__)

# Try to import noisereduce (optional but recommended)
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
    logger.info("noisereduce library is available")
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    nr = None
    logger.warning("noisereduce not available, using fallback noise reduction")


class NoiseReductionProcessor:
    """
    Noise Reduction Processor

    Reduces background noise from audio using either:
    - noisereduce library (recommended) - Statistical noise reduction
    - Spectral subtraction fallback - Simple frequency-domain filtering
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,  # 20ms @ 16kHz
        reduction_strength: float = 0.8,
        use_noisereduce: bool = True,
        stationary: bool = True
    ):
        """
        Initialize noise reduction processor

        Args:
            sample_rate: Audio sample rate (default: 16000)
            frame_size: Frame size in samples (default: 320)
            reduction_strength: Strength of noise reduction 0.0-1.0 (default: 0.8)
            use_noisereduce: Use noisereduce library if available (default: True)
            stationary: Assume stationary noise (default: True)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.reduction_strength = np.clip(reduction_strength, 0.0, 1.0)
        self.use_noisereduce = use_noisereduce and NOISEREDUCE_AVAILABLE
        self.stationary = stationary

        # Noise profile
        self.noise_profile: Optional[np.ndarray] = None
        self.noise_learning_frames = 10  # Number of initial frames to learn noise
        self.frames_processed = 0

        # Buffer for noisereduce (needs a bit of context)
        self.audio_buffer: deque = deque(maxlen=10)  # Keep last 10 frames

        # Spectral subtraction state
        self.fft_size = self._next_power_of_2(frame_size)
        self.noise_spectrum: Optional[np.ndarray] = None

        self.is_initialized = True

        logger.info(
            f"NoiseReductionProcessor initialized: "
            f"{'noisereduce' if self.use_noisereduce else 'spectral subtraction'}, "
            f"strength={reduction_strength:.2f}"
        )

    def process(self, audio_data: np.ndarray, is_speech: bool = False) -> np.ndarray:
        """
        Process audio to reduce noise

        Args:
            audio_data: Input audio as numpy array
            is_speech: Whether frame contains speech (optional hint)

        Returns:
            Noise-reduced audio
        """
        try:
            # Normalize frame
            audio = self._normalize_frame(audio_data)
            if audio is None:
                return audio_data

            # Add to buffer
            self.audio_buffer.append(audio.copy())

            # Learn noise profile from initial frames (if no speech detected)
            if self.frames_processed < self.noise_learning_frames and not is_speech:
                self._update_noise_profile(audio)
                self.frames_processed += 1
                return audio  # Return unchanged during learning phase

            self.frames_processed += 1

            # Apply noise reduction
            if self.use_noisereduce and NOISEREDUCE_AVAILABLE:
                return self._reduce_noise_noisereduce(audio)
            else:
                return self._reduce_noise_spectral_subtraction(audio)

        except Exception as e:
            logger.error(f"Error in noise reduction: {e}")
            return audio_data

    def _reduce_noise_noisereduce(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce noise using noisereduce library

        Args:
            audio: Input audio

        Returns:
            Noise-reduced audio
        """
        try:
            # Ensure float
            if audio.dtype != np.float32:
                audio_float = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
            else:
                audio_float = audio

            # Apply noise reduction
            reduced = nr.reduce_noise(
                y=audio_float,
                sr=self.sample_rate,
                stationary=self.stationary,
                prop_decrease=self.reduction_strength
            )

            # Convert back to original dtype
            if audio_data.dtype == np.int16:
                reduced = (reduced * 32767).astype(np.int16)
            else:
                reduced = reduced.astype(audio.dtype)

            return reduced

        except Exception as e:
            logger.error(f"noisereduce processing error: {e}")
            return audio

    def _reduce_noise_spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce noise using spectral subtraction fallback

        Args:
            audio: Input audio

        Returns:
            Noise-reduced audio
        """
        try:
            # Ensure float
            if audio.dtype != np.float32:
                audio_float = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
            else:
                audio_float = audio

            # Pad to FFT size
            if len(audio_float) < self.fft_size:
                audio_padded = np.pad(audio_float, (0, self.fft_size - len(audio_float)), mode='constant')
            else:
                audio_padded = audio_float[:self.fft_size]

            # FFT
            spectrum = np.fft.rfft(audio_padded)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Spectral subtraction
            if self.noise_spectrum is not None:
                # Subtract noise spectrum
                clean_magnitude = magnitude - (self.reduction_strength * self.noise_spectrum)

                # Floor to prevent negative magnitudes
                clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            else:
                clean_magnitude = magnitude

            # Reconstruct spectrum
            clean_spectrum = clean_magnitude * np.exp(1j * phase)

            # IFFT
            clean_audio = np.fft.irfft(clean_spectrum, n=self.fft_size)

            # Trim to original size
            clean_audio = clean_audio[:self.frame_size]

            # Convert back to original dtype
            if audio_data.dtype == np.int16:
                clean_audio = (clean_audio * 32767).astype(np.int16)
            else:
                clean_audio = clean_audio.astype(audio.dtype)

            return clean_audio

        except Exception as e:
            logger.error(f"Spectral subtraction error: {e}")
            return audio

    def _update_noise_profile(self, audio: np.ndarray):
        """Update noise profile from current frame"""
        try:
            # Ensure float
            if audio.dtype != np.float32:
                audio_float = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio.astype(np.float32)
            else:
                audio_float = audio

            # Pad to FFT size
            if len(audio_float) < self.fft_size:
                audio_padded = np.pad(audio_float, (0, self.fft_size - len(audio_float)), mode='constant')
            else:
                audio_padded = audio_float[:self.fft_size]

            # Get magnitude spectrum
            spectrum = np.fft.rfft(audio_padded)
            magnitude = np.abs(spectrum)

            # Update noise profile (running average)
            if self.noise_spectrum is None:
                self.noise_spectrum = magnitude
            else:
                alpha = 0.9  # Smoothing factor
                self.noise_spectrum = alpha * self.noise_spectrum + (1 - alpha) * magnitude

            logger.debug(f"Updated noise profile (frame {self.frames_processed + 1}/{self.noise_learning_frames})")

        except Exception as e:
            logger.error(f"Error updating noise profile: {e}")

    def learn_noise_from_silence(self, silence_audio: np.ndarray):
        """
        Explicitly learn noise profile from silence

        Args:
            silence_audio: Audio containing only noise (no speech)
        """
        try:
            # Process in chunks
            chunk_count = len(silence_audio) // self.frame_size

            for i in range(chunk_count):
                start = i * self.frame_size
                end = start + self.frame_size
                chunk = silence_audio[start:end]

                self._update_noise_profile(chunk)

            logger.info(f"Learned noise profile from {chunk_count} chunks of silence")

        except Exception as e:
            logger.error(f"Error learning noise from silence: {e}")

    def _normalize_frame(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Normalize audio frame to expected size"""
        try:
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.frombuffer(audio_data, dtype=np.int16)

            if len(audio_data) == self.frame_size:
                return audio_data
            elif len(audio_data) < self.frame_size:
                return np.pad(audio_data, (0, self.frame_size - len(audio_data)), mode='constant')
            else:
                return audio_data[:self.frame_size]

        except Exception as e:
            logger.error(f"Error normalizing frame: {e}")
            return None

    def _next_power_of_2(self, n: int) -> int:
        """Get next power of 2 >= n"""
        return 1 << (n - 1).bit_length()

    def reset(self):
        """Reset noise reduction state"""
        self.noise_profile = None
        self.noise_spectrum = None
        self.frames_processed = 0
        self.audio_buffer.clear()
        logger.info("Noise reduction state reset")

    def get_stats(self) -> dict:
        """Get noise reduction statistics"""
        return {
            "engine": "noisereduce" if self.use_noisereduce else "spectral_subtraction",
            "frames_processed": self.frames_processed,
            "reduction_strength": self.reduction_strength,
            "noise_profile_learned": self.noise_spectrum is not None,
            "sample_rate": self.sample_rate
        }

    def __repr__(self) -> str:
        """String representation"""
        engine = "noisereduce" if self.use_noisereduce else "spectral_subtraction"
        return (
            f"NoiseReductionProcessor("
            f"engine={engine}, "
            f"strength={self.reduction_strength:.2f}, "
            f"frames_processed={self.frames_processed})"
        )
