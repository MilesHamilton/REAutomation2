"""
Packet Loss Concealment (PLC) for Audio Streams

Provides algorithms to conceal packet loss in real-time audio,
from simple repetition to spectral interpolation.
"""

import logging
import numpy as np
from typing import Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class PacketLossConcealment:
    """
    Packet Loss Concealment for VoIP audio

    Provides multiple strategies for concealing lost audio packets:
    - Simple: Repeat last packet
    - Linear: Interpolate between packets
    - Spectral: Use FFT-based estimation (advanced)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 320,  # 20ms @ 16kHz
        history_size: int = 10,
        method: str = "linear"
    ):
        """
        Initialize packet loss concealment

        Args:
            sample_rate: Audio sample rate (default: 16000)
            chunk_size: Audio chunk size in samples (default: 320)
            history_size: Number of packets to keep in history (default: 10)
            method: Concealment method (simple, linear, spectral)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.history_size = history_size
        self.method = method

        # Packet history (for interpolation)
        self.history: deque = deque(maxlen=history_size)

        # State
        self.last_packet: Optional[np.ndarray] = None
        self.concealed_count = 0
        self.consecutive_losses = 0

        logger.info(
            f"PacketLossConcealment initialized: "
            f"{sample_rate}Hz, {chunk_size} samples/chunk, "
            f"method={method}"
        )

    def conceal_packet(self) -> np.ndarray:
        """
        Generate concealment for a lost packet

        Returns:
            Concealed audio data as numpy array
        """
        try:
            self.concealed_count += 1
            self.consecutive_losses += 1

            # Choose concealment method
            if self.method == "simple":
                return self._conceal_simple()
            elif self.method == "linear":
                return self._conceal_linear()
            elif self.method == "spectral":
                return self._conceal_spectral()
            else:
                logger.warning(f"Unknown method '{self.method}', using simple")
                return self._conceal_simple()

        except Exception as e:
            logger.error(f"Error in packet concealment: {e}")
            return self._generate_silence()

    def _conceal_simple(self) -> np.ndarray:
        """
        Simple concealment: Repeat last packet

        Good for: <5% packet loss
        """
        if self.last_packet is not None:
            # Repeat last packet
            concealed = self.last_packet.copy()

            # Apply fade-out if multiple consecutive losses
            if self.consecutive_losses > 1:
                fade_factor = 0.9 ** (self.consecutive_losses - 1)
                concealed = (concealed * fade_factor).astype(self.last_packet.dtype)

            return concealed

        # No history - return silence
        return self._generate_silence()

    def _conceal_linear(self) -> np.ndarray:
        """
        Linear interpolation between packets

        Better than simple for 5-10% loss
        """
        if len(self.history) >= 2:
            # Get last two packets
            packet_prev = self.history[-2]
            packet_last = self.history[-1]

            # Linear interpolation
            # Predict next packet by extrapolating the trend
            diff = packet_last.astype(np.float32) - packet_prev.astype(np.float32)

            # Dampen the prediction to avoid divergence
            damping = 0.7 ** self.consecutive_losses
            concealed = packet_last.astype(np.float32) + (diff * damping)

            # Clip to valid range
            if packet_last.dtype == np.int16:
                concealed = np.clip(concealed, -32768, 32767).astype(np.int16)
            else:
                concealed = np.clip(concealed, -1.0, 1.0).astype(np.float32)

            return concealed

        # Fallback to simple
        return self._conceal_simple()

    def _conceal_spectral(self) -> np.ndarray:
        """
        Spectral concealment using FFT

        Best quality but higher latency
        """
        if len(self.history) >= 3:
            try:
                # Get last 3 packets for better spectral estimation
                packets = [self.history[-3], self.history[-2], self.history[-1]]

                # Convert to frequency domain
                spectra = [np.fft.rfft(p.astype(np.float32)) for p in packets]

                # Estimate phase and magnitude progression
                magnitudes = [np.abs(s) for s in spectra]
                phases = [np.angle(s) for s in spectra]

                # Linear extrapolation in frequency domain
                mag_diff_1 = magnitudes[-1] - magnitudes[-2]
                mag_diff_2 = magnitudes[-2] - magnitudes[-3]
                mag_avg_diff = (mag_diff_1 + mag_diff_2) / 2

                phase_diff_1 = phases[-1] - phases[-2]
                phase_diff_2 = phases[-2] - phases[-3]
                phase_avg_diff = (phase_diff_1 + phase_diff_2) / 2

                # Predict next spectrum
                damping = 0.8 ** self.consecutive_losses
                pred_magnitude = magnitudes[-1] + (mag_avg_diff * damping)
                pred_phase = phases[-1] + (phase_avg_diff * damping)

                # Reconstruct complex spectrum
                pred_spectrum = pred_magnitude * np.exp(1j * pred_phase)

                # Convert back to time domain
                concealed = np.fft.irfft(pred_spectrum, n=self.chunk_size)

                # Clip to valid range
                if self.last_packet.dtype == np.int16:
                    concealed = np.clip(concealed, -32768, 32767).astype(np.int16)
                else:
                    concealed = np.clip(concealed, -1.0, 1.0).astype(np.float32)

                return concealed

            except Exception as e:
                logger.error(f"Spectral concealment failed: {e}")
                return self._conceal_linear()

        # Fallback to linear
        return self._conceal_linear()

    def update_history(self, packet: np.ndarray):
        """
        Update packet history with a successfully received packet

        Args:
            packet: Audio packet data
        """
        try:
            # Ensure correct size
            if len(packet) != self.chunk_size:
                logger.warning(f"Packet size mismatch: {len(packet)} != {self.chunk_size}")
                # Resize packet
                if len(packet) < self.chunk_size:
                    packet = np.pad(packet, (0, self.chunk_size - len(packet)), mode='constant')
                else:
                    packet = packet[:self.chunk_size]

            # Add to history
            self.history.append(packet.copy())
            self.last_packet = packet.copy()

            # Reset consecutive loss counter
            self.consecutive_losses = 0

        except Exception as e:
            logger.error(f"Error updating PLC history: {e}")

    def _generate_silence(self) -> np.ndarray:
        """Generate silence packet"""
        # Determine dtype from last packet or default to int16
        dtype = self.last_packet.dtype if self.last_packet is not None else np.int16
        return np.zeros(self.chunk_size, dtype=dtype)

    def get_concealment_stats(self) -> dict:
        """Get concealment statistics"""
        return {
            "total_concealed": self.concealed_count,
            "consecutive_losses": self.consecutive_losses,
            "history_size": len(self.history),
            "method": self.method
        }

    def reset(self):
        """Reset PLC state"""
        self.history.clear()
        self.last_packet = None
        self.concealed_count = 0
        self.consecutive_losses = 0
        logger.info("PLC state reset")

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"PacketLossConcealment("
            f"method={self.method}, "
            f"concealed={self.concealed_count}, "
            f"consecutive={self.consecutive_losses}, "
            f"history={len(self.history)})"
        )
