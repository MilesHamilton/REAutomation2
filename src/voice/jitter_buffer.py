"""
Adaptive Jitter Buffer for VoIP Audio Streaming

Handles network jitter, packet reordering, and packet loss
for real-time audio communication.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque
import heapq

from .packet_loss_concealment import PacketLossConcealment

logger = logging.getLogger(__name__)


@dataclass
class JitterMetrics:
    """Metrics for jitter buffer performance"""
    jitter_ms: float = 0.0
    packet_loss_rate: float = 0.0
    buffer_delay_ms: float = 0.0
    late_packets: int = 0
    duplicate_packets: int = 0
    out_of_order_packets: int = 0
    total_packets_received: int = 0
    total_packets_played: int = 0
    concealed_packets: int = 0
    buffer_adjustments: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AudioPacket:
    """Audio packet with metadata"""
    sequence_number: int
    timestamp: float
    data: np.ndarray
    arrival_time: float = field(default_factory=time.time)
    played: bool = False


class AdaptiveJitterBuffer:
    """
    Adaptive jitter buffer for VoIP audio

    Features:
    - Adaptive buffer sizing based on network jitter
    - Packet reordering by sequence number
    - Packet loss detection and concealment
    - Late packet handling
    - Duplicate packet detection
    """

    def __init__(
        self,
        target_delay_ms: int = 80,
        min_delay_ms: int = 40,
        max_delay_ms: int = 200,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        adaptive: bool = True
    ):
        """
        Initialize adaptive jitter buffer

        Args:
            target_delay_ms: Target buffer delay (default: 80ms)
            min_delay_ms: Minimum buffer delay (default: 40ms)
            max_delay_ms: Maximum buffer delay (default: 200ms)
            sample_rate: Audio sample rate (default: 16000)
            chunk_duration_ms: Audio chunk duration (default: 20ms)
            adaptive: Enable adaptive sizing (default: True)
        """
        self.target_delay_ms = target_delay_ms
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.adaptive = adaptive

        # Calculate chunk size
        self.chunk_size = int((chunk_duration_ms / 1000.0) * sample_rate)

        # Packet buffer (priority queue by sequence number)
        self.buffer: list = []  # Min heap: [(seq_num, AudioPacket), ...]
        self.received_packets: Dict[int, AudioPacket] = {}

        # State
        self.next_sequence = 0
        self.last_played_sequence = -1
        self.is_initialized = False
        self.playout_started = False

        # Metrics
        self.metrics = JitterMetrics()

        # Jitter calculation
        self._arrival_times: deque = deque(maxlen=100)  # Last 100 packet arrival times
        self._inter_arrival_times: deque = deque(maxlen=100)

        # Packet loss concealment
        self.plc = PacketLossConcealment(
            sample_rate=sample_rate,
            chunk_size=self.chunk_size
        )

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info(
            f"AdaptiveJitterBuffer initialized: "
            f"target={target_delay_ms}ms, "
            f"range={min_delay_ms}-{max_delay_ms}ms, "
            f"adaptive={'ON' if adaptive else 'OFF'}"
        )

    async def add_packet(
        self,
        audio_data: np.ndarray,
        sequence_number: int,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Add audio packet to jitter buffer

        Args:
            audio_data: Audio data as numpy array
            sequence_number: Packet sequence number
            timestamp: Optional RTP timestamp

        Returns:
            True if packet added successfully
        """
        try:
            async with self._lock:
                arrival_time = time.time()

                # Check for duplicate
                if sequence_number in self.received_packets:
                    self.metrics.duplicate_packets += 1
                    logger.debug(f"Duplicate packet: {sequence_number}")
                    return False

                # Check if packet is late (already played)
                if self.playout_started and sequence_number < self.last_played_sequence:
                    self.metrics.late_packets += 1
                    logger.debug(f"Late packet: {sequence_number}, last played: {self.last_played_sequence}")
                    return False

                # Create packet
                packet = AudioPacket(
                    sequence_number=sequence_number,
                    timestamp=timestamp or arrival_time,
                    data=audio_data,
                    arrival_time=arrival_time
                )

                # Add to buffer
                heapq.heappush(self.buffer, (sequence_number, packet))
                self.received_packets[sequence_number] = packet
                self.metrics.total_packets_received += 1

                # Update jitter metrics
                self._update_jitter_metrics(arrival_time, sequence_number)

                # Check for out-of-order
                if sequence_number < self.next_sequence - 1:
                    self.metrics.out_of_order_packets += 1

                # Initialize on first packet
                if not self.is_initialized:
                    self.next_sequence = sequence_number
                    self.is_initialized = True
                    logger.info(f"Jitter buffer initialized with sequence: {sequence_number}")

                # Adaptive buffer adjustment
                if self.adaptive:
                    await self._adjust_buffer_size()

                return True

        except Exception as e:
            logger.error(f"Error adding packet to jitter buffer: {e}")
            return False

    async def get_packet(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get next audio packet from buffer

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Audio data or None if not available
        """
        try:
            start_time = time.time()

            while True:
                async with self._lock:
                    # Wait until buffer has enough delay
                    if not self.playout_started:
                        buffer_delay = self._get_current_buffer_delay_ms()
                        if buffer_delay >= self.target_delay_ms:
                            self.playout_started = True
                            logger.info(f"Starting playout with buffer delay: {buffer_delay:.1f}ms")

                    # Try to get next packet
                    if self.playout_started and len(self.buffer) > 0:
                        # Get packet with lowest sequence number
                        seq_num, packet = heapq.heappop(self.buffer)

                        # Check if this is the expected packet
                        if seq_num == self.next_sequence:
                            # Perfect - got expected packet
                            self.last_played_sequence = seq_num
                            self.next_sequence += 1
                            self.metrics.total_packets_played += 1
                            packet.played = True

                            # Update PLC with good packet
                            self.plc.update_history(packet.data)

                            return packet.data

                        elif seq_num > self.next_sequence:
                            # Missing packet(s) - need concealment
                            missing_count = seq_num - self.next_sequence
                            logger.warning(f"Missing {missing_count} packet(s): {self.next_sequence} to {seq_num-1}")

                            # Put packet back
                            heapq.heappush(self.buffer, (seq_num, packet))

                            # Generate concealment for next expected packet
                            concealed = self.plc.conceal_packet()
                            self.metrics.concealed_packets += 1
                            self.last_played_sequence = self.next_sequence
                            self.next_sequence += 1

                            # Update packet loss rate
                            self._update_packet_loss_rate()

                            return concealed

                        else:
                            # Should not happen (packet < next_sequence)
                            logger.error(f"Unexpected sequence number: {seq_num} < {self.next_sequence}")
                            continue

                # Check timeout
                if timeout is not None and (time.time() - start_time) > timeout:
                    logger.debug("Get packet timeout")
                    return None

                # Wait before retry
                await asyncio.sleep(self.chunk_duration_ms / 2000.0)

        except Exception as e:
            logger.error(f"Error getting packet from jitter buffer: {e}")
            return None

    def _update_jitter_metrics(self, arrival_time: float, sequence_number: int):
        """Update jitter statistics"""
        try:
            # Record arrival time
            self._arrival_times.append((sequence_number, arrival_time))

            # Calculate inter-arrival time
            if len(self._arrival_times) >= 2:
                # Get last two packets
                prev_seq, prev_time = self._arrival_times[-2]
                curr_seq, curr_time = self._arrival_times[-1]

                # Inter-arrival time (actual)
                inter_arrival = curr_time - prev_time

                # Expected inter-arrival time (based on sequence numbers)
                expected_interval = (curr_seq - prev_seq) * (self.chunk_duration_ms / 1000.0)

                # Jitter = deviation from expected
                jitter = abs(inter_arrival - expected_interval) * 1000.0  # Convert to ms

                self._inter_arrival_times.append(jitter)

                # Calculate average jitter
                if len(self._inter_arrival_times) > 0:
                    self.metrics.jitter_ms = np.mean(self._inter_arrival_times)

        except Exception as e:
            logger.error(f"Error updating jitter metrics: {e}")

    def _get_current_buffer_delay_ms(self) -> float:
        """Get current buffer delay in milliseconds"""
        if len(self.buffer) == 0:
            return 0.0

        # Time difference between oldest packet and now
        _, oldest_packet = self.buffer[0]
        delay = (time.time() - oldest_packet.arrival_time) * 1000.0
        return delay

    async def _adjust_buffer_size(self):
        """Adjust target delay based on network conditions"""
        try:
            if not self.adaptive:
                return

            # Adjust based on jitter
            if self.metrics.jitter_ms > 50.0:
                # High jitter - increase buffer
                new_target = min(self.target_delay_ms + 20, self.max_delay_ms)
                if new_target != self.target_delay_ms:
                    self.target_delay_ms = new_target
                    self.metrics.buffer_adjustments += 1
                    logger.info(f"Increased buffer delay to {self.target_delay_ms}ms (high jitter: {self.metrics.jitter_ms:.1f}ms)")

            elif self.metrics.jitter_ms < 10.0 and self.metrics.packet_loss_rate < 0.01:
                # Low jitter and low loss - decrease buffer
                new_target = max(self.target_delay_ms - 10, self.min_delay_ms)
                if new_target != self.target_delay_ms:
                    self.target_delay_ms = new_target
                    self.metrics.buffer_adjustments += 1
                    logger.info(f"Decreased buffer delay to {self.target_delay_ms}ms (low jitter: {self.metrics.jitter_ms:.1f}ms)")

        except Exception as e:
            logger.error(f"Error adjusting buffer size: {e}")

    def _update_packet_loss_rate(self):
        """Update packet loss rate metric"""
        if self.metrics.total_packets_received > 0:
            expected = self.next_sequence
            received = self.metrics.total_packets_received
            lost = expected - received
            self.metrics.packet_loss_rate = max(0.0, lost / expected)

    def get_metrics(self) -> JitterMetrics:
        """Get current jitter buffer metrics"""
        self.metrics.buffer_delay_ms = self._get_current_buffer_delay_ms()
        self.metrics.timestamp = time.time()
        return self.metrics

    def get_buffer_level(self) -> Tuple[int, float]:
        """
        Get buffer fill level

        Returns:
            Tuple of (packet_count, delay_ms)
        """
        return (len(self.buffer), self._get_current_buffer_delay_ms())

    async def reset(self):
        """Reset jitter buffer to initial state"""
        async with self._lock:
            self.buffer.clear()
            self.received_packets.clear()
            self.next_sequence = 0
            self.last_played_sequence = -1
            self.is_initialized = False
            self.playout_started = False
            self._arrival_times.clear()
            self._inter_arrival_times.clear()
            self.plc.reset()
            logger.info("Jitter buffer reset")

    def __len__(self) -> int:
        """Return number of packets in buffer"""
        return len(self.buffer)

    def __repr__(self) -> str:
        """String representation"""
        packet_count, delay_ms = self.get_buffer_level()
        return (
            f"AdaptiveJitterBuffer("
            f"packets={packet_count}, "
            f"delay={delay_ms:.1f}ms, "
            f"jitter={self.metrics.jitter_ms:.1f}ms, "
            f"loss={self.metrics.packet_loss_rate*100:.1f}%, "
            f"concealed={self.metrics.concealed_packets})"
        )
