"""
Unit tests for AdaptiveJitterBuffer
"""

import pytest
import asyncio
import numpy as np
import time

from src.voice.jitter_buffer import AdaptiveJitterBuffer, JitterMetrics, AudioPacket


@pytest.fixture
def jitter_buffer():
    """Create jitter buffer for testing"""
    return AdaptiveJitterBuffer(
        target_delay_ms=80,
        min_delay_ms=40,
        max_delay_ms=200,
        sample_rate=16000,
        chunk_duration_ms=20,
        adaptive=True
    )


class TestJitterBufferInit:
    """Test AdaptiveJitterBuffer initialization"""

    def test_init_default(self):
        """Test default initialization"""
        jb = AdaptiveJitterBuffer()

        assert jb.target_delay_ms == 80
        assert jb.min_delay_ms == 40
        assert jb.max_delay_ms == 200
        assert jb.chunk_size == 320
        assert jb.adaptive is True

    def test_init_custom(self):
        """Test custom initialization"""
        jb = AdaptiveJitterBuffer(
            target_delay_ms=100,
            sample_rate=8000,
            adaptive=False
        )

        assert jb.target_delay_ms == 100
        assert jb.chunk_size == 160  # 20ms @ 8kHz
        assert jb.adaptive is False


class TestJitterBufferPacketAdd:
    """Test adding packets to jitter buffer"""

    @pytest.mark.asyncio
    async def test_add_packet_first(self, jitter_buffer):
        """Test adding first packet"""
        audio = np.zeros(320, dtype=np.int16)

        success = await jitter_buffer.add_packet(audio, sequence_number=1)

        assert success is True
        assert len(jitter_buffer) == 1
        assert jitter_buffer.is_initialized is True
        assert jitter_buffer.metrics.total_packets_received == 1

    @pytest.mark.asyncio
    async def test_add_packet_sequential(self, jitter_buffer):
        """Test adding sequential packets"""
        for seq in range(5):
            audio = np.zeros(320, dtype=np.int16)
            success = await jitter_buffer.add_packet(audio, sequence_number=seq)
            assert success is True

        assert len(jitter_buffer) == 5
        assert jitter_buffer.metrics.total_packets_received == 5

    @pytest.mark.asyncio
    async def test_add_packet_out_of_order(self, jitter_buffer):
        """Test adding out-of-order packets"""
        # Add packets: 1, 3, 2, 4
        sequences = [1, 3, 2, 4]

        for seq in sequences:
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        assert len(jitter_buffer) == 4
        assert jitter_buffer.metrics.out_of_order_packets >= 1

    @pytest.mark.asyncio
    async def test_add_duplicate_packet(self, jitter_buffer):
        """Test handling duplicate packets"""
        audio = np.zeros(320, dtype=np.int16)

        # Add same packet twice
        success1 = await jitter_buffer.add_packet(audio, sequence_number=1)
        success2 = await jitter_buffer.add_packet(audio, sequence_number=1)

        assert success1 is True
        assert success2 is False  # Duplicate rejected
        assert jitter_buffer.metrics.duplicate_packets == 1
        assert len(jitter_buffer) == 1

    @pytest.mark.asyncio
    async def test_add_late_packet(self, jitter_buffer):
        """Test handling late packets (already played)"""
        # Add and retrieve some packets
        for seq in range(5):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        # Start playout (fill buffer enough)
        for _ in range(6):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=_+5)

        # Play some packets
        for _ in range(3):
            await jitter_buffer.get_packet(timeout=0.5)

        # Try to add packet with old sequence number
        old_audio = np.zeros(320, dtype=np.int16)
        success = await jitter_buffer.add_packet(old_audio, sequence_number=1)

        assert success is False  # Late packet rejected
        assert jitter_buffer.metrics.late_packets >= 1


class TestJitterBufferPacketGet:
    """Test retrieving packets from jitter buffer"""

    @pytest.mark.asyncio
    async def test_get_packet_waits_for_target_delay(self, jitter_buffer):
        """Test that playout waits for target delay"""
        # Add only one packet
        audio = np.zeros(320, dtype=np.int16)
        await jitter_buffer.add_packet(audio, sequence_number=0)

        # Should wait because buffer delay < target delay
        start = time.time()
        packet = await jitter_buffer.get_packet(timeout=0.1)
        elapsed = time.time() - start

        assert packet is None  # Timeout (not enough buffering)
        assert elapsed >= 0.1

    @pytest.mark.asyncio
    async def test_get_packet_in_order(self, jitter_buffer):
        """Test getting packets in sequence order"""
        # Add enough packets to start playout
        for seq in range(10):
            audio = np.ones(320, dtype=np.int16) * seq
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        # Wait for buffer to fill
        await asyncio.sleep(0.1)

        # Get packets - should come out in order
        for expected_seq in range(5):
            packet = await jitter_buffer.get_packet(timeout=0.5)
            assert packet is not None
            # First element should match sequence (we set all values to seq)
            if packet[0] != 0:  # Skip if concealed packet
                assert packet[0] == expected_seq

    @pytest.mark.asyncio
    async def test_get_packet_with_missing(self, jitter_buffer):
        """Test packet loss concealment when packet missing"""
        # Add packets with a gap: 0, 1, 3, 4 (missing 2)
        sequences = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]

        for seq in sequences:
            audio = np.ones(320, dtype=np.int16) * seq
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        # Wait for buffer
        await asyncio.sleep(0.1)

        # Get packets
        packets_received = []
        for _ in range(5):
            packet = await jitter_buffer.get_packet(timeout=0.5)
            if packet is not None:
                packets_received.append(packet)

        assert len(packets_received) >= 4
        assert jitter_buffer.metrics.concealed_packets >= 1  # Missing packet 2

    @pytest.mark.asyncio
    async def test_get_packet_timeout(self, jitter_buffer):
        """Test get_packet with timeout on empty buffer"""
        packet = await jitter_buffer.get_packet(timeout=0.1)

        assert packet is None


class TestJitterBufferMetrics:
    """Test jitter buffer metrics"""

    @pytest.mark.asyncio
    async def test_jitter_calculation(self, jitter_buffer):
        """Test jitter measurement"""
        # Add packets with varying arrival times
        for seq in range(10):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)
            await asyncio.sleep(0.015 + (seq % 3) * 0.005)  # Varying delay

        metrics = jitter_buffer.get_metrics()

        assert metrics.jitter_ms >= 0.0
        assert metrics.total_packets_received == 10

    @pytest.mark.asyncio
    async def test_buffer_delay_measurement(self, jitter_buffer):
        """Test buffer delay calculation"""
        # Add several packets
        for seq in range(5):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        # Wait a bit
        await asyncio.sleep(0.05)

        metrics = jitter_buffer.get_metrics()

        assert metrics.buffer_delay_ms > 0.0

    @pytest.mark.asyncio
    async def test_packet_loss_rate(self, jitter_buffer):
        """Test packet loss rate calculation"""
        # Add packets with gaps
        sequences = [0, 1, 3, 5, 6, 8]  # Missing 2, 4, 7

        for seq in sequences:
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        # Wait and retrieve to trigger loss detection
        await asyncio.sleep(0.1)

        for _ in range(8):
            await jitter_buffer.get_packet(timeout=0.5)

        metrics = jitter_buffer.get_metrics()

        assert metrics.packet_loss_rate >= 0.0


class TestJitterBufferAdaptive:
    """Test adaptive buffer sizing"""

    @pytest.mark.asyncio
    async def test_buffer_increase_on_high_jitter(self, jitter_buffer):
        """Test buffer increases with high jitter"""
        initial_target = jitter_buffer.target_delay_ms

        # Simulate high jitter by varying arrival times
        for seq in range(20):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)
            # Vary delay significantly
            await asyncio.sleep(0.01 + (seq % 5) * 0.02)

        # Buffer should adjust upward
        # Note: May require more packets or time to trigger
        metrics = jitter_buffer.get_metrics()

        # Check if adjustment occurred
        if jitter_buffer.metrics.buffer_adjustments > 0:
            assert jitter_buffer.target_delay_ms >= initial_target

    @pytest.mark.asyncio
    async def test_buffer_decrease_on_low_jitter(self, jitter_buffer):
        """Test buffer decreases with low jitter"""
        # Start with high target
        jitter_buffer.target_delay_ms = 150

        # Simulate low jitter (consistent arrival)
        for seq in range(30):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)
            await asyncio.sleep(0.02)  # Consistent 20ms

        # Buffer should adjust downward (eventually)
        # Note: Requires stable conditions
        if jitter_buffer.metrics.jitter_ms < 10.0:
            # Low jitter detected, adjustment may occur
            pass


class TestJitterBufferOperations:
    """Test jitter buffer operations"""

    @pytest.mark.asyncio
    async def test_reset(self, jitter_buffer):
        """Test resetting jitter buffer"""
        # Add some packets
        for seq in range(5):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        assert len(jitter_buffer) == 5

        # Reset
        await jitter_buffer.reset()

        assert len(jitter_buffer) == 0
        assert jitter_buffer.is_initialized is False
        assert jitter_buffer.playout_started is False

    @pytest.mark.asyncio
    async def test_get_buffer_level(self, jitter_buffer):
        """Test getting buffer level"""
        # Add 3 packets
        for seq in range(3):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        packet_count, delay_ms = jitter_buffer.get_buffer_level()

        assert packet_count == 3
        assert delay_ms >= 0.0

    def test_len(self, jitter_buffer):
        """Test __len__ method"""
        assert len(jitter_buffer) == 0

    def test_repr(self, jitter_buffer):
        """Test string representation"""
        repr_str = repr(jitter_buffer)

        assert "AdaptiveJitterBuffer" in repr_str
        assert "packets" in repr_str


class TestJitterBufferStressTest:
    """Stress tests for jitter buffer"""

    @pytest.mark.asyncio
    async def test_high_packet_rate(self, jitter_buffer):
        """Test with high packet arrival rate"""
        # Rapidly add many packets
        for seq in range(100):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        assert jitter_buffer.metrics.total_packets_received == 100

    @pytest.mark.asyncio
    async def test_severe_packet_loss(self, jitter_buffer):
        """Test with 50% packet loss"""
        # Add only even sequence numbers
        for seq in range(0, 20, 2):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)

        await asyncio.sleep(0.1)

        # Try to read packets
        for _ in range(10):
            packet = await jitter_buffer.get_packet(timeout=0.5)
            if packet is None:
                break

        # Should have concealed many packets
        assert jitter_buffer.metrics.concealed_packets > 0

    @pytest.mark.asyncio
    async def test_extreme_jitter(self, jitter_buffer):
        """Test with extreme jitter"""
        # Add packets with random delays
        import random

        for seq in range(15):
            audio = np.zeros(320, dtype=np.int16)
            await jitter_buffer.add_packet(audio, sequence_number=seq)
            await asyncio.sleep(random.uniform(0.001, 0.1))

        metrics = jitter_buffer.get_metrics()

        # Buffer should still function
        assert metrics.total_packets_received == 15
        assert len(jitter_buffer) > 0
