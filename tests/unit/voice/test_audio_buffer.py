"""
Unit tests for AudioBufferManager
"""

import pytest
import asyncio
import numpy as np
import time

from src.voice.audio_buffer import AudioBufferManager, BufferMetrics


@pytest.fixture
def buffer_manager():
    """Create audio buffer manager for testing"""
    return AudioBufferManager(
        sample_rate=16000,
        chunk_duration_ms=20,
        buffer_duration_ms=500,
        channels=1,
        dtype="int16"
    )


class TestAudioBufferManagerInit:
    """Test AudioBufferManager initialization"""

    def test_init_default(self):
        """Test default initialization"""
        buffer = AudioBufferManager()

        assert buffer.sample_rate == 16000
        assert buffer.chunk_duration_ms == 20
        assert buffer.chunk_size == 320  # 20ms @ 16kHz
        assert buffer.max_chunks == 25  # 500ms / 20ms

    def test_init_custom(self):
        """Test custom initialization"""
        buffer = AudioBufferManager(
            sample_rate=8000,
            chunk_duration_ms=10,
            buffer_duration_ms=200
        )

        assert buffer.sample_rate == 8000
        assert buffer.chunk_duration_ms == 10
        assert buffer.chunk_size == 80  # 10ms @ 8kHz
        assert buffer.max_chunks == 20  # 200ms / 10ms


class TestAudioBufferWrite:
    """Test writing audio chunks to buffer"""

    @pytest.mark.asyncio
    async def test_write_chunk_perfect_size(self, buffer_manager):
        """Test writing chunk with perfect size"""
        audio = np.zeros(320, dtype=np.int16)

        success = await buffer_manager.write_chunk(audio)

        assert success is True
        assert len(buffer_manager) == 1
        assert buffer_manager.metrics.total_chunks_processed == 1

    @pytest.mark.asyncio
    async def test_write_chunk_with_timestamp(self, buffer_manager):
        """Test writing chunk with custom timestamp"""
        audio = np.zeros(320, dtype=np.int16)
        timestamp = time.time()

        success = await buffer_manager.write_chunk(audio, timestamp)

        assert success is True
        assert len(buffer_manager) == 1

    @pytest.mark.asyncio
    async def test_write_chunk_too_small(self, buffer_manager):
        """Test writing chunk smaller than expected (should pad)"""
        audio = np.zeros(200, dtype=np.int16)

        success = await buffer_manager.write_chunk(audio)

        assert success is True
        assert len(buffer_manager) == 1

    @pytest.mark.asyncio
    async def test_write_chunk_too_large(self, buffer_manager):
        """Test writing chunk larger than expected (should truncate)"""
        audio = np.zeros(500, dtype=np.int16)

        success = await buffer_manager.write_chunk(audio)

        assert success is True
        assert len(buffer_manager) == 1

    @pytest.mark.asyncio
    async def test_write_multiple_chunks(self, buffer_manager):
        """Test writing multiple chunks"""
        for i in range(10):
            audio = np.ones(320, dtype=np.int16) * i
            await buffer_manager.write_chunk(audio)

        assert len(buffer_manager) == 10
        assert buffer_manager.metrics.total_chunks_processed == 10

    @pytest.mark.asyncio
    async def test_write_overflow(self, buffer_manager):
        """Test buffer overflow (exceeds max_chunks)"""
        # Fill buffer to max
        for i in range(buffer_manager.max_chunks + 5):
            audio = np.zeros(320, dtype=np.int16)
            await buffer_manager.write_chunk(audio)

        # Should have dropped oldest chunks
        assert len(buffer_manager) == buffer_manager.max_chunks
        assert buffer_manager.metrics.overruns == 5


class TestAudioBufferRead:
    """Test reading audio chunks from buffer"""

    @pytest.mark.asyncio
    async def test_read_chunk_empty_buffer(self, buffer_manager):
        """Test reading from empty buffer (should return None with timeout)"""
        chunk = await buffer_manager.read_chunk(timeout=0.1)

        assert chunk is None
        assert buffer_manager.metrics.underruns == 1

    @pytest.mark.asyncio
    async def test_read_chunk_single(self, buffer_manager):
        """Test reading single chunk"""
        audio = np.ones(320, dtype=np.int16) * 100
        await buffer_manager.write_chunk(audio)

        chunk = await buffer_manager.read_chunk(timeout=1.0)

        assert chunk is not None
        assert len(chunk) == 320
        assert np.array_equal(chunk, audio)
        assert len(buffer_manager) == 0

    @pytest.mark.asyncio
    async def test_read_chunk_nowait(self, buffer_manager):
        """Test synchronous read"""
        audio = np.ones(320, dtype=np.int16) * 100
        await buffer_manager.write_chunk(audio)

        chunk = buffer_manager.read_chunk_nowait()

        assert chunk is not None
        assert len(chunk) == 320

    @pytest.mark.asyncio
    async def test_read_chunk_nowait_empty(self, buffer_manager):
        """Test synchronous read from empty buffer"""
        chunk = buffer_manager.read_chunk_nowait()

        assert chunk is None
        assert buffer_manager.metrics.underruns == 1

    @pytest.mark.asyncio
    async def test_read_multiple_chunks_fifo(self, buffer_manager):
        """Test reading multiple chunks in FIFO order"""
        # Write chunks with unique values
        for i in range(5):
            audio = np.ones(320, dtype=np.int16) * i
            await buffer_manager.write_chunk(audio)

        # Read chunks and verify order
        for i in range(5):
            chunk = await buffer_manager.read_chunk(timeout=1.0)
            assert chunk is not None
            assert chunk[0] == i  # First element should match index

    @pytest.mark.asyncio
    async def test_latency_tracking(self, buffer_manager):
        """Test latency tracking"""
        # Write chunk with timestamp
        audio = np.zeros(320, dtype=np.int16)
        timestamp = time.time()
        await buffer_manager.write_chunk(audio, timestamp)

        # Wait a bit
        await asyncio.sleep(0.05)

        # Read chunk
        chunk = await buffer_manager.read_chunk(timeout=1.0)

        assert chunk is not None
        assert buffer_manager.metrics.average_latency_ms > 0


class TestAudioBufferMetrics:
    """Test buffer metrics and monitoring"""

    @pytest.mark.asyncio
    async def test_get_metrics(self, buffer_manager):
        """Test getting metrics"""
        metrics = buffer_manager.get_metrics()

        assert isinstance(metrics, BufferMetrics)
        assert metrics.buffer_fill_percentage == 0.0
        assert metrics.underruns == 0
        assert metrics.overruns == 0

    @pytest.mark.asyncio
    async def test_buffer_fill_percentage(self, buffer_manager):
        """Test buffer fill percentage calculation"""
        # Fill half the buffer
        num_chunks = buffer_manager.max_chunks // 2
        for _ in range(num_chunks):
            audio = np.zeros(320, dtype=np.int16)
            await buffer_manager.write_chunk(audio)

        metrics = buffer_manager.get_metrics()

        assert 45.0 <= metrics.buffer_fill_percentage <= 55.0  # ~50%

    @pytest.mark.asyncio
    async def test_get_buffer_level(self, buffer_manager):
        """Test getting buffer level"""
        # Add 10 chunks
        for _ in range(10):
            audio = np.zeros(320, dtype=np.int16)
            await buffer_manager.write_chunk(audio)

        current, max_chunks, fill_pct = buffer_manager.get_buffer_level()

        assert current == 10
        assert max_chunks == buffer_manager.max_chunks
        assert 35.0 <= fill_pct <= 45.0  # ~40%

    @pytest.mark.asyncio
    async def test_get_latency_ms(self, buffer_manager):
        """Test latency calculation"""
        # Add 5 chunks
        for _ in range(5):
            audio = np.zeros(320, dtype=np.int16)
            await buffer_manager.write_chunk(audio)

        latency = buffer_manager.get_latency_ms()

        assert latency == 100.0  # 5 chunks * 20ms

    @pytest.mark.asyncio
    async def test_is_healthy_normal(self, buffer_manager):
        """Test health check with normal fill level"""
        # Fill to 50%
        num_chunks = buffer_manager.max_chunks // 2
        for _ in range(num_chunks):
            audio = np.zeros(320, dtype=np.int16)
            await buffer_manager.write_chunk(audio)

        assert buffer_manager.is_healthy() is True

    @pytest.mark.asyncio
    async def test_is_healthy_too_empty(self, buffer_manager):
        """Test health check with too low fill level"""
        # Add only 1 chunk (4% of 25 chunks buffer)
        audio = np.zeros(320, dtype=np.int16)
        await buffer_manager.write_chunk(audio)

        assert buffer_manager.is_healthy(min_fill=10.0) is False

    @pytest.mark.asyncio
    async def test_is_healthy_too_full(self, buffer_manager):
        """Test health check with too high fill level"""
        # Fill to 95%
        num_chunks = int(buffer_manager.max_chunks * 0.95)
        for _ in range(num_chunks):
            audio = np.zeros(320, dtype=np.int16)
            await buffer_manager.write_chunk(audio)

        assert buffer_manager.is_healthy(max_fill=90.0) is False


class TestAudioBufferOperations:
    """Test buffer operations"""

    @pytest.mark.asyncio
    async def test_clear_buffer(self, buffer_manager):
        """Test clearing buffer"""
        # Add some chunks
        for _ in range(10):
            audio = np.zeros(320, dtype=np.int16)
            await buffer_manager.write_chunk(audio)

        assert len(buffer_manager) == 10

        # Clear buffer
        await buffer_manager.clear()

        assert len(buffer_manager) == 0

    def test_len(self, buffer_manager):
        """Test __len__ method"""
        assert len(buffer_manager) == 0

    def test_repr(self, buffer_manager):
        """Test string representation"""
        repr_str = repr(buffer_manager)

        assert "AudioBufferManager" in repr_str
        assert "latency" in repr_str
        assert "underruns" in repr_str


class TestAudioBufferConcurrency:
    """Test concurrent read/write operations"""

    @pytest.mark.asyncio
    async def test_concurrent_write(self, buffer_manager):
        """Test multiple concurrent writes"""
        async def write_chunks(count):
            for i in range(count):
                audio = np.ones(320, dtype=np.int16) * i
                await buffer_manager.write_chunk(audio)

        # Start multiple write tasks
        tasks = [write_chunks(5) for _ in range(3)]
        await asyncio.gather(*tasks)

        assert len(buffer_manager) == 15
        assert buffer_manager.metrics.total_chunks_processed == 15

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, buffer_manager):
        """Test concurrent reads and writes"""
        write_count = 0
        read_count = 0

        async def writer():
            nonlocal write_count
            for i in range(20):
                audio = np.zeros(320, dtype=np.int16)
                await buffer_manager.write_chunk(audio)
                write_count += 1
                await asyncio.sleep(0.01)

        async def reader():
            nonlocal read_count
            for _ in range(20):
                chunk = await buffer_manager.read_chunk(timeout=1.0)
                if chunk is not None:
                    read_count += 1
                await asyncio.sleep(0.01)

        # Run concurrently
        await asyncio.gather(writer(), reader())

        # Should have processed most/all chunks
        assert write_count == 20
        assert read_count >= 15  # May have some still in buffer


class TestAudioBufferDtypes:
    """Test different audio data types"""

    @pytest.mark.asyncio
    async def test_int16_dtype(self):
        """Test with int16 dtype"""
        buffer = AudioBufferManager(dtype="int16")
        audio = np.zeros(320, dtype=np.int16)

        await buffer.write_chunk(audio)
        chunk = await buffer.read_chunk(timeout=1.0)

        assert chunk.dtype == np.int16

    @pytest.mark.asyncio
    async def test_float32_dtype(self):
        """Test with float32 dtype"""
        buffer = AudioBufferManager(dtype="float32")
        audio = np.zeros(320, dtype=np.float32)

        await buffer.write_chunk(audio)
        chunk = await buffer.read_chunk(timeout=1.0)

        assert chunk.dtype == np.float32

    @pytest.mark.asyncio
    async def test_dtype_conversion(self):
        """Test automatic dtype conversion"""
        buffer = AudioBufferManager(dtype="int16")
        audio = np.zeros(320, dtype=np.float32)

        await buffer.write_chunk(audio)
        chunk = await buffer.read_chunk(timeout=1.0)

        assert chunk.dtype == np.int16
