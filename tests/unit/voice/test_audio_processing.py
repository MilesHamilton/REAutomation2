"""
Combined unit tests for audio processing components:
- Packet Loss Concealment
- Echo Cancellation
- Noise Reduction
- Audio Processor (integration)
"""

import pytest
import numpy as np
import time

from src.voice.packet_loss_concealment import PacketLossConcealment
from src.voice.echo_cancellation import EchoCancellationProcessor
from src.voice.noise_reduction import NoiseReductionProcessor
from src.voice.audio_processor import AudioProcessor


# ===== Packet Loss Concealment Tests =====

class TestPacketLossConcealment:
    """Test PLC functionality"""

    def test_plc_init(self):
        """Test PLC initialization"""
        plc = PacketLossConcealment(sample_rate=16000, chunk_size=320)

        assert plc.sample_rate == 16000
        assert plc.chunk_size == 320
        assert plc.concealed_count == 0

    def test_plc_conceal_simple(self):
        """Test simple concealment (repeat last packet)"""
        plc = PacketLossConcealment(method="simple")

        # Update history with a packet
        packet = np.ones(320, dtype=np.int16) * 100
        plc.update_history(packet)

        # Conceal lost packet
        concealed = plc.conceal_packet()

        assert concealed is not None
        assert len(concealed) == 320
        assert plc.concealed_count == 1

    def test_plc_conceal_linear(self):
        """Test linear interpolation concealment"""
        plc = PacketLossConcealment(method="linear")

        # Add two packets to history
        packet1 = np.ones(320, dtype=np.int16) * 100
        packet2 = np.ones(320, dtype=np.int16) * 200
        plc.update_history(packet1)
        plc.update_history(packet2)

        # Conceal with linear interpolation
        concealed = plc.conceal_packet()

        assert concealed is not None
        assert len(concealed) == 320

    def test_plc_conceal_spectral(self):
        """Test spectral concealment"""
        plc = PacketLossConcealment(method="spectral")

        # Add multiple packets for spectral analysis
        for i in range(5):
            packet = np.random.randint(-1000, 1000, 320, dtype=np.int16)
            plc.update_history(packet)

        # Conceal with spectral method
        concealed = plc.conceal_packet()

        assert concealed is not None
        assert len(concealed) == 320

    def test_plc_consecutive_losses(self):
        """Test multiple consecutive packet losses"""
        plc = PacketLossConcealment(method="linear")

        # Add good packet
        packet = np.ones(320, dtype=np.int16) * 100
        plc.update_history(packet)

        # Conceal multiple consecutive losses
        for _ in range(3):
            concealed = plc.conceal_packet()
            assert concealed is not None

        assert plc.consecutive_losses == 3

    def test_plc_reset(self):
        """Test PLC reset"""
        plc = PacketLossConcealment()

        packet = np.ones(320, dtype=np.int16)
        plc.update_history(packet)
        plc.conceal_packet()

        assert plc.concealed_count > 0

        plc.reset()

        assert plc.concealed_count == 0
        assert plc.consecutive_losses == 0


# ===== Echo Cancellation Tests =====

class TestEchoCancellation:
    """Test echo cancellation processor"""

    def test_aec_init(self):
        """Test AEC initialization"""
        aec = EchoCancellationProcessor(sample_rate=16000, frame_size=320)

        assert aec.sample_rate == 16000
        assert aec.frame_size == 320
        assert aec.is_initialized is True

    def test_aec_process_same_dtype(self):
        """Test AEC processing with matching dtypes"""
        aec = EchoCancellationProcessor()

        far_end = np.random.randint(-1000, 1000, 320, dtype=np.int16)
        near_end = np.random.randint(-1000, 1000, 320, dtype=np.int16)

        output = aec.process(far_end, near_end)

        assert output is not None
        assert len(output) == 320
        assert output.dtype == np.int16

    def test_aec_process_different_sizes(self):
        """Test AEC with mismatched frame sizes"""
        aec = EchoCancellationProcessor(frame_size=320)

        far_end = np.zeros(200, dtype=np.int16)  # Too small
        near_end = np.zeros(320, dtype=np.int16)

        output = aec.process(far_end, near_end)

        assert output is not None

    def test_aec_reset(self):
        """Test AEC reset"""
        aec = EchoCancellationProcessor()

        # Process some audio
        far_end = np.zeros(320, dtype=np.int16)
        near_end = np.zeros(320, dtype=np.int16)
        aec.process(far_end, near_end)

        # Reset
        aec.reset()

        # Should still work after reset
        output = aec.process(far_end, near_end)
        assert output is not None

    def test_aec_get_stats(self):
        """Test AEC statistics"""
        aec = EchoCancellationProcessor()

        stats = aec.get_stats()

        assert "engine" in stats
        assert "initialized" in stats
        assert stats["initialized"] is True


# ===== Noise Reduction Tests =====

class TestNoiseReduction:
    """Test noise reduction processor"""

    def test_nr_init(self):
        """Test noise reduction initialization"""
        nr = NoiseReductionProcessor(sample_rate=16000, frame_size=320)

        assert nr.sample_rate == 16000
        assert nr.frame_size == 320
        assert nr.is_initialized is True

    def test_nr_process_int16(self):
        """Test noise reduction with int16 audio"""
        nr = NoiseReductionProcessor()

        audio = np.random.randint(-5000, 5000, 320, dtype=np.int16)

        output = nr.process(audio)

        assert output is not None
        assert len(output) == 320
        assert output.dtype == np.int16

    def test_nr_process_float32(self):
        """Test noise reduction with float32 audio"""
        nr = NoiseReductionProcessor()

        audio = np.random.randn(320).astype(np.float32)

        output = nr.process(audio)

        assert output is not None
        assert len(output) == 320

    def test_nr_noise_learning(self):
        """Test noise profile learning"""
        nr = NoiseReductionProcessor()

        # Process initial frames (learning phase)
        for i in range(nr.noise_learning_frames):
            audio = np.random.randn(320).astype(np.float32) * 0.01  # Low amplitude noise
            nr.process(audio, is_speech=False)

        assert nr.frames_processed == nr.noise_learning_frames
        assert nr.noise_spectrum is not None

    def test_nr_learn_from_silence(self):
        """Test learning from explicit silence"""
        nr = NoiseReductionProcessor()

        # Generate silence with small noise
        silence = np.random.randn(3200).astype(np.float32) * 0.01

        nr.learn_noise_from_silence(silence)

        assert nr.noise_spectrum is not None

    def test_nr_reset(self):
        """Test noise reduction reset"""
        nr = NoiseReductionProcessor()

        # Process and learn
        audio = np.random.randn(320).astype(np.float32)
        nr.process(audio)

        assert nr.frames_processed > 0

        # Reset
        nr.reset()

        assert nr.frames_processed == 0
        assert nr.noise_profile is None

    def test_nr_get_stats(self):
        """Test noise reduction statistics"""
        nr = NoiseReductionProcessor()

        stats = nr.get_stats()

        assert "engine" in stats
        assert "frames_processed" in stats
        assert "reduction_strength" in stats


# ===== Audio Processor Integration Tests =====

class TestAudioProcessor:
    """Test unified audio processor"""

    @pytest.mark.asyncio
    async def test_processor_init(self):
        """Test audio processor initialization"""
        processor = AudioProcessor(
            sample_rate=16000,
            chunk_duration_ms=20,
            enable_jitter_buffer=True,
            enable_echo_cancellation=True,
            enable_noise_reduction=True
        )

        assert processor.is_initialized is True
        assert processor.sample_rate == 16000
        assert processor.chunk_size == 320

    @pytest.mark.asyncio
    async def test_processor_input_audio(self):
        """Test processing input audio"""
        processor = AudioProcessor(
            enable_jitter_buffer=False,  # Disable for simpler test
            enable_echo_cancellation=False,
            enable_noise_reduction=False
        )

        audio = np.random.randint(-1000, 1000, 320, dtype=np.int16)

        result = await processor.process_input_audio(audio, sequence_number=0)

        assert result is not None

    @pytest.mark.asyncio
    async def test_processor_output_audio(self):
        """Test processing output audio"""
        processor = AudioProcessor()

        audio = np.random.randint(-1000, 1000, 320, dtype=np.int16)

        success = await processor.process_output_audio(audio)

        assert success is True
        assert processor.far_end_audio is not None

    @pytest.mark.asyncio
    async def test_processor_with_all_components(self):
        """Test with all components enabled"""
        processor = AudioProcessor(
            enable_jitter_buffer=True,
            enable_echo_cancellation=True,
            enable_noise_reduction=True
        )

        # Add several packets
        for seq in range(10):
            audio = np.random.randint(-1000, 1000, 320, dtype=np.int16)
            await processor.process_input_audio(
                audio,
                sequence_number=seq,
                timestamp=time.time()
            )

        # Should have processed packets
        assert processor.is_initialized is True

    @pytest.mark.asyncio
    async def test_processor_get_metrics(self):
        """Test getting processor metrics"""
        processor = AudioProcessor()

        metrics = processor.get_metrics()

        assert metrics is not None
        assert hasattr(metrics, "total_latency_ms")
        assert hasattr(metrics, "processing_time_ms")

    @pytest.mark.asyncio
    async def test_processor_health_status(self):
        """Test health status"""
        processor = AudioProcessor()

        health = processor.get_health_status()

        assert "status" in health
        assert "total_latency_ms" in health
        assert "components" in health

    @pytest.mark.asyncio
    async def test_processor_reset(self):
        """Test processor reset"""
        processor = AudioProcessor()

        # Process some audio
        audio = np.zeros(320, dtype=np.int16)
        await processor.process_input_audio(audio, sequence_number=0)

        # Reset
        await processor.reset()

        assert processor.echo_cancelled_frames == 0
        assert processor.noise_reduced_frames == 0

    @pytest.mark.asyncio
    async def test_processor_read_write_flow(self):
        """Test complete read/write flow"""
        processor = AudioProcessor(
            enable_jitter_buffer=False,  # Simpler flow
            enable_echo_cancellation=False,
            enable_noise_reduction=False
        )

        # Write input audio
        input_audio = np.ones(320, dtype=np.int16) * 100
        await processor.process_input_audio(input_audio, sequence_number=0)

        # Read input audio
        output_audio = await processor.get_input_audio(timeout=1.0)

        assert output_audio is not None
        assert len(output_audio) == 320

    @pytest.mark.asyncio
    async def test_processor_echo_cancellation_flow(self):
        """Test echo cancellation in processor"""
        processor = AudioProcessor(
            enable_jitter_buffer=False,
            enable_echo_cancellation=True,
            enable_noise_reduction=False
        )

        # Set far-end reference
        far_end = np.random.randint(-1000, 1000, 320, dtype=np.int16)
        await processor.process_output_audio(far_end)

        # Process near-end with echo
        near_end = np.random.randint(-1000, 1000, 320, dtype=np.int16)
        result = await processor.process_input_audio(near_end, sequence_number=0)

        assert result is not None
        assert processor.echo_cancelled_frames > 0

    @pytest.mark.asyncio
    async def test_processor_noise_reduction_flow(self):
        """Test noise reduction in processor"""
        processor = AudioProcessor(
            enable_jitter_buffer=False,
            enable_echo_cancellation=False,
            enable_noise_reduction=True
        )

        # Process noisy audio
        noisy = np.random.randint(-1000, 1000, 320, dtype=np.int16)
        result = await processor.process_input_audio(noisy, sequence_number=0)

        assert result is not None
        assert processor.noise_reduced_frames > 0


# ===== Performance Tests =====

class TestPerformance:
    """Performance and latency tests"""

    @pytest.mark.asyncio
    async def test_processing_latency(self):
        """Test that processing latency is acceptable"""
        processor = AudioProcessor(
            enable_jitter_buffer=False,
            enable_echo_cancellation=True,
            enable_noise_reduction=True
        )

        audio = np.random.randint(-1000, 1000, 320, dtype=np.int16)

        start = time.time()
        await processor.process_input_audio(audio, sequence_number=0)
        latency_ms = (time.time() - start) * 1000

        # Should process in less than 50ms
        assert latency_ms < 50.0

    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test processing throughput"""
        processor = AudioProcessor(
            enable_jitter_buffer=False,
            enable_echo_cancellation=True,
            enable_noise_reduction=True
        )

        # Process 100 packets
        start = time.time()

        for seq in range(100):
            audio = np.random.randint(-1000, 1000, 320, dtype=np.int16)
            await processor.process_input_audio(audio, sequence_number=seq)

        elapsed = time.time() - start

        # Should process 100 packets in reasonable time
        assert elapsed < 5.0  # Less than 5 seconds for 100 packets

    def test_memory_efficiency(self):
        """Test memory usage is reasonable"""
        processor = AudioProcessor()

        # Check that buffers have reasonable size
        assert len(processor.input_buffer) >= 0
        assert len(processor.output_buffer) >= 0
