"""Unit tests for STT Phase 3: faster-whisper Integration and Performance"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from src.voice.stt_service import STTService, FASTER_WHISPER_AVAILABLE
from src.voice.models import STTConfig, AudioChunk


class TestFasterWhisperBackend:
    """Test faster-whisper backend integration"""

    @pytest.fixture
    def standard_config(self):
        """Config for standard whisper"""
        return STTConfig(
            model="whisper-small",
            use_faster_whisper=False
        )

    @pytest.fixture
    def faster_config(self):
        """Config for faster-whisper"""
        return STTConfig(
            model="whisper-small",
            use_faster_whisper=True,
            compute_type="float16",
            beam_size=5,
            num_workers=1
        )

    @pytest.mark.asyncio
    async def test_backend_selection_standard(self, standard_config):
        """Test that standard whisper is used when configured"""
        service = STTService(config=standard_config)

        with patch.object(service, '_load_whisper_model') as mock_load_standard:
            mock_load_standard.return_value = Mock()

            await service.initialize()

            assert service.use_faster_whisper is False
            mock_load_standard.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not FASTER_WHISPER_AVAILABLE, reason="faster-whisper not installed")
    async def test_backend_selection_faster(self, faster_config):
        """Test that faster-whisper is used when configured and available"""
        service = STTService(config=faster_config)

        with patch.object(service, '_load_faster_whisper_model') as mock_load_faster:
            mock_load_faster.return_value = Mock()

            await service.initialize()

            assert service.use_faster_whisper is True
            mock_load_faster.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_when_faster_unavailable(self, faster_config):
        """Test fallback to standard whisper when faster-whisper unavailable"""
        # Temporarily disable faster-whisper availability
        import src.voice.stt_service as stt_module
        original_available = stt_module.FASTER_WHISPER_AVAILABLE

        try:
            stt_module.FASTER_WHISPER_AVAILABLE = False

            service = STTService(config=faster_config)

            with patch.object(service, '_load_whisper_model') as mock_load_standard:
                mock_load_standard.return_value = Mock()

                await service.initialize()

                # Should fall back to standard whisper
                assert service.use_faster_whisper is False
                mock_load_standard.assert_called_once()

        finally:
            # Restore original value
            stt_module.FASTER_WHISPER_AVAILABLE = original_available

    def test_compute_type_adjustment_for_cpu(self, faster_config):
        """Test that compute_type is adjusted from float16 to int8 on CPU"""
        service = STTService(config=faster_config)

        with patch('torch.cuda.is_available', return_value=False):
            if FASTER_WHISPER_AVAILABLE:
                with patch('src.voice.stt_service.FasterWhisperModel') as MockModel:
                    service._load_faster_whisper_model()

                    # Should adjust compute_type to int8 for CPU
                    call_args = MockModel.call_args
                    assert call_args[1]['compute_type'] == 'int8'
                    assert call_args[1]['device'] == 'cpu'

    @pytest.mark.skipif(not FASTER_WHISPER_AVAILABLE, reason="faster-whisper not installed")
    def test_faster_whisper_transcription_format(self):
        """Test that faster-whisper transcription returns correct format"""
        service = STTService(config=STTConfig(use_faster_whisper=True))
        service.model = Mock()
        service.use_faster_whisper = True

        # Mock faster-whisper response
        mock_segment = Mock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = "Test transcription"
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        service.model.transcribe.return_value = ([mock_segment], mock_info)

        audio = np.random.randn(16000).astype(np.float32)
        result = service._transcribe_with_faster_whisper(audio)

        assert result['text'] == "Test transcription"
        assert len(result['segments']) == 1
        assert result['segments'][0]['text'] == "Test transcription"
        assert result['segments'][0]['avg_logprob'] == -0.5
        assert result['language'] == "en"
        assert result['language_probability'] == 0.95


class TestBatchProcessing:
    """Test batch transcription optimization"""

    @pytest.mark.asyncio
    async def test_batch_transcription_basic(self):
        """Test basic batch transcription"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        # Create test chunks
        chunks = []
        for i in range(3):
            audio_bytes = (np.random.randn(8000) * 0.3).astype(np.float32)
            audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

            chunks.append(AudioChunk(
                data=audio_bytes,
                timestamp=float(i),
                chunk_id=i,
                sample_rate=16000,
                channels=1
            ))

        # Mock transcription
        async def mock_transcribe(chunk, call_id):
            result = Mock()
            result.text = f"Chunk {chunk.chunk_id}"
            result.chunk_id = chunk.chunk_id
            return result

        service.transcribe_audio_chunk = mock_transcribe

        # Batch transcribe
        results = await service.transcribe_batch(chunks, "test_call")

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert results[0].text == "Chunk 0"
        assert results[1].text == "Chunk 1"
        assert results[2].text == "Chunk 2"

    @pytest.mark.asyncio
    async def test_batch_handles_errors(self):
        """Test that batch transcription handles individual failures"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        chunks = []
        for i in range(3):
            audio_bytes = (np.random.randn(8000) * 0.3).astype(np.float32)
            audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

            chunks.append(AudioChunk(
                data=audio_bytes,
                timestamp=float(i),
                chunk_id=i,
                sample_rate=16000,
                channels=1
            ))

        # Mock transcription with one failure
        async def mock_transcribe(chunk, call_id):
            if chunk.chunk_id == 1:
                raise ValueError("Transcription failed")

            result = Mock()
            result.text = f"Chunk {chunk.chunk_id}"
            return result

        service.transcribe_audio_chunk = mock_transcribe

        # Batch transcribe
        results = await service.transcribe_batch(chunks, "test_call")

        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None  # Failed
        assert results[2] is not None

    @pytest.mark.asyncio
    async def test_batch_maintains_order(self):
        """Test that batch results maintain input order"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        chunks = []
        for i in range(5):
            audio_bytes = (np.random.randn(8000) * 0.3).astype(np.float32)
            audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

            chunks.append(AudioChunk(
                data=audio_bytes,
                timestamp=float(i),
                chunk_id=i,
                sample_rate=16000,
                channels=1
            ))

        # Mock with random delays
        async def mock_transcribe(chunk, call_id):
            import asyncio
            # Random delay between 0.01 and 0.05 seconds
            await asyncio.sleep(0.01 + (chunk.chunk_id % 5) * 0.01)

            result = Mock()
            result.text = f"Chunk {chunk.chunk_id}"
            result.chunk_id = chunk.chunk_id
            return result

        service.transcribe_audio_chunk = mock_transcribe

        results = await service.transcribe_batch(chunks, "test_call")

        # Verify order is maintained despite varying processing times
        for i, result in enumerate(results):
            assert result.chunk_id == i


class TestPerformanceBenchmark:
    """Performance benchmarking tests"""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_standard_whisper_performance(self):
        """Benchmark standard whisper performance"""
        config = STTConfig(use_faster_whisper=False)
        service = STTService(config=config)
        service.is_initialized = True
        service.model = Mock()
        service.use_faster_whisper = False

        # Mock standard whisper transcription
        def mock_standard_transcribe(audio, **kwargs):
            # Simulate processing time
            time.sleep(0.1)
            return {
                'text': 'Test transcription',
                'segments': [{'avg_logprob': -0.5}],
                'language': 'en',
                'language_probability': 0.95
            }

        service.model.transcribe = mock_standard_transcribe

        # Generate test audio
        audio_bytes = (np.random.randn(16000) * 0.3).astype(np.float32)
        audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

        chunk = AudioChunk(
            data=audio_bytes,
            timestamp=time.time(),
            chunk_id=1,
            sample_rate=16000,
            channels=1
        )

        # Benchmark
        start = time.time()
        result = await service.transcribe_audio_chunk(chunk, "test_call")
        elapsed = time.time() - start

        assert result is not None
        assert elapsed > 0.1  # Should take at least simulated time
        print(f"\nStandard Whisper: {elapsed:.3f}s")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.skipif(not FASTER_WHISPER_AVAILABLE, reason="faster-whisper not installed")
    async def test_faster_whisper_performance(self):
        """Benchmark faster-whisper performance"""
        config = STTConfig(use_faster_whisper=True, beam_size=5)
        service = STTService(config=config)
        service.is_initialized = True
        service.model = Mock()
        service.use_faster_whisper = True

        # Mock faster-whisper transcription (should be faster)
        def mock_faster_transcribe(audio, **kwargs):
            # Simulate faster processing
            time.sleep(0.05)  # Half the time of standard

            mock_segment = Mock()
            mock_segment.start = 0.0
            mock_segment.end = 1.0
            mock_segment.text = "Test transcription"
            mock_segment.avg_logprob = -0.5
            mock_segment.no_speech_prob = 0.1

            mock_info = Mock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95

            return ([mock_segment], mock_info)

        service.model.transcribe = mock_faster_transcribe

        # Generate test audio
        audio_bytes = (np.random.randn(16000) * 0.3).astype(np.float32)
        audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

        chunk = AudioChunk(
            data=audio_bytes,
            timestamp=time.time(),
            chunk_id=1,
            sample_rate=16000,
            channels=1
        )

        # Benchmark
        start = time.time()
        result = await service.transcribe_audio_chunk(chunk, "test_call")
        elapsed = time.time() - start

        assert result is not None
        assert elapsed > 0.05
        print(f"\nFaster-Whisper: {elapsed:.3f}s")

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_batch_vs_sequential_performance(self):
        """Compare batch vs sequential processing performance"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        # Create test chunks
        chunks = []
        for i in range(5):
            audio_bytes = (np.random.randn(8000) * 0.3).astype(np.float32)
            audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

            chunks.append(AudioChunk(
                data=audio_bytes,
                timestamp=float(i),
                chunk_id=i,
                sample_rate=16000,
                channels=1
            ))

        # Mock with simulated delay
        async def mock_transcribe(chunk, call_id):
            import asyncio
            await asyncio.sleep(0.05)  # 50ms per chunk

            result = Mock()
            result.text = f"Chunk {chunk.chunk_id}"
            return result

        service.transcribe_audio_chunk = mock_transcribe

        # Sequential processing
        start_sequential = time.time()
        sequential_results = []
        for chunk in chunks:
            result = await service.transcribe_audio_chunk(chunk, "test_call")
            sequential_results.append(result)
        sequential_time = time.time() - start_sequential

        # Batch processing
        start_batch = time.time()
        batch_results = await service.transcribe_batch(chunks, "test_call")
        batch_time = time.time() - start_batch

        print(f"\nSequential: {sequential_time:.3f}s")
        print(f"Batch: {batch_time:.3f}s")
        print(f"Speedup: {sequential_time / batch_time:.2f}x")

        # Batch should be significantly faster (near-parallel execution)
        assert batch_time < sequential_time * 0.6  # At least 40% faster


class TestConfiguration:
    """Test Phase 3 configuration options"""

    def test_beam_size_configuration(self):
        """Test beam_size configuration"""
        config = STTConfig(beam_size=10)
        assert config.beam_size == 10

    def test_compute_type_configuration(self):
        """Test compute_type configuration"""
        config = STTConfig(compute_type="int8")
        assert config.compute_type == "int8"

    def test_num_workers_configuration(self):
        """Test num_workers configuration"""
        config = STTConfig(num_workers=4)
        assert config.num_workers == 4

    def test_use_faster_whisper_default(self):
        """Test use_faster_whisper defaults to True"""
        config = STTConfig()
        assert config.use_faster_whisper is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not benchmark"])
