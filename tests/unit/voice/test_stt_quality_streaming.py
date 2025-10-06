"""Unit tests for STT Phase 2: Audio Quality Assessment and Streaming"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from src.voice.stt_service import STTService
from src.voice.models import STTConfig, AudioChunk


class TestAudioQualityAssessment:
    """Test audio quality assessment features"""

    @pytest.fixture
    def stt_service(self):
        """Create STTService instance for testing"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()
        return service

    def test_assess_high_quality_audio(self, stt_service):
        """Test quality assessment on high-quality audio (high SNR, no clipping)"""
        # Generate clean sine wave with moderate amplitude
        duration = 1.0
        sample_rate = 16000
        frequency = 440  # A4 note

        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t) * 0.3  # 30% amplitude
        audio = audio.astype(np.float32)

        quality = stt_service._assess_audio_quality(audio)

        assert quality['quality_score'] >= 0.7  # Should be high quality
        assert quality['snr'] >= 15.0  # Good SNR
        assert quality['clipping_rate'] < 0.01  # No clipping
        assert quality['assessment'] in ['good', 'excellent']

    def test_assess_clipped_audio(self, stt_service):
        """Test quality assessment on clipped audio"""
        # Generate audio that clips
        duration = 0.5
        sample_rate = 16000

        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        audio = np.clip(audio * 2.0, -1.0, 1.0)  # Force clipping

        quality = stt_service._assess_audio_quality(audio)

        assert quality['clipping_rate'] > 0.2  # High clipping rate
        assert quality['quality_score'] < 0.6  # Reduced quality due to clipping

    def test_assess_noisy_audio(self, stt_service):
        """Test quality assessment on noisy audio (low SNR)"""
        # Generate signal + noise
        duration = 1.0
        sample_rate = 16000

        # Weak signal
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))) * 0.1

        # Strong noise
        noise = np.random.randn(int(sample_rate * duration)) * 0.3

        audio = (signal + noise).astype(np.float32)

        quality = stt_service._assess_audio_quality(audio)

        assert quality['snr'] < 10.0  # Low SNR
        assert quality['quality_score'] < 0.5  # Poor quality

    def test_assess_silent_audio(self, stt_service):
        """Test quality assessment on mostly silent audio"""
        # Generate mostly silence with occasional blips
        audio = np.zeros(16000, dtype=np.float32)
        audio[1000:1100] = 0.1  # Small blip

        quality = stt_service._assess_audio_quality(audio)

        assert quality['silence_ratio'] > 0.9  # Very high silence
        assert quality['quality_score'] < 0.4  # Poor quality due to excessive silence
        assert quality['assessment'] in ['poor', 'very_poor']

    def test_assess_empty_audio(self, stt_service):
        """Test quality assessment on empty audio"""
        audio = np.array([], dtype=np.float32)

        quality = stt_service._assess_audio_quality(audio)

        assert quality['quality_score'] == 0.0
        assert quality['assessment'] == 'empty'
        assert quality['silence_ratio'] == 1.0

    def test_calculate_snr_pure_signal(self, stt_service):
        """Test SNR calculation on pure signal (no noise)"""
        # Pure sine wave
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        snr = stt_service._calculate_snr(audio)

        # Pure signal should have very high SNR
        assert snr >= 40.0

    def test_calculate_snr_noisy_signal(self, stt_service):
        """Test SNR calculation on noisy signal"""
        # Signal with significant noise
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))

        signal = np.sin(2 * np.pi * 440 * t) * 0.5
        noise = np.random.randn(len(signal)) * 0.3
        audio = (signal + noise).astype(np.float32)

        snr = stt_service._calculate_snr(audio)

        # Should have moderate to low SNR
        assert 0.0 <= snr <= 20.0

    def test_quality_score_calculation(self, stt_service):
        """Test overall quality score calculation"""
        # High quality: good SNR, no clipping, normal silence
        score_high = stt_service._calculate_quality_score(
            snr=30.0, clipping_rate=0.0, silence_ratio=0.2
        )
        assert score_high >= 0.8

        # Medium quality: moderate SNR, slight clipping
        score_medium = stt_service._calculate_quality_score(
            snr=15.0, clipping_rate=0.05, silence_ratio=0.3
        )
        assert 0.4 <= score_medium <= 0.7

        # Low quality: poor SNR, clipping, excessive silence
        score_low = stt_service._calculate_quality_score(
            snr=5.0, clipping_rate=0.2, silence_ratio=0.9
        )
        assert score_low <= 0.4


class TestVAD:
    """Test Voice Activity Detection"""

    @pytest.fixture
    def stt_service(self):
        """Create STTService instance for testing"""
        return STTService()

    def test_vad_detects_speech(self, stt_service):
        """Test VAD detects speech in audio with clear signal"""
        # Generate audio with speech-like characteristics
        duration = 1.0
        sample_rate = 16000

        # Simulate speech with varying amplitude
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.3
        audio[:8000] *= 2.0  # First half louder (speech)

        vad_result = stt_service._detect_speech_vad(audio)

        assert vad_result['speech_detected'] is True
        assert vad_result['speech_ratio'] > 0.1
        assert vad_result['energy_level'] > 0.001

    def test_vad_rejects_silence(self, stt_service):
        """Test VAD rejects pure silence"""
        # Generate silent audio
        audio = np.zeros(16000, dtype=np.float32)

        vad_result = stt_service._detect_speech_vad(audio)

        assert vad_result['speech_detected'] is False
        assert vad_result['speech_ratio'] < 0.1
        assert vad_result['energy_level'] < 0.001

    def test_vad_low_energy_noise(self, stt_service):
        """Test VAD on low-energy background noise"""
        # Generate very quiet noise
        audio = np.random.randn(16000).astype(np.float32) * 0.001

        vad_result = stt_service._detect_speech_vad(audio)

        assert vad_result['speech_detected'] is False

    def test_vad_empty_audio(self, stt_service):
        """Test VAD on empty audio"""
        audio = np.array([], dtype=np.float32)

        vad_result = stt_service._detect_speech_vad(audio)

        assert vad_result['speech_detected'] is False
        assert vad_result['speech_ratio'] == 0.0
        assert vad_result['energy_level'] == 0.0

    def test_vad_adaptive_threshold(self, stt_service):
        """Test VAD adaptive thresholding with varying energy"""
        # Generate audio with gradual increase in energy
        audio = np.random.randn(16000).astype(np.float32)

        # First half: low energy (noise)
        audio[:8000] *= 0.05

        # Second half: high energy (speech)
        audio[8000:] *= 0.5

        vad_result = stt_service._detect_speech_vad(audio)

        # Should detect the high-energy portion
        assert vad_result['speech_detected'] is True
        assert vad_result['speech_ratio'] > 0.3  # At least 30% speech


class TestQualityIntegration:
    """Integration tests for quality assessment in transcription"""

    @pytest.mark.asyncio
    async def test_transcription_includes_quality_metrics(self):
        """Test that transcription results include quality metrics"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        # Mock Whisper result
        mock_result = {
            'text': 'Test transcription',
            'segments': [{'avg_logprob': -0.5}],
            'language': 'en',
            'language_probability': 0.98
        }

        with patch.object(service, '_transcribe_with_whisper', return_value=mock_result):
            # Generate good quality audio
            audio_data = (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.3).astype(np.float32)
            audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()

            chunk = AudioChunk(
                data=audio_bytes,
                timestamp=1234567890.0,
                chunk_id=1,
                sample_rate=16000,
                channels=1
            )

            result = await service.transcribe_audio_chunk(chunk, "test_call")

            assert result is not None
            assert result.audio_quality_score is not None
            assert result.snr_db is not None
            assert result.quality_assessment is not None
            assert result.clipping_detected is False

    @pytest.mark.asyncio
    async def test_low_quality_audio_rejected(self):
        """Test that very low quality audio is rejected"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        # Generate very low quality audio (pure silence)
        audio_bytes = np.zeros(16000, dtype=np.int16).tobytes()

        chunk = AudioChunk(
            data=audio_bytes,
            timestamp=1234567890.0,
            chunk_id=1,
            sample_rate=16000,
            channels=1
        )

        result = await service.transcribe_audio_chunk(chunk, "test_call")

        # Should be rejected due to low quality
        assert result is None

    @pytest.mark.asyncio
    async def test_clipped_audio_flagged(self):
        """Test that clipped audio is flagged in results"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        # Mock Whisper result
        mock_result = {
            'text': 'Clipped audio test',
            'segments': [{'avg_logprob': -1.0}],
            'language': 'en',
            'language_probability': 0.95
        }

        with patch.object(service, '_transcribe_with_whisper', return_value=mock_result):
            # Generate clipped audio
            audio_data = np.clip(np.random.randn(16000) * 2.0, -1.0, 1.0).astype(np.float32)
            audio_bytes = (audio_data * 32768).astype(np.int16).tobytes()

            chunk = AudioChunk(
                data=audio_bytes,
                timestamp=1234567890.0,
                chunk_id=1,
                sample_rate=16000,
                channels=1
            )

            result = await service.transcribe_audio_chunk(chunk, "test_call")

            assert result is not None
            assert result.clipping_detected is True


class TestAdaptiveStreaming:
    """Test adaptive buffer management for streaming"""

    @pytest.mark.asyncio
    async def test_adaptive_buffer_adjusts_to_quality(self):
        """Test that buffer duration adapts based on audio quality"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        # Track transcription calls
        transcribe_calls = []

        async def mock_transcribe(chunk, call_id):
            # Simulate quality scores
            result = Mock()
            result.audio_quality_score = 0.9 if len(transcribe_calls) < 2 else 0.4
            result.text = f"Chunk {len(transcribe_calls)}"
            transcribe_calls.append((chunk, call_id))
            return result

        service.transcribe_audio_chunk = mock_transcribe

        # Create audio stream
        async def audio_generator():
            for i in range(10):
                # Generate 0.5 second chunks
                audio_bytes = (np.random.randn(8000) * 0.3).astype(np.float32)
                audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

                yield AudioChunk(
                    data=audio_bytes,
                    timestamp=float(i),
                    chunk_id=i,
                    sample_rate=16000,
                    channels=1,
                    silence_duration=0.1
                )

        results = []

        def callback(result):
            results.append(result)

        # Run streaming transcription
        await service.transcribe_stream(audio_generator(), "test_call", callback)

        # Should have processed multiple chunks
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_vad_prevents_silence_processing(self):
        """Test that VAD prevents processing of pure silence in streaming"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        transcribe_calls = []

        async def mock_transcribe(chunk, call_id):
            result = Mock()
            result.audio_quality_score = 0.8
            result.text = "Speech"
            transcribe_calls.append(call_id)
            return result

        service.transcribe_audio_chunk = mock_transcribe

        # Create stream with silence chunks
        async def audio_generator():
            for i in range(5):
                # Generate silent audio
                audio_bytes = np.zeros(8000, dtype=np.int16).tobytes()

                yield AudioChunk(
                    data=audio_bytes,
                    timestamp=float(i),
                    chunk_id=i,
                    sample_rate=16000,
                    channels=1,
                    silence_duration=0.5
                )

        results = []

        def callback(result):
            results.append(result)

        await service.transcribe_stream(audio_generator(), "test_call", callback)

        # Should process very few chunks (VAD filtering)
        assert len(transcribe_calls) <= 2

    @pytest.mark.asyncio
    async def test_streaming_respects_max_buffer(self):
        """Test that streaming respects maximum buffer duration"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        buffer_durations = []

        original_combine = service._combine_chunks

        def track_combine(chunks):
            # Calculate total duration
            duration = sum(len(c.data) / (c.sample_rate * c.channels * 2) for c in chunks)
            buffer_durations.append(duration)
            return original_combine(chunks)

        service._combine_chunks = track_combine

        async def mock_transcribe(chunk, call_id):
            result = Mock()
            result.audio_quality_score = 0.7
            result.text = "Test"
            return result

        service.transcribe_audio_chunk = mock_transcribe

        # Create continuous audio stream
        async def audio_generator():
            for i in range(20):
                audio_bytes = (np.random.randn(8000) * 0.3).astype(np.float32)
                audio_bytes = (audio_bytes * 32768).astype(np.int16).tobytes()

                yield AudioChunk(
                    data=audio_bytes,
                    timestamp=float(i * 0.5),
                    chunk_id=i,
                    sample_rate=16000,
                    channels=1,
                    silence_duration=0.0  # No silence
                )

        results = []

        def callback(result):
            results.append(result)

        await service.transcribe_stream(audio_generator(), "test_call", callback)

        # All buffers should respect max duration (3.0 seconds)
        for duration in buffer_durations:
            assert duration <= 3.1  # Allow small tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
