"""Unit tests for STT confidence scoring and language detection"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.voice.stt_service import STTService
from src.voice.models import STTConfig, AudioChunk


class TestConfidenceScoring:
    """Test confidence score calculation from Whisper segments"""

    @pytest.fixture
    def stt_service(self):
        """Create STTService instance for testing"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()  # Mock Whisper model
        return service

    def test_high_confidence_segments(self, stt_service):
        """Test that high-quality segments yield high confidence"""
        segments = [
            {'avg_logprob': -0.3, 'no_speech_prob': 0.05},
            {'avg_logprob': -0.4, 'no_speech_prob': 0.03},
            {'avg_logprob': -0.5, 'no_speech_prob': 0.02}
        ]

        confidence = stt_service._calculate_confidence(segments)

        # High confidence should be close to 1.0
        assert confidence >= 0.9
        assert confidence <= 1.0

    def test_low_confidence_segments(self, stt_service):
        """Test that low-quality segments yield low confidence"""
        segments = [
            {'avg_logprob': -3.5, 'no_speech_prob': 0.6},
            {'avg_logprob': -4.0, 'no_speech_prob': 0.7},
            {'avg_logprob': -3.2, 'no_speech_prob': 0.55}
        ]

        confidence = stt_service._calculate_confidence(segments)

        # Low confidence should be close to 0.0
        assert confidence <= 0.3
        assert confidence >= 0.0

    def test_medium_confidence_segments(self, stt_service):
        """Test that medium-quality segments yield medium confidence"""
        segments = [
            {'avg_logprob': -1.5, 'no_speech_prob': 0.2},
            {'avg_logprob': -1.8, 'no_speech_prob': 0.25},
            {'avg_logprob': -1.6, 'no_speech_prob': 0.15}
        ]

        confidence = stt_service._calculate_confidence(segments)

        # Medium confidence should be in middle range
        assert confidence >= 0.4
        assert confidence <= 0.7

    def test_empty_segments(self, stt_service):
        """Test that empty segments return 0 confidence"""
        segments = []

        confidence = stt_service._calculate_confidence(segments)

        assert confidence == 0.0

    def test_segments_without_probabilities(self, stt_service):
        """Test that segments without probability info return default confidence"""
        segments = [
            {'text': 'hello world'},
            {'text': 'how are you'}
        ]

        confidence = stt_service._calculate_confidence(segments)

        # Should return default moderate confidence
        assert confidence == 0.7

    def test_no_speech_probability_reduces_confidence(self, stt_service):
        """Test that high no_speech_prob reduces confidence"""
        # High confidence with low no_speech_prob
        segments_low_nsp = [
            {'avg_logprob': -0.3, 'no_speech_prob': 0.05}
        ]
        confidence_low_nsp = stt_service._calculate_confidence(segments_low_nsp)

        # High confidence with high no_speech_prob
        segments_high_nsp = [
            {'avg_logprob': -0.3, 'no_speech_prob': 0.8}
        ]
        confidence_high_nsp = stt_service._calculate_confidence(segments_high_nsp)

        # High no_speech_prob should significantly reduce confidence
        assert confidence_high_nsp < confidence_low_nsp
        assert confidence_high_nsp < 0.5

    def test_confidence_bounds(self, stt_service):
        """Test that confidence is always bounded between 0 and 1"""
        # Test with extreme values
        extreme_segments = [
            {'avg_logprob': -10.0, 'no_speech_prob': 0.99},
            {'avg_logprob': 5.0, 'no_speech_prob': 0.0},
            {'avg_logprob': -0.1, 'no_speech_prob': 0.01}
        ]

        confidence = stt_service._calculate_confidence(extreme_segments)

        assert confidence >= 0.0
        assert confidence <= 1.0

    def test_mixed_segment_quality(self, stt_service):
        """Test confidence with mixed quality segments"""
        segments = [
            {'avg_logprob': -0.5, 'no_speech_prob': 0.1},  # Good
            {'avg_logprob': -2.0, 'no_speech_prob': 0.3},  # Medium
            {'avg_logprob': -3.5, 'no_speech_prob': 0.5}   # Poor
        ]

        confidence = stt_service._calculate_confidence(segments)

        # Should be medium confidence (average)
        assert confidence >= 0.3
        assert confidence <= 0.7


class TestLanguageDetection:
    """Test language detection functionality"""

    @pytest.fixture
    def stt_service(self):
        """Create STTService instance for testing"""
        config = STTConfig(auto_detect_language=True)
        service = STTService(config=config)
        service.is_initialized = True
        service.model = Mock()
        return service

    @pytest.mark.asyncio
    async def test_language_detection_enabled(self, stt_service):
        """Test that language detection works when enabled"""
        # Mock Whisper transcription result
        mock_result = {
            'text': 'Bonjour, comment allez-vous?',
            'segments': [{'avg_logprob': -0.5}],
            'language': 'fr',
            'language_probability': 0.95
        }

        with patch.object(stt_service, '_transcribe_with_whisper', return_value=mock_result):
            # Create test audio chunk
            audio_chunk = AudioChunk(
                data=np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes(),
                timestamp=1234567890.0,
                chunk_id=1,
                sample_rate=16000,
                channels=1
            )

            result = await stt_service.transcribe_audio_chunk(audio_chunk, "test_call_123")

            assert result is not None
            assert result.detected_language == 'fr'
            assert result.language_probability == 0.95
            assert result.text == 'Bonjour, comment allez-vous?'

    @pytest.mark.asyncio
    async def test_language_detection_disabled(self):
        """Test that language is fixed when auto-detection is disabled"""
        config = STTConfig(auto_detect_language=False, language='en')
        service = STTService(config=config)
        service.is_initialized = True
        service.model = Mock()

        # Mock Whisper transcription result
        mock_result = {
            'text': 'Hello, how are you?',
            'segments': [{'avg_logprob': -0.5}],
            'language': 'en',
            'language_probability': 1.0
        }

        with patch.object(service, '_transcribe_with_whisper', return_value=mock_result):
            audio_chunk = AudioChunk(
                data=np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes(),
                timestamp=1234567890.0,
                chunk_id=1,
                sample_rate=16000,
                channels=1
            )

            result = await service.transcribe_audio_chunk(audio_chunk, "test_call_123")

            assert result is not None
            assert result.language == 'en'
            assert result.detected_language == 'en'

    def test_transcribe_with_whisper_auto_detect(self, stt_service):
        """Test _transcribe_with_whisper with auto language detection"""
        # Create mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': 'Hola, ¿cómo estás?',
            'segments': [
                {'avg_logprob': -0.6, 'no_speech_prob': 0.1}
            ],
            'language': 'es',
            'language_probability': 0.92
        }
        stt_service.model = mock_model

        # Create test audio
        audio_array = np.random.randn(16000).astype(np.float32)

        result = stt_service._transcribe_with_whisper(audio_array)

        # Verify transcribe was called with language=None for auto-detection
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs['language'] is None
        assert call_kwargs['task'] == 'transcribe'
        assert call_kwargs['word_timestamps'] is True

        # Verify result structure
        assert result['text'] == 'Hola, ¿cómo estás?'
        assert result['language'] == 'es'
        assert result['language_probability'] == 0.92

    def test_transcribe_with_whisper_fixed_language(self):
        """Test _transcribe_with_whisper with fixed language"""
        config = STTConfig(auto_detect_language=False, language='de')
        service = STTService(config=config)
        service.is_initialized = True

        # Create mock model
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            'text': 'Guten Tag, wie geht es Ihnen?',
            'segments': [{'avg_logprob': -0.7}],
            'language': 'de',
            'language_probability': 1.0
        }
        service.model = mock_model

        audio_array = np.random.randn(16000).astype(np.float32)

        result = service._transcribe_with_whisper(audio_array)

        # Verify transcribe was called with language='de'
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs['language'] == 'de'


class TestIntegration:
    """Integration tests for confidence and language detection"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_confidence(self):
        """Test complete transcription pipeline with real confidence"""
        config = STTConfig(auto_detect_language=True)
        service = STTService(config=config)
        service.is_initialized = True
        service.model = Mock()

        # Mock comprehensive Whisper result
        mock_result = {
            'text': 'This is a test transcription.',
            'segments': [
                {'avg_logprob': -0.8, 'no_speech_prob': 0.05},
                {'avg_logprob': -0.9, 'no_speech_prob': 0.08},
                {'avg_logprob': -0.7, 'no_speech_prob': 0.03}
            ],
            'language': 'en',
            'language_probability': 0.98
        }

        with patch.object(service, '_transcribe_with_whisper', return_value=mock_result):
            audio_chunk = AudioChunk(
                data=np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes(),
                timestamp=1234567890.0,
                chunk_id=1,
                sample_rate=16000,
                channels=1
            )

            result = await service.transcribe_audio_chunk(audio_chunk, "test_call_456")

            assert result is not None
            assert result.text == 'This is a test transcription.'
            assert result.confidence >= 0.8  # Should be high confidence
            assert result.detected_language == 'en'
            assert result.language_probability == 0.98
            assert result.call_id == "test_call_456"
            assert result.chunk_id == 1

    @pytest.mark.asyncio
    async def test_low_quality_audio_rejection(self):
        """Test that low-confidence audio can be identified"""
        service = STTService()
        service.is_initialized = True
        service.model = Mock()

        # Mock poor quality result
        mock_result = {
            'text': 'um... uh...',
            'segments': [
                {'avg_logprob': -4.0, 'no_speech_prob': 0.8}
            ],
            'language': 'en',
            'language_probability': 0.3
        }

        with patch.object(service, '_transcribe_with_whisper', return_value=mock_result):
            audio_chunk = AudioChunk(
                data=np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes(),
                timestamp=1234567890.0,
                chunk_id=1,
                sample_rate=16000,
                channels=1
            )

            result = await service.transcribe_audio_chunk(audio_chunk, "test_call_789")

            assert result is not None
            # Confidence should be very low
            assert result.confidence < 0.3
            # Language probability should reflect uncertainty
            assert result.language_probability < 0.5

    @pytest.mark.asyncio
    async def test_multilingual_conversation(self):
        """Test handling multiple languages in same call"""
        config = STTConfig(auto_detect_language=True)
        service = STTService(config=config)
        service.is_initialized = True
        service.model = Mock()

        # Simulate English chunk
        mock_result_en = {
            'text': 'Hello, can you help me?',
            'segments': [{'avg_logprob': -0.5}],
            'language': 'en',
            'language_probability': 0.95
        }

        # Simulate Spanish chunk
        mock_result_es = {
            'text': 'Sí, por supuesto.',
            'segments': [{'avg_logprob': -0.6}],
            'language': 'es',
            'language_probability': 0.93
        }

        with patch.object(service, '_transcribe_with_whisper', side_effect=[mock_result_en, mock_result_es]):
            audio_chunk_1 = AudioChunk(
                data=np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes(),
                timestamp=1234567890.0,
                chunk_id=1,
                sample_rate=16000,
                channels=1
            )

            audio_chunk_2 = AudioChunk(
                data=np.random.randint(-32768, 32767, 16000, dtype=np.int16).tobytes(),
                timestamp=1234567892.0,
                chunk_id=2,
                sample_rate=16000,
                channels=1
            )

            result_1 = await service.transcribe_audio_chunk(audio_chunk_1, "test_call_multi")
            result_2 = await service.transcribe_audio_chunk(audio_chunk_2, "test_call_multi")

            # First chunk should be English
            assert result_1.detected_language == 'en'
            assert result_1.text == 'Hello, can you help me?'

            # Second chunk should be Spanish
            assert result_2.detected_language == 'es'
            assert result_2.text == 'Sí, por supuesto.'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
