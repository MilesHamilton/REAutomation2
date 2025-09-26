"""Tests for local TTS service"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pipecat.frames.frames import TTSAudioRawFrame

from src.voice.local_tts_service import LocalTTSService, TTSEngine
from src.voice.models import TTSProvider


class TestLocalTTSService:
    """Test LocalTTSService functionality"""

    @pytest.fixture
    def tts_config(self):
        """TTS service configuration"""
        return {
            "voice_id": "test_voice",
            "sample_rate": 16000,
            "channels": 1
        }

    @pytest.fixture
    def local_tts_service(self, tts_config):
        """Create local TTS service for testing"""
        return LocalTTSService(**tts_config)

    def test_tts_service_initialization(self, local_tts_service):
        """Test TTS service initialization"""
        assert local_tts_service._voice_id == "test_voice"
        assert local_tts_service._sample_rate == 16000
        assert local_tts_service._channels == 1
        assert local_tts_service._engine is None

    @pytest.mark.asyncio
    async def test_initialize_service(self, local_tts_service):
        """Test service initialization"""
        with patch('src.voice.local_tts_service.TTSEngine') as mock_engine:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock()
            mock_engine.return_value = mock_instance

            await local_tts_service.initialize()

            assert local_tts_service._engine is not None
            mock_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_text_success(self, local_tts_service, mock_audio_data):
        """Test successful text synthesis"""
        # Mock engine
        mock_engine = AsyncMock()
        mock_engine.synthesize.return_value = mock_audio_data["wav_audio"]
        local_tts_service._engine = mock_engine

        result = await local_tts_service.synthesize("Hello, world!")

        assert result is not None
        mock_engine.synthesize.assert_called_once_with("Hello, world!", "test_voice")

    @pytest.mark.asyncio
    async def test_synthesize_text_failure(self, local_tts_service):
        """Test text synthesis failure handling"""
        # Mock engine that fails
        mock_engine = AsyncMock()
        mock_engine.synthesize.side_effect = Exception("TTS engine error")
        local_tts_service._engine = mock_engine

        result = await local_tts_service.synthesize("Hello, world!")

        # Should return None on failure
        assert result is None

    @pytest.mark.asyncio
    async def test_process_text_to_audio_frames(self, local_tts_service, mock_audio_data):
        """Test processing text to audio frames"""
        # Mock successful synthesis
        mock_engine = AsyncMock()
        mock_engine.synthesize.return_value = mock_audio_data["wav_audio"]
        local_tts_service._engine = mock_engine

        frames = []
        async for frame in local_tts_service._process_text_to_audio_frames("Test message"):
            frames.append(frame)

        # Should produce audio frames
        assert len(frames) > 0
        assert all(isinstance(frame, TTSAudioRawFrame) for frame in frames)

    @pytest.mark.asyncio
    async def test_cleanup_service(self, local_tts_service):
        """Test service cleanup"""
        mock_engine = MagicMock()
        mock_engine.cleanup = AsyncMock()
        local_tts_service._engine = mock_engine

        await local_tts_service.cleanup()

        mock_engine.cleanup.assert_called_once()
        assert local_tts_service._engine is None

    def test_get_service_metrics(self, local_tts_service):
        """Test getting service metrics"""
        # Set some metrics
        local_tts_service._total_characters_processed = 1500
        local_tts_service._total_synthesis_time = 5.2
        local_tts_service._synthesis_count = 12

        metrics = local_tts_service.get_metrics()

        assert "total_characters_processed" in metrics
        assert "total_synthesis_time" in metrics
        assert "synthesis_count" in metrics
        assert "avg_synthesis_time" in metrics
        assert metrics["total_characters_processed"] == 1500

    @pytest.mark.asyncio
    async def test_set_voice(self, local_tts_service):
        """Test changing voice"""
        await local_tts_service.set_voice("new_voice_id")

        assert local_tts_service._voice_id == "new_voice_id"

    @pytest.mark.asyncio
    async def test_get_available_voices(self, local_tts_service):
        """Test getting available voices"""
        mock_engine = MagicMock()
        mock_engine.get_available_voices.return_value = ["voice1", "voice2", "voice3"]
        local_tts_service._engine = mock_engine

        voices = await local_tts_service.get_available_voices()

        assert len(voices) == 3
        assert "voice1" in voices


class TestTTSEngine:
    """Test TTSEngine functionality"""

    @pytest.fixture
    def tts_engine(self):
        """Create TTS engine for testing"""
        return TTSEngine()

    def test_engine_initialization(self, tts_engine):
        """Test engine initialization"""
        assert tts_engine._is_initialized is False
        assert tts_engine._model is None

    @pytest.mark.asyncio
    async def test_initialize_piper_engine(self, tts_engine):
        """Test Piper engine initialization"""
        with patch('src.voice.local_tts_service.PiperTTS') as mock_piper:
            mock_instance = MagicMock()
            mock_instance.load_model = AsyncMock()
            mock_piper.return_value = mock_instance

            await tts_engine.initialize(engine_type="piper")

            assert tts_engine._engine_type == "piper"
            assert tts_engine._model is not None
            mock_instance.load_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_coqui_engine(self, tts_engine):
        """Test Coqui engine initialization"""
        with patch('src.voice.local_tts_service.CoquiTTS') as mock_coqui:
            mock_instance = MagicMock()
            mock_instance.load_model = AsyncMock()
            mock_coqui.return_value = mock_instance

            await tts_engine.initialize(engine_type="coqui")

            assert tts_engine._engine_type == "coqui"
            assert tts_engine._model is not None

    @pytest.mark.asyncio
    async def test_initialize_mock_engine(self, tts_engine):
        """Test mock engine initialization"""
        await tts_engine.initialize(engine_type="mock")

        assert tts_engine._engine_type == "mock"
        assert tts_engine._is_initialized is True

    @pytest.mark.asyncio
    async def test_synthesize_with_piper(self, tts_engine, mock_audio_data):
        """Test synthesis with Piper engine"""
        # Initialize with Piper
        with patch('src.voice.local_tts_service.PiperTTS') as mock_piper:
            mock_instance = MagicMock()
            mock_instance.load_model = AsyncMock()
            mock_instance.synthesize = AsyncMock(return_value=mock_audio_data["wav_audio"])
            mock_piper.return_value = mock_instance
            tts_engine._model = mock_instance

            await tts_engine.initialize(engine_type="piper")

            audio_data = await tts_engine.synthesize("Test text", "default_voice")

            assert audio_data is not None
            mock_instance.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_with_mock(self, tts_engine):
        """Test synthesis with mock engine"""
        await tts_engine.initialize(engine_type="mock")

        audio_data = await tts_engine.synthesize("Test text", "mock_voice")

        # Mock engine should return synthetic audio
        assert audio_data is not None
        assert len(audio_data) > 0

    @pytest.mark.asyncio
    async def test_synthesize_not_initialized(self, tts_engine):
        """Test synthesis when not initialized"""
        with pytest.raises(RuntimeError, match="TTS engine not initialized"):
            await tts_engine.synthesize("Test text", "voice")

    def test_get_available_voices_piper(self, tts_engine):
        """Test getting available voices for Piper"""
        tts_engine._engine_type = "piper"
        voices = tts_engine.get_available_voices()

        assert len(voices) > 0
        assert "en_US-lessac-medium" in voices

    def test_get_available_voices_coqui(self, tts_engine):
        """Test getting available voices for Coqui"""
        tts_engine._engine_type = "coqui"
        voices = tts_engine.get_available_voices()

        assert len(voices) > 0

    def test_get_available_voices_mock(self, tts_engine):
        """Test getting available voices for mock engine"""
        tts_engine._engine_type = "mock"
        voices = tts_engine.get_available_voices()

        assert voices == ["mock_voice_1", "mock_voice_2", "mock_voice_3"]

    @pytest.mark.asyncio
    async def test_cleanup_engine(self, tts_engine):
        """Test engine cleanup"""
        # Initialize first
        await tts_engine.initialize(engine_type="mock")

        await tts_engine.cleanup()

        assert tts_engine._is_initialized is False
        assert tts_engine._model is None

    def test_is_engine_ready(self, tts_engine):
        """Test engine readiness check"""
        # Not ready initially
        assert tts_engine.is_ready() is False

    @pytest.mark.asyncio
    async def test_engine_ready_after_init(self, tts_engine):
        """Test engine readiness after initialization"""
        await tts_engine.initialize(engine_type="mock")

        assert tts_engine.is_ready() is True

    @pytest.mark.asyncio
    async def test_validate_voice_id(self, tts_engine):
        """Test voice ID validation"""
        await tts_engine.initialize(engine_type="mock")

        # Valid voice
        is_valid = tts_engine.validate_voice_id("mock_voice_1")
        assert is_valid is True

        # Invalid voice
        is_valid = tts_engine.validate_voice_id("nonexistent_voice")
        assert is_valid is False

    def test_get_engine_info(self, tts_engine):
        """Test getting engine information"""
        tts_engine._engine_type = "piper"
        tts_engine._is_initialized = True

        info = tts_engine.get_engine_info()

        assert info["engine_type"] == "piper"
        assert info["is_initialized"] is True
        assert "available_voices" in info