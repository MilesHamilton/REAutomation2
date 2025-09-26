"""Tests for Pipecat integration"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pipecat.frames.frames import AudioRawFrame, TextFrame, TTSStartedFrame, TTSStoppedFrame

from src.voice.pipecat_integration import ConversationProcessor, PipecatVoicePipeline
from src.voice.models import VoicePipelineConfig, AudioConfig, TTSProvider


class TestConversationProcessor:
    """Test ConversationProcessor functionality"""

    @pytest.fixture
    def processor_config(self):
        """Configuration for conversation processor"""
        return {
            "llm_service": AsyncMock(),
            "cost_calculator": MagicMock(),
            "call_id": "test_call_123"
        }

    @pytest.fixture
    def conversation_processor(self, processor_config):
        """Create conversation processor for testing"""
        return ConversationProcessor(**processor_config)

    @pytest.mark.asyncio
    async def test_process_frame_text(self, conversation_processor):
        """Test processing text frame"""
        # Mock LLM response
        conversation_processor._llm_service.generate_response.return_value.content = "Hello! How can I help you?"

        text_frame = TextFrame("Hello there")

        # Process the frame
        await conversation_processor.process_frame(text_frame, None)

        # Verify LLM was called
        conversation_processor._llm_service.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_frame_audio(self, conversation_processor, mock_audio_data):
        """Test processing audio frame"""
        audio_frame = AudioRawFrame(
            audio=mock_audio_data["raw_audio"],
            sample_rate=16000,
            num_channels=1
        )

        # Process the frame (should queue for STT)
        await conversation_processor.process_frame(audio_frame, None)

        # Verify audio was queued
        assert len(conversation_processor._audio_buffer) > 0

    @pytest.mark.asyncio
    async def test_handle_conversation_turn(self, conversation_processor):
        """Test handling a complete conversation turn"""
        user_input = "I'm interested in buying a house"

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "That's great! What type of property are you looking for?"
        mock_response.usage_tokens = 25
        conversation_processor._llm_service.generate_response.return_value = mock_response

        response = await conversation_processor._handle_conversation_turn(user_input)

        assert response == "That's great! What type of property are you looking for?"
        conversation_processor._cost_calculator.calculate_llm_cost.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_llm_response(self, conversation_processor):
        """Test LLM response generation"""
        user_message = "What's the market like?"

        mock_response = MagicMock()
        mock_response.content = "The market is quite active right now."
        conversation_processor._llm_service.generate_response.return_value = mock_response

        response = await conversation_processor._generate_llm_response(user_message)

        assert response == "The market is quite active right now."

    def test_add_conversation_message(self, conversation_processor):
        """Test adding message to conversation history"""
        conversation_processor._add_conversation_message("user", "Hello")
        conversation_processor._add_conversation_message("assistant", "Hi there!")

        assert len(conversation_processor._conversation_history) == 2
        assert conversation_processor._conversation_history[0]["role"] == "user"
        assert conversation_processor._conversation_history[1]["role"] == "assistant"


class TestPipecatVoicePipeline:
    """Test PipecatVoicePipeline functionality"""

    @pytest.fixture
    def pipeline_config(self):
        """Configuration for voice pipeline"""
        return VoicePipelineConfig(
            audio_config=AudioConfig(
                sample_rate=16000,
                channels=1,
                chunk_size=1024
            ),
            tts_provider=TTSProvider.LOCAL_PIPER,
            stt_provider="whisper-local",
            llm_model="llama3.1:8b"
        )

    @pytest.fixture
    def voice_pipeline(self, pipeline_config):
        """Create voice pipeline for testing"""
        pipeline = PipecatVoicePipeline(pipeline_config)
        return pipeline

    @pytest.mark.asyncio
    async def test_initialize_pipeline(self, voice_pipeline):
        """Test pipeline initialization"""
        with patch('src.voice.pipecat_integration.Pipeline') as mock_pipeline:
            mock_pipeline.return_value = MagicMock()

            await voice_pipeline.initialize()

            assert voice_pipeline._pipeline is not None
            assert voice_pipeline._is_initialized is True

    @pytest.mark.asyncio
    async def test_start_call(self, voice_pipeline, mock_websocket):
        """Test starting a call"""
        # Initialize first
        with patch('src.voice.pipecat_integration.Pipeline'):
            await voice_pipeline.initialize()

        # Mock pipeline start
        voice_pipeline._pipeline.start = AsyncMock()

        await voice_pipeline.start_call("test_call_123", mock_websocket)

        assert voice_pipeline._current_call_id == "test_call_123"
        voice_pipeline._pipeline.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_call(self, voice_pipeline):
        """Test stopping a call"""
        # Set up active call
        voice_pipeline._current_call_id = "test_call_123"
        voice_pipeline._pipeline = MagicMock()
        voice_pipeline._pipeline.stop = AsyncMock()

        await voice_pipeline.stop_call()

        assert voice_pipeline._current_call_id is None
        voice_pipeline._pipeline.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_audio_data(self, voice_pipeline, mock_audio_data):
        """Test sending audio data to pipeline"""
        voice_pipeline._pipeline = MagicMock()
        voice_pipeline._pipeline.push_frame = AsyncMock()

        await voice_pipeline.send_audio_data(mock_audio_data["raw_audio"])

        voice_pipeline._pipeline.push_frame.assert_called_once()

    def test_create_tts_service_local_piper(self, voice_pipeline):
        """Test creating local Piper TTS service"""
        voice_pipeline._config.tts_provider = TTSProvider.LOCAL_PIPER

        tts_service = voice_pipeline._create_tts_service()

        assert tts_service is not None
        # Should be LocalTTSService instance

    def test_create_tts_service_elevenlabs(self, voice_pipeline):
        """Test creating ElevenLabs TTS service"""
        voice_pipeline._config.tts_provider = TTSProvider.ELEVENLABS

        with patch.dict('os.environ', {'ELEVENLABS_API_KEY': 'test_key'}):
            tts_service = voice_pipeline._create_tts_service()
            assert tts_service is not None

    def test_create_stt_service_local(self, voice_pipeline):
        """Test creating local STT service"""
        voice_pipeline._config.stt_provider = "whisper-local"

        stt_service = voice_pipeline._create_stt_service()

        assert stt_service is not None

    def test_create_llm_service(self, voice_pipeline):
        """Test creating LLM service"""
        llm_service = voice_pipeline._create_llm_service()

        assert llm_service is not None

    @pytest.mark.asyncio
    async def test_handle_pipeline_error(self, voice_pipeline):
        """Test pipeline error handling"""
        error = Exception("Test pipeline error")

        # Mock error handler
        voice_pipeline._error_callback = AsyncMock()

        await voice_pipeline._handle_pipeline_error(error)

        voice_pipeline._error_callback.assert_called_once_with(error)

    @pytest.mark.asyncio
    async def test_cleanup_resources(self, voice_pipeline):
        """Test resource cleanup"""
        # Set up resources
        voice_pipeline._pipeline = MagicMock()
        voice_pipeline._pipeline.cleanup = AsyncMock()
        voice_pipeline._current_call_id = "test_call"

        await voice_pipeline.cleanup()

        voice_pipeline._pipeline.cleanup.assert_called_once()
        assert voice_pipeline._current_call_id is None
        assert voice_pipeline._is_initialized is False

    def test_get_pipeline_metrics(self, voice_pipeline):
        """Test getting pipeline metrics"""
        # Mock some metrics
        voice_pipeline._total_calls = 5
        voice_pipeline._total_audio_processed_seconds = 150.0
        voice_pipeline._total_errors = 1

        metrics = voice_pipeline.get_pipeline_metrics()

        assert "total_calls" in metrics
        assert "total_audio_processed_seconds" in metrics
        assert "total_errors" in metrics
        assert metrics["total_calls"] == 5

    @pytest.mark.asyncio
    async def test_switch_tts_provider(self, voice_pipeline):
        """Test switching TTS provider during call"""
        # Initialize pipeline
        with patch('src.voice.pipecat_integration.Pipeline'):
            await voice_pipeline.initialize()

        # Switch provider
        new_provider = TTSProvider.ELEVENLABS
        await voice_pipeline.switch_tts_provider(new_provider)

        assert voice_pipeline._config.tts_provider == new_provider

    @pytest.mark.asyncio
    async def test_get_call_audio_stats(self, voice_pipeline):
        """Test getting call audio statistics"""
        voice_pipeline._current_call_id = "test_call_123"
        voice_pipeline._call_audio_stats = {
            "test_call_123": {
                "audio_processed_seconds": 45.5,
                "frames_processed": 2275,
                "avg_processing_time_ms": 12.3
            }
        }

        stats = voice_pipeline.get_call_audio_stats("test_call_123")

        assert stats["audio_processed_seconds"] == 45.5
        assert stats["frames_processed"] == 2275

    def test_is_call_active(self, voice_pipeline):
        """Test checking if call is active"""
        # No active call
        assert voice_pipeline.is_call_active() is False

        # Set active call
        voice_pipeline._current_call_id = "test_call_123"
        assert voice_pipeline.is_call_active() is True

    @pytest.mark.asyncio
    async def test_mute_unmute_audio(self, voice_pipeline):
        """Test muting and unmuting audio"""
        voice_pipeline._pipeline = MagicMock()
        voice_pipeline._pipeline.mute_audio = AsyncMock()
        voice_pipeline._pipeline.unmute_audio = AsyncMock()

        # Mute
        await voice_pipeline.mute_audio()
        voice_pipeline._pipeline.mute_audio.assert_called_once()

        # Unmute
        await voice_pipeline.unmute_audio()
        voice_pipeline._pipeline.unmute_audio.assert_called_once()

    @pytest.mark.asyncio
    async def test_adjust_audio_volume(self, voice_pipeline):
        """Test adjusting audio volume"""
        voice_pipeline._pipeline = MagicMock()
        voice_pipeline._pipeline.set_volume = AsyncMock()

        await voice_pipeline.adjust_audio_volume(0.8)

        voice_pipeline._pipeline.set_volume.assert_called_once_with(0.8)