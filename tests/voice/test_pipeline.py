import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import time

from src.voice.pipeline import VoicePipeline
from src.voice.models import TTSProvider, VoiceCallState, AudioConfig


class TestVoicePipeline:
    """Test suite for VoicePipeline"""

    @pytest.fixture
    async def pipeline(self):
        """Create a voice pipeline for testing"""
        pipeline = VoicePipeline()
        # Mock the initialization to avoid external dependencies
        with patch.object(pipeline.tts_manager, 'initialize', return_value=True), \
             patch.object(pipeline.stt_service, 'initialize', return_value=True), \
             patch.object(pipeline.twilio_integration, 'initialize', return_value=True):
            await pipeline.initialize()
        return pipeline

    @pytest.fixture
    def mock_lead_data(self):
        """Sample lead data for testing"""
        return {
            "name": "John Doe",
            "company": "Test Corp",
            "phone": "+1234567890",
            "industry": "Technology"
        }

    @pytest.mark.asyncio
    async def test_initialize_pipeline(self):
        """Test pipeline initialization"""
        pipeline = VoicePipeline()

        with patch.object(pipeline.tts_manager, 'initialize', return_value=True) as mock_tts, \
             patch.object(pipeline.stt_service, 'initialize', return_value=True) as mock_stt, \
             patch.object(pipeline.twilio_integration, 'initialize', return_value=True) as mock_twilio:

            result = await pipeline.initialize()

            assert result is True
            assert pipeline.is_initialized is True
            mock_tts.assert_called_once()
            mock_stt.assert_called_once()
            mock_twilio.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_pipeline_failure(self):
        """Test pipeline initialization failure"""
        pipeline = VoicePipeline()

        with patch.object(pipeline.tts_manager, 'initialize', return_value=False):
            result = await pipeline.initialize()

            assert result is False
            assert pipeline.is_initialized is False

    @pytest.mark.asyncio
    async def test_start_call_success(self, pipeline, mock_lead_data):
        """Test successful call start"""
        call_id = "test-call-123"
        phone_number = "+1234567890"

        with patch.object(pipeline.twilio_integration, 'start_call', return_value=True):
            result = await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data,
                initial_tier=TTSProvider.LOCAL_PIPER
            )

            assert result is True
            assert call_id in pipeline.active_calls

            session = pipeline.active_calls[call_id]
            assert session.call_id == call_id
            assert session.phone_number == phone_number
            assert session.current_tier == TTSProvider.LOCAL_PIPER
            assert session.state == VoiceCallState.RINGING

    @pytest.mark.asyncio
    async def test_start_call_failure(self, pipeline, mock_lead_data):
        """Test call start failure"""
        call_id = "test-call-456"
        phone_number = "+1234567890"

        with patch.object(pipeline.twilio_integration, 'start_call', return_value=False):
            result = await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data,
                initial_tier=TTSProvider.LOCAL_PIPER
            )

            assert result is False
            assert call_id not in pipeline.active_calls

    @pytest.mark.asyncio
    async def test_start_call_duplicate(self, pipeline, mock_lead_data):
        """Test starting duplicate call"""
        call_id = "test-call-789"
        phone_number = "+1234567890"

        with patch.object(pipeline.twilio_integration, 'start_call', return_value=True):
            # Start first call
            result1 = await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data
            )
            assert result1 is True

            # Try to start duplicate call
            result2 = await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data
            )
            assert result2 is False

    @pytest.mark.asyncio
    async def test_switch_tier(self, pipeline, mock_lead_data):
        """Test TTS tier switching"""
        call_id = "test-call-switch"
        phone_number = "+1234567890"

        with patch.object(pipeline.twilio_integration, 'start_call', return_value=True), \
             patch.object(pipeline.tts_manager, 'switch_tier', return_value=True):

            # Start call with local tier
            await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data,
                initial_tier=TTSProvider.LOCAL_PIPER
            )

            # Switch to premium tier
            result = await pipeline.switch_tier(
                call_id=call_id,
                new_tier=TTSProvider.ELEVENLABS,
                trigger="qualification"
            )

            assert result is True
            session = pipeline.active_calls[call_id]
            assert session.current_tier == TTSProvider.ELEVENLABS
            assert session.metrics.tier_switches == 1

    @pytest.mark.asyncio
    async def test_switch_tier_nonexistent_call(self, pipeline):
        """Test tier switching for nonexistent call"""
        result = await pipeline.switch_tier(
            call_id="nonexistent-call",
            new_tier=TTSProvider.ELEVENLABS
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_end_call(self, pipeline, mock_lead_data):
        """Test ending a call"""
        call_id = "test-call-end"
        phone_number = "+1234567890"

        with patch.object(pipeline.twilio_integration, 'start_call', return_value=True), \
             patch.object(pipeline.twilio_integration, 'end_call', return_value=True):

            # Start call
            await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data
            )

            assert call_id in pipeline.active_calls

            # End call
            result = await pipeline.end_call(call_id, "completed")

            assert result is True
            assert call_id not in pipeline.active_calls

    @pytest.mark.asyncio
    async def test_end_nonexistent_call(self, pipeline):
        """Test ending nonexistent call"""
        result = await pipeline.end_call("nonexistent-call", "completed")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_call_session(self, pipeline, mock_lead_data):
        """Test getting call session"""
        call_id = "test-call-session"
        phone_number = "+1234567890"

        with patch.object(pipeline.twilio_integration, 'start_call', return_value=True):
            await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data
            )

            session = pipeline.get_call_session(call_id)
            assert session is not None
            assert session.call_id == call_id

            # Test nonexistent session
            assert pipeline.get_call_session("nonexistent") is None

    @pytest.mark.asyncio
    async def test_health_check(self, pipeline):
        """Test pipeline health check"""
        with patch.object(pipeline.tts_manager, 'health_check', return_value={"status": "healthy"}) as mock_tts_health, \
             patch.object(pipeline.stt_service, 'health_check', return_value={"status": "healthy"}) as mock_stt_health, \
             patch.object(pipeline.twilio_integration, 'health_check', return_value={"status": "healthy"}) as mock_twilio_health:

            health = await pipeline.health_check()

            assert health["status"] == "healthy"
            assert health["initialized"] is True
            assert health["active_calls"] == 0

            mock_tts_health.assert_called_once()
            mock_stt_health.assert_called_once()
            mock_twilio_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """Test health check when not initialized"""
        pipeline = VoicePipeline()

        health = await pipeline.health_check()

        assert health["status"] == "unhealthy"
        assert "not initialized" in health["error"].lower()

    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline, mock_lead_data):
        """Test pipeline cleanup"""
        call_id = "test-call-cleanup"
        phone_number = "+1234567890"

        with patch.object(pipeline.twilio_integration, 'start_call', return_value=True), \
             patch.object(pipeline.twilio_integration, 'end_call', return_value=True), \
             patch.object(pipeline.tts_manager, 'cleanup') as mock_tts_cleanup, \
             patch.object(pipeline.stt_service, 'cleanup') as mock_stt_cleanup, \
             patch.object(pipeline.twilio_integration, 'cleanup') as mock_twilio_cleanup:

            # Start a call
            await pipeline.start_call(
                call_id=call_id,
                phone_number=phone_number,
                lead_data=mock_lead_data
            )

            assert len(pipeline.active_calls) == 1

            # Cleanup
            await pipeline.cleanup()

            assert len(pipeline.active_calls) == 0
            assert pipeline.is_initialized is False

            mock_tts_cleanup.assert_called_once()
            mock_stt_cleanup.assert_called_once()
            mock_twilio_cleanup.assert_called_once()

    def test_callback_setters(self, pipeline):
        """Test callback setter methods"""
        mock_callback = Mock()

        # Test setting callbacks
        pipeline.on_call_started(mock_callback)
        pipeline.on_call_ended(mock_callback)
        pipeline.on_tier_switched(mock_callback)
        pipeline.on_transcript(mock_callback)

        assert pipeline._on_call_started == mock_callback
        assert pipeline._on_call_ended == mock_callback
        assert pipeline._on_tier_switched == mock_callback
        assert pipeline._on_transcript == mock_callback

    @pytest.mark.asyncio
    async def test_pipeline_not_initialized_start_call(self):
        """Test starting call when pipeline not initialized"""
        pipeline = VoicePipeline()

        result = await pipeline.start_call(
            call_id="test-call",
            phone_number="+1234567890"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_active_calls(self, pipeline, mock_lead_data):
        """Test getting active calls"""
        with patch.object(pipeline.twilio_integration, 'start_call', return_value=True):
            # Start multiple calls
            await pipeline.start_call("call-1", "+1111111111", mock_lead_data)
            await pipeline.start_call("call-2", "+2222222222", mock_lead_data)

            active_calls = pipeline.get_active_calls()

            assert len(active_calls) == 2
            assert "call-1" in active_calls
            assert "call-2" in active_calls


@pytest.mark.performance
class TestVoicePipelinePerformance:
    """Performance tests for VoicePipeline"""

    @pytest.mark.asyncio
    async def test_concurrent_call_startup(self):
        """Test starting multiple calls concurrently"""
        pipeline = VoicePipeline()

        with patch.object(pipeline.tts_manager, 'initialize', return_value=True), \
             patch.object(pipeline.stt_service, 'initialize', return_value=True), \
             patch.object(pipeline.twilio_integration, 'initialize', return_value=True), \
             patch.object(pipeline.twilio_integration, 'start_call', return_value=True):

            await pipeline.initialize()

            # Test concurrent call startup
            start_time = time.time()

            tasks = []
            for i in range(5):
                task = pipeline.start_call(
                    call_id=f"concurrent-call-{i}",
                    phone_number=f"+123456789{i}",
                    lead_data={"name": f"Test {i}"}
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            end_time = time.time()
            duration = end_time - start_time

            # All calls should succeed
            assert all(results)
            assert len(pipeline.active_calls) == 5

            # Should complete within reasonable time (adjust threshold as needed)
            assert duration < 2.0  # 2 seconds for 5 concurrent calls

            await pipeline.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])