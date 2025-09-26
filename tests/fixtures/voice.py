"""Voice pipeline test fixtures and utilities"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

from src.voice.models import (
    CallSession, VoiceCallState, TTSProvider, TTSConfig,
    VoiceMetrics, AudioChunk, STTResult, TTSResponse
)
from src.voice.pipeline import VoicePipeline
from src.voice.pipecat_integration import PipecatVoicePipeline
from src.voice.local_tts_service import LocalTTSService


@pytest.fixture
def mock_audio_data():
    """Generate mock audio data"""
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)

    # Generate a simple sine wave
    t = np.linspace(0, duration, samples)
    frequency = 440  # A4 note
    audio = (np.sin(2 * np.pi * frequency * t) * 0.1 * 32767).astype(np.int16)

    return audio.tobytes()


@pytest.fixture
def sample_audio_chunk(mock_audio_data):
    """Create a sample audio chunk"""
    return AudioChunk(
        data=mock_audio_data,
        timestamp=datetime.now().timestamp(),
        chunk_id=1,
        sample_rate=16000,
        channels=1,
        is_speech=True
    )


@pytest.fixture
def sample_stt_result():
    """Create a sample STT result"""
    return STTResult(
        text="Hello, this is a test transcription.",
        confidence=0.95,
        processing_time_ms=150.0,
        language="en",
        segments=[
            {
                "start": 0.0,
                "end": 2.5,
                "text": "Hello, this is a test transcription.",
                "confidence": 0.95
            }
        ]
    )


@pytest.fixture
def sample_tts_response(mock_audio_data):
    """Create a sample TTS response"""
    return TTSResponse(
        audio_data=mock_audio_data,
        sample_rate=16000,
        channels=1,
        duration_ms=1000,
        generation_time_ms=50.0,
        cost=0.002,
        provider="local_piper",
        voice_id="en_US-lessac-medium"
    )


@pytest.fixture
def sample_call_session():
    """Create a sample call session"""
    tts_config = TTSConfig(provider=TTSProvider.LOCAL_PIPER)
    metrics = VoiceMetrics(call_id="test_call_123")

    return CallSession(
        call_id="test_call_123",
        phone_number="+1234567890",
        current_tier=TTSProvider.LOCAL_PIPER,
        tts_config=tts_config,
        metrics=metrics,
        lead_data={"name": "Test Lead", "company": "Test Corp"},
        state=VoiceCallState.CONNECTED
    )


@pytest.fixture
def mock_twilio_client():
    """Mock Twilio client"""
    mock_client = MagicMock()

    # Mock account
    mock_account = MagicMock()
    mock_account.sid = "ACtest123"
    mock_account.friendly_name = "Test Account"
    mock_client.api.account.fetch.return_value = mock_account

    # Mock call creation
    mock_call = MagicMock()
    mock_call.sid = "CAtest123"
    mock_client.calls.create.return_value = mock_call

    return mock_client


@pytest.fixture
async def mock_voice_pipeline():
    """Create a mock voice pipeline"""
    with patch('src.voice.pipeline.VoicePipeline') as mock_pipeline_class:
        pipeline = AsyncMock(spec=VoicePipeline)
        pipeline.is_initialized = True
        pipeline.use_pipecat = False
        pipeline.active_calls = {}

        # Mock methods
        pipeline.initialize.return_value = True
        pipeline.start_call.return_value = True
        pipeline.end_call.return_value = True
        pipeline.switch_tier.return_value = True
        pipeline.get_call_session.return_value = None
        pipeline.health_check.return_value = {"status": "healthy"}

        mock_pipeline_class.return_value = pipeline
        yield pipeline


@pytest.fixture
async def mock_pipecat_pipeline():
    """Create a mock Pipecat pipeline"""
    with patch('src.voice.pipecat_integration.PipecatVoicePipeline') as mock_pipeline_class:
        pipeline = AsyncMock(spec=PipecatVoicePipeline)
        pipeline.is_initialized = True
        pipeline.call_sessions = {}
        pipeline.active_pipelines = {}
        pipeline.active_tasks = {}

        # Mock methods
        pipeline.initialize.return_value = True
        pipeline.start_call.return_value = True
        pipeline.end_call.return_value = True
        pipeline.switch_tier.return_value = True
        pipeline.get_call_session.return_value = None
        pipeline.health_check.return_value = {"status": "healthy"}

        mock_pipeline_class.return_value = pipeline
        yield pipeline


@pytest.fixture
def mock_local_tts_service():
    """Mock local TTS service"""
    with patch('src.voice.local_tts_service.LocalTTSService') as mock_service_class:
        service = AsyncMock(spec=LocalTTSService)
        service._is_initialized = True
        service.engine = "piper"

        # Mock methods
        service.start.return_value = None
        service.stop.return_value = None
        service.health_check.return_value = {"status": "healthy"}

        mock_service_class.return_value = service
        yield service


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()

    return websocket


@pytest.fixture
def voice_test_data():
    """Common voice test data"""
    return {
        "call_id": "test_call_123",
        "phone_number": "+1234567890",
        "twilio_stream_sid": "MZtest123",
        "lead_data": {
            "name": "Test Lead",
            "company": "Test Corp",
            "email": "test@example.com"
        },
        "conversation_messages": [
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "I'm interested in your services."},
            {"role": "assistant", "content": "Great! Let me tell you about what we offer."}
        ],
        "qualification_score": 0.85,
        "conversation_length": 3
    }


@pytest.fixture
def tts_test_scenarios():
    """Different TTS testing scenarios"""
    return [
        {
            "name": "short_message",
            "text": "Hello",
            "expected_duration_range": (0.5, 1.5),  # seconds
            "expected_cost_range": (0.0, 0.001)
        },
        {
            "name": "medium_message",
            "text": "Hello, this is a test message for TTS synthesis.",
            "expected_duration_range": (2.0, 4.0),
            "expected_cost_range": (0.001, 0.005)
        },
        {
            "name": "long_message",
            "text": "This is a much longer message that will test the TTS system's ability to handle extended text synthesis. It should take longer to process and cost more than shorter messages.",
            "expected_duration_range": (8.0, 15.0),
            "expected_cost_range": (0.005, 0.015)
        }
    ]


@pytest.fixture
def stt_test_scenarios(mock_audio_data):
    """Different STT testing scenarios"""
    return [
        {
            "name": "clear_speech",
            "audio_data": mock_audio_data,
            "expected_confidence_min": 0.8,
            "expected_text_contains": ["hello", "test"]
        },
        {
            "name": "noisy_audio",
            "audio_data": mock_audio_data,  # Would be modified with noise in real test
            "expected_confidence_min": 0.6,
            "expected_text_contains": []
        },
        {
            "name": "silent_audio",
            "audio_data": b'\x00' * 1000,  # Silent audio
            "expected_confidence_min": 0.0,
            "expected_text_contains": []
        }
    ]


@pytest.fixture
def tier_switch_scenarios():
    """Different tier switching scenarios"""
    return [
        {
            "name": "qualification_escalation",
            "current_tier": TTSProvider.LOCAL_PIPER,
            "qualification_score": 0.85,
            "budget_utilization": 0.3,
            "conversation_length": 5,
            "expected_switch": True,
            "expected_target": TTSProvider.ELEVENLABS,
            "expected_reason": "high_qualification_score"
        },
        {
            "name": "budget_constraint",
            "current_tier": TTSProvider.LOCAL_PIPER,
            "qualification_score": 0.9,
            "budget_utilization": 0.95,
            "conversation_length": 3,
            "expected_switch": False,
            "expected_target": TTSProvider.LOCAL_PIPER,
            "expected_reason": "insufficient_escalation_factors"
        },
        {
            "name": "downgrade_scenario",
            "current_tier": TTSProvider.ELEVENLABS,
            "qualification_score": 0.2,
            "budget_utilization": 0.9,
            "conversation_length": 8,
            "expected_switch": True,
            "expected_target": TTSProvider.LOCAL_PIPER,
            "expected_reason": "budget_pressure_low_qualification"
        }
    ]