from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class TTSProvider(str, Enum):
    LOCAL_PIPER = "piper"
    LOCAL_COQUI = "coqui"
    ELEVENLABS = "elevenlabs"


class VoiceCallState(str, Enum):
    IDLE = "idle"
    RINGING = "ringing"
    CONNECTED = "connected"
    SPEAKING = "speaking"
    LISTENING = "listening"
    PROCESSING = "processing"
    TIER_SWITCHING = "tier_switching"
    ENDED = "ended"
    ERROR = "error"


class AudioConfig(BaseModel):
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)
    chunk_size: int = Field(default=1024)
    format: str = Field(default="PCM")
    bit_depth: int = Field(default=16)


class TTSConfig(BaseModel):
    provider: TTSProvider
    voice_id: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.5, le=2.0)
    stability: float = Field(default=0.75, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0)
    model_id: Optional[str] = None
    optimize_streaming_latency: int = Field(default=2)


class STTConfig(BaseModel):
    model: str = Field(default="whisper-small")
    language: str = Field(default="en")
    vad_enabled: bool = Field(default=True)
    silence_threshold: float = Field(default=0.5)
    max_silence_duration: float = Field(default=2.0)


class VoiceMetrics(BaseModel):
    call_id: str
    audio_latency_ms: float = 0
    tts_latency_ms: float = 0
    stt_latency_ms: float = 0
    total_processing_latency_ms: float = 0
    audio_quality_score: Optional[float] = None
    speech_clarity_score: Optional[float] = None
    tier_switches: int = 0
    total_audio_duration_ms: float = 0
    cost: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class AudioChunk(BaseModel):
    data: bytes
    timestamp: float
    chunk_id: int
    sample_rate: int
    channels: int
    is_speech: bool = False
    silence_duration: float = 0.0


class TTSRequest(BaseModel):
    text: str
    call_id: str
    provider: TTSProvider
    config: TTSConfig
    priority: int = Field(default=1)
    chunk_id: Optional[int] = None


class TTSResponse(BaseModel):
    audio_data: bytes
    call_id: str
    chunk_id: Optional[int] = None
    generation_time_ms: float
    audio_duration_ms: float
    provider: TTSProvider
    cost: float = 0.0
    quality_score: Optional[float] = None


class STTResult(BaseModel):
    text: str
    call_id: str
    confidence: float
    processing_time_ms: float
    is_final: bool = True
    language: str = "en"
    chunk_id: Optional[int] = None


class CallSession(BaseModel):
    call_id: str
    phone_number: str
    state: VoiceCallState = VoiceCallState.IDLE
    current_tier: TTSProvider = TTSProvider.LOCAL_PIPER
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    tts_config: TTSConfig
    stt_config: STTConfig = Field(default_factory=STTConfig)
    metrics: VoiceMetrics
    conversation_id: Optional[str] = None
    lead_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error_message: Optional[str] = None


class TierSwitchEvent(BaseModel):
    call_id: str
    from_tier: TTSProvider
    to_tier: TTSProvider
    trigger: str  # "qualification", "manual", "fallback"
    timestamp: float = Field(default_factory=time.time)
    qualification_score: Optional[float] = None