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
    auto_detect_language: bool = Field(default=False)
    vad_enabled: bool = Field(default=True)
    silence_threshold: float = Field(default=0.5)
    max_silence_duration: float = Field(default=2.0)
    # Phase 3: Performance optimization
    use_faster_whisper: bool = Field(default=True)  # Use faster-whisper if available
    compute_type: str = Field(default="float16")  # float16, int8, int8_float16
    num_workers: int = Field(default=1)  # Number of worker threads for faster-whisper
    beam_size: int = Field(default=5)  # Beam search size (1 = greedy, higher = more accurate but slower)


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
    detected_language: Optional[str] = None
    language_probability: Optional[float] = None
    chunk_id: Optional[int] = None
    # Audio quality metrics (Phase 2)
    audio_quality_score: Optional[float] = None
    snr_db: Optional[float] = None
    clipping_detected: bool = False
    quality_assessment: Optional[str] = None


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

    # Workflow integration fields
    workflow_context_id: Optional[str] = None
    current_agent: Optional[str] = None
    agent_transition_history: List[str] = Field(default_factory=list)
    workflow_state: Optional[str] = None
    last_state_sync: Optional[float] = None
    integration_enabled: bool = Field(default=False)


class VoicePipelineConfig(BaseModel):
    """Configuration for voice pipeline"""
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    tts_provider: TTSProvider = TTSProvider.LOCAL_PIPER
    stt_provider: str = Field(default="whisper-local")
    llm_model: str = Field(default="llama3.1:8b")
    tier_escalation_threshold: float = Field(default=0.7)
    max_call_duration_minutes: int = Field(default=30)
    enable_cost_tracking: bool = Field(default=True)
    enable_analytics: bool = Field(default=True)


class TierSwitchEvent(BaseModel):
    call_id: str
    from_tier: TTSProvider
    to_tier: TTSProvider
    trigger: str  # "qualification", "manual", "fallback"
    timestamp: float = Field(default_factory=time.time)
    qualification_score: Optional[float] = None


class AgentTransition(BaseModel):
    """Model for tracking agent transitions in voice calls"""
    call_id: str
    from_agent: str
    to_agent: str
    timestamp: float = Field(default_factory=time.time)
    trigger: str  # "workflow", "escalation", "manual"
    context_preserved: bool = True
    transition_duration_ms: float = 0.0


class VoiceAgentIntegrationContext(BaseModel):
    """Context for voice and agent integration"""
    call_id: str
    voice_session_id: str
    workflow_context_id: Optional[str] = None
    sync_status: str = "synced"  # "synced", "pending", "failed"
    last_sync_timestamp: float = Field(default_factory=time.time)
    error_count: int = 0
    fallback_active: bool = False
    integration_metadata: Dict[str, Any] = Field(default_factory=dict)
