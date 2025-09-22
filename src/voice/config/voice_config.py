from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum

from ...config import settings


class AudioFormat(str, Enum):
    PCM_16 = "pcm_16"
    MULAW = "mulaw"
    ALAW = "alaw"
    OPUS = "opus"


class VoiceQuality(str, Enum):
    FAST = "fast"
    NORMAL = "normal"
    HIGH = "high"
    PREMIUM = "premium"


class VoiceConfig(BaseModel):
    """Voice pipeline configuration"""

    # Audio Configuration
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    bit_depth: int = Field(default=16, description="Audio bit depth")
    format: AudioFormat = Field(default=AudioFormat.PCM_16, description="Audio format")
    buffer_size: int = Field(default=1024, description="Audio buffer size")

    # Processing Configuration
    chunk_duration_ms: int = Field(default=100, description="Audio chunk duration in ms")
    max_silence_duration: float = Field(default=2.0, description="Max silence before processing")
    vad_threshold: float = Field(default=0.5, description="Voice activity detection threshold")

    # Performance Configuration
    max_latency_ms: float = Field(default=200, description="Maximum acceptable latency")
    concurrent_calls_limit: int = Field(default=5, description="Maximum concurrent calls")
    processing_timeout: float = Field(default=30.0, description="Processing timeout in seconds")

    # Quality Configuration
    quality_level: VoiceQuality = Field(default=VoiceQuality.NORMAL, description="Voice quality level")
    noise_suppression: bool = Field(default=True, description="Enable noise suppression")
    echo_cancellation: bool = Field(default=True, description="Enable echo cancellation")

    # Model Configuration
    stt_model: str = Field(default="whisper-small", description="STT model to use")
    tts_voice: str = Field(default="default", description="Default TTS voice")
    language_code: str = Field(default="en-US", description="Language code for processing")


class TwilioConfig(BaseModel):
    """Twilio-specific configuration"""

    account_sid: str = Field(description="Twilio account SID")
    auth_token: str = Field(description="Twilio auth token")
    phone_number: str = Field(description="Twilio phone number")

    # WebRTC Configuration
    region: str = Field(default="us1", description="Twilio region")
    edge_locations: List[str] = Field(default=["ashburn"], description="Edge locations")

    # Call Configuration
    timeout_seconds: int = Field(default=300, description="Call timeout in seconds")
    record_calls: bool = Field(default=False, description="Enable call recording")
    call_webhook_url: Optional[str] = Field(default=None, description="Webhook URL for call events")

    # Media Configuration
    media_encryption: bool = Field(default=True, description="Enable media encryption")
    codec_preferences: List[str] = Field(default=["opus", "pcmu"], description="Codec preferences")


class STTConfig(BaseModel):
    """Speech-to-Text configuration"""

    model: str = Field(default="whisper-small", description="STT model name")
    language: str = Field(default="en", description="Language for transcription")

    # Whisper Configuration
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")
    compute_type: str = Field(default="float16", description="Compute type for inference")
    beam_size: int = Field(default=5, description="Beam size for decoding")

    # Processing Configuration
    vad_enabled: bool = Field(default=True, description="Enable voice activity detection")
    silence_threshold: float = Field(default=0.5, description="Silence detection threshold")
    min_speech_duration: float = Field(default=0.5, description="Minimum speech duration")
    max_speech_duration: float = Field(default=30.0, description="Maximum speech duration")

    # Quality Configuration
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for results")
    filter_profanity: bool = Field(default=False, description="Filter profanity from results")


class TTSConfig(BaseModel):
    """Text-to-Speech configuration"""

    # Engine Configuration
    primary_engine: str = Field(default="piper", description="Primary TTS engine")
    fallback_engine: str = Field(default="coqui", description="Fallback TTS engine")

    # Voice Configuration
    voice_id: str = Field(default="default", description="Voice ID to use")
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Speaking rate")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Voice pitch")
    volume: float = Field(default=1.0, ge=0.1, le=2.0, description="Voice volume")

    # Quality Configuration
    quality: VoiceQuality = Field(default=VoiceQuality.NORMAL, description="TTS quality level")
    stability: float = Field(default=0.75, ge=0.0, le=1.0, description="Voice stability")
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0, description="Similarity boost")

    # Processing Configuration
    streaming: bool = Field(default=True, description="Enable streaming TTS")
    chunk_size: int = Field(default=1024, description="Audio chunk size for streaming")
    optimize_latency: bool = Field(default=True, description="Optimize for low latency")


class ElevenLabsConfig(BaseModel):
    """11Labs premium TTS configuration"""

    api_key: Optional[str] = Field(default=None, description="11Labs API key")
    model_id: str = Field(default="eleven_turbo_v2", description="11Labs model ID")
    voice_id: Optional[str] = Field(default=None, description="11Labs voice ID")

    # Quality Configuration
    stability: float = Field(default=0.75, ge=0.0, le=1.0, description="Voice stability")
    similarity_boost: float = Field(default=0.75, ge=0.0, le=1.0, description="Similarity boost")
    style: float = Field(default=0.0, ge=0.0, le=1.0, description="Style exaggeration")
    use_speaker_boost: bool = Field(default=True, description="Use speaker boost")

    # Optimization Configuration
    optimize_streaming_latency: int = Field(default=2, ge=0, le=4, description="Latency optimization level")
    output_format: str = Field(default="mp3_44100_128", description="Output audio format")

    # Cost Controls
    max_characters_per_request: int = Field(default=1000, description="Max characters per request")
    daily_character_limit: int = Field(default=10000, description="Daily character limit")


class VoicePipelineConfig(BaseModel):
    """Complete voice pipeline configuration"""

    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    twilio: TwilioConfig
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)

    # Pipeline Configuration
    enable_middleware: bool = Field(default=True, description="Enable voice middleware")
    enable_quality_monitoring: bool = Field(default=True, description="Enable quality monitoring")
    enable_latency_optimization: bool = Field(default=True, description="Enable latency optimization")

    # Tier Switching Configuration
    tier_escalation_enabled: bool = Field(default=True, description="Enable tier escalation")
    tier_escalation_threshold: float = Field(default=0.7, description="Threshold for tier escalation")
    cost_limit_per_call: float = Field(default=0.10, description="Cost limit per call in USD")

    # Monitoring Configuration
    collect_metrics: bool = Field(default=True, description="Collect performance metrics")
    log_audio_events: bool = Field(default=True, description="Log audio processing events")
    debug_mode: bool = Field(default=False, description="Enable debug mode")


def create_voice_config() -> VoicePipelineConfig:
    """Create voice pipeline configuration from settings"""

    twilio_config = TwilioConfig(
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        phone_number=settings.twilio_phone_number
    )

    # STT Configuration
    stt_config = STTConfig(
        model=settings.stt_model,
        device="cuda" if settings.llm_gpu_memory_limit > 0 else "cpu"
    )

    # TTS Configuration
    tts_config = TTSConfig(
        primary_engine=settings.tts_engine,
        quality=VoiceQuality.NORMAL if settings.tts_engine == "local" else VoiceQuality.HIGH
    )

    # 11Labs Configuration
    elevenlabs_config = ElevenLabsConfig(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice,
        model_id=settings.elevenlabs_model
    )

    # Voice Configuration
    voice_config = VoiceConfig(
        concurrent_calls_limit=settings.max_concurrent_calls,
        max_latency_ms=200.0
    )

    return VoicePipelineConfig(
        voice=voice_config,
        twilio=twilio_config,
        stt=stt_config,
        tts=tts_config,
        elevenlabs=elevenlabs_config,
        tier_escalation_threshold=settings.tier_escalation_threshold,
        cost_limit_per_call=settings.cost_per_call_limit,
        debug_mode=settings.debug
    )


# Global configuration instance
voice_config = create_voice_config()