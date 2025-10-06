from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    # LLM Configuration
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3.1:8b-instruct-q4_0", env="OLLAMA_MODEL")
    llm_max_concurrent: int = Field(default=5, env="LLM_MAX_CONCURRENT")
    llm_gpu_memory_limit: int = Field(default=6144, env="LLM_GPU_MEMORY_LIMIT")

    # Ollama Connection Pooling
    ollama_connection_pool_size: int = Field(default=20, env="OLLAMA_CONNECTION_POOL_SIZE")
    ollama_connection_pool_limit_per_host: int = Field(default=10, env="OLLAMA_CONNECTION_POOL_LIMIT_PER_HOST")
    ollama_keepalive_timeout: int = Field(default=30, env="OLLAMA_KEEPALIVE_TIMEOUT")
    ollama_connection_timeout: int = Field(default=10, env="OLLAMA_CONNECTION_TIMEOUT")
    ollama_request_timeout: int = Field(default=30, env="OLLAMA_REQUEST_TIMEOUT")

    # Ollama Request Batching
    ollama_batch_enabled: bool = Field(default=True, env="OLLAMA_BATCH_ENABLED")
    ollama_batch_window_ms: int = Field(default=100, env="OLLAMA_BATCH_WINDOW_MS")
    ollama_batch_max_size: int = Field(default=5, env="OLLAMA_BATCH_MAX_SIZE")
    ollama_batch_similarity_threshold: float = Field(default=0.9, env="OLLAMA_BATCH_SIMILARITY_THRESHOLD")

    # Ollama Streaming
    ollama_streaming_enabled: bool = Field(default=True, env="OLLAMA_STREAMING_ENABLED")
    ollama_streaming_chunk_size: int = Field(default=512, env="OLLAMA_STREAMING_CHUNK_SIZE")

    # Ollama GPU Management
    ollama_gpu_memory_threshold_mb: int = Field(default=5120, env="OLLAMA_GPU_MEMORY_THRESHOLD_MB")
    ollama_gpu_monitoring_interval: int = Field(default=10, env="OLLAMA_GPU_MONITORING_INTERVAL")
    ollama_auto_unload_models: bool = Field(default=True, env="OLLAMA_AUTO_UNLOAD_MODELS")

    # Ollama Adaptive Concurrency
    ollama_adaptive_concurrency: bool = Field(default=True, env="OLLAMA_ADAPTIVE_CONCURRENCY")
    ollama_min_concurrent: int = Field(default=3, env="OLLAMA_MIN_CONCURRENT")
    ollama_max_concurrent: int = Field(default=8, env="OLLAMA_MAX_CONCURRENT")

    # Context Window Management
    context_window_size: int = Field(default=8192, env="CONTEXT_WINDOW_SIZE")  # Llama 3.1 8B
    context_reserve_tokens: int = Field(default=1024, env="CONTEXT_RESERVE_TOKENS")  # Reserve for response
    context_warning_threshold: float = Field(default=0.8, env="CONTEXT_WARNING_THRESHOLD")  # Warn at 80%
    context_max_messages: int = Field(default=50, env="CONTEXT_MAX_MESSAGES")  # Hard limit on messages

    # Context Pruning Configuration
    context_pruning_enabled: bool = Field(default=True, env="CONTEXT_PRUNING_ENABLED")
    context_pruning_strategy: str = Field(default="sliding_window", env="CONTEXT_PRUNING_STRATEGY")  # sliding_window, importance, hybrid, summarize
    context_sliding_window_size: int = Field(default=10, env="CONTEXT_SLIDING_WINDOW_SIZE")
    context_preserve_important: bool = Field(default=True, env="CONTEXT_PRESERVE_IMPORTANT")
    context_importance_threshold: float = Field(default=0.7, env="CONTEXT_IMPORTANCE_THRESHOLD")

    # Context Summarization
    context_summarization_enabled: bool = Field(default=True, env="CONTEXT_SUMMARIZATION_ENABLED")
    context_summary_max_tokens: int = Field(default=200, env="CONTEXT_SUMMARY_MAX_TOKENS")
    context_summary_batch_size: int = Field(default=15, env="CONTEXT_SUMMARY_BATCH_SIZE")  # Summarize every N old messages

    # Twilio Configuration
    twilio_account_sid: str = Field(env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: str = Field(env="TWILIO_PHONE_NUMBER")

    # 11Labs Configuration
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    elevenlabs_model: str = Field(default="eleven_turbo_v2", env="ELEVENLABS_MODEL")
    elevenlabs_voice: Optional[str] = Field(default=None, env="ELEVENLABS_VOICE")

    # Database Configuration
    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # Voice Configuration
    tts_engine: str = Field(default="local", env="TTS_ENGINE")  # local or elevenlabs
    stt_model: str = Field(default="whisper-small", env="STT_MODEL")
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    audio_channels: int = Field(default=1, env="AUDIO_CHANNELS")

    # Local TTS Configuration (Phase 1: Piper Integration)
    piper_models_path: str = Field(default="./models/piper", env="PIPER_MODELS_PATH")
    piper_default_voice: str = Field(default="en_US-lessac-medium", env="PIPER_DEFAULT_VOICE")
    piper_enable_streaming: bool = Field(default=True, env="PIPER_ENABLE_STREAMING")
    piper_chunk_size_ms: int = Field(default=100, env="PIPER_CHUNK_SIZE_MS")
    piper_prewarm_model: bool = Field(default=True, env="PIPER_PREWARM_MODEL")
    piper_use_gpu: bool = Field(default=False, env="PIPER_USE_GPU")  # ONNX GPU support
    piper_speaker_id: Optional[int] = Field(default=None, env="PIPER_SPEAKER_ID")  # For multi-speaker models

    # Coqui TTS Configuration
    coqui_model_name: str = Field(default="tts_models/en/ljspeech/fast_pitch", env="COQUI_MODEL_NAME")
    coqui_enable_streaming: bool = Field(default=True, env="COQUI_ENABLE_STREAMING")
    coqui_use_gpu: bool = Field(default=True, env="COQUI_USE_GPU")

    # Call Configuration
    max_concurrent_calls: int = Field(default=5, env="MAX_CONCURRENT_CALLS")
    call_timeout: int = Field(default=300, env="CALL_TIMEOUT")
    qualification_threshold: float = Field(default=0.8, env="QUALIFICATION_THRESHOLD")
    tier_escalation_threshold: float = Field(default=0.7, env="TIER_ESCALATION_THRESHOLD")

    # Cost Controls
    daily_budget: float = Field(default=50.00, env="DAILY_BUDGET")
    cost_per_call_limit: float = Field(default=0.10, env="COST_PER_CALL_LIMIT")
    monthly_call_limit: int = Field(default=5000, env="MONTHLY_CALL_LIMIT")

    # Monitoring
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    performance_monitoring_enabled: bool = Field(default=True, env="PERFORMANCE_MONITORING_ENABLED")
    system_metrics_enabled: bool = Field(default=True, env="SYSTEM_METRICS_ENABLED")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Voice-Agent Integration Configuration
    voice_agent_integration_enabled: bool = Field(default=True, env="VOICE_AGENT_INTEGRATION_ENABLED")
    agent_processing_timeout_ms: int = Field(default=500, env="AGENT_PROCESSING_TIMEOUT_MS")
    voice_response_max_latency_ms: int = Field(default=2000, env="VOICE_RESPONSE_MAX_LATENCY_MS")
    state_sync_interval_ms: int = Field(default=1000, env="STATE_SYNC_INTERVAL_MS")
    fallback_to_direct_llm: bool = Field(default=True, env="FALLBACK_TO_DIRECT_LLM")

    # Circuit Breaker Configuration
    circuit_breaker_enabled: bool = Field(default=True, env="CIRCUIT_BREAKER_ENABLED")
    circuit_breaker_failure_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
    circuit_breaker_timeout_seconds: int = Field(default=60, env="CIRCUIT_BREAKER_TIMEOUT_SECONDS")
    circuit_breaker_half_open_max_calls: int = Field(default=3, env="CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS")

    # Performance Optimization
    response_cache_enabled: bool = Field(default=True, env="RESPONSE_CACHE_ENABLED")
    response_cache_ttl_seconds: int = Field(default=300, env="RESPONSE_CACHE_TTL_SECONDS")
    agent_context_prewarm: bool = Field(default=True, env="AGENT_CONTEXT_PREWARM")
    parallel_processing_enabled: bool = Field(default=True, env="PARALLEL_PROCESSING_ENABLED")

    # Google Sheets Configuration
    google_sheets_credentials_file: str = Field(default="credentials.json", env="GOOGLE_SHEETS_CREDENTIALS_FILE")
    input_spreadsheet_id: Optional[str] = Field(default=None, env="INPUT_SPREADSHEET_ID")
    output_spreadsheet_id: Optional[str] = Field(default=None, env="OUTPUT_SPREADSHEET_ID")
    sheets_enabled: bool = Field(default=True, env="SHEETS_ENABLED")

    # LangSmith Monitoring Configuration
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="reautomation2", env="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    langsmith_enabled: bool = Field(default=True, env="LANGSMITH_ENABLED")
    langsmith_batch_size: int = Field(default=100, env="LANGSMITH_BATCH_SIZE")
    langsmith_flush_interval: int = Field(default=30, env="LANGSMITH_FLUSH_INTERVAL")
    langsmith_fallback_enabled: bool = Field(default=True, env="LANGSMITH_FALLBACK_ENABLED")

    # Audio Processing Configuration
    enable_echo_cancellation: bool = Field(default=True, env="ENABLE_ECHO_CANCELLATION")
    enable_noise_reduction: bool = Field(default=True, env="ENABLE_NOISE_REDUCTION")
    enable_jitter_buffer: bool = Field(default=True, env="ENABLE_JITTER_BUFFER")
    audio_chunk_duration_ms: int = Field(default=20, env="AUDIO_CHUNK_DURATION_MS")  # 20ms chunks
    jitter_buffer_target_ms: int = Field(default=80, env="JITTER_BUFFER_TARGET_MS")  # 60-100ms range
    jitter_buffer_min_ms: int = Field(default=40, env="JITTER_BUFFER_MIN_MS")
    jitter_buffer_max_ms: int = Field(default=200, env="JITTER_BUFFER_MAX_MS")
    noise_reduction_strength: float = Field(default=0.8, env="NOISE_REDUCTION_STRENGTH")  # 0.0-1.0
    echo_filter_length: int = Field(default=1024, env="ECHO_FILTER_LENGTH")  # Adaptive filter taps
    audio_buffer_duration_ms: int = Field(default=500, env="AUDIO_BUFFER_DURATION_MS")
    max_audio_latency_ms: int = Field(default=200, env="MAX_AUDIO_LATENCY_MS")  # Target <200ms

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
