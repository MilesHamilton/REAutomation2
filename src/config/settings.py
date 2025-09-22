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
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Google Sheets Configuration
    google_sheets_credentials_file: str = Field(default="credentials.json", env="GOOGLE_SHEETS_CREDENTIALS_FILE")
    input_spreadsheet_id: Optional[str] = Field(default=None, env="INPUT_SPREADSHEET_ID")
    output_spreadsheet_id: Optional[str] = Field(default=None, env="OUTPUT_SPREADSHEET_ID")
    sheets_enabled: bool = Field(default=True, env="SHEETS_ENABLED")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
