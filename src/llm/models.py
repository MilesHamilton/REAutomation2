from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[float] = None


class ConversationContext(BaseModel):
    call_id: str
    messages: List[Message] = Field(default_factory=list)
    lead_info: Dict[str, Any] = Field(default_factory=dict)
    qualification_score: Optional[float] = None
    conversation_state: str = "initial"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = Field(default=150, le=500)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    system_prompt: Optional[str] = None
    structured_output: bool = Field(default=False)
    response_format: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    content: str
    usage_tokens: int = 0
    response_time_ms: float = 0
    model_used: str = ""
    confidence_score: Optional[float] = None
    structured_data: Optional[Dict[str, Any]] = None


class HealthStatus(BaseModel):
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float
    concurrent_requests: int
    memory_usage_mb: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    last_check: float
    details: Optional[Dict[str, Any]] = None


class PerformanceMetrics(BaseModel):
    avg_response_time_ms: float
    requests_per_minute: float
    success_rate: float
    concurrent_peak: int
    memory_peak_mb: float
    gpu_memory_peak_mb: Optional[float] = None
    error_count: int
    timestamp: float