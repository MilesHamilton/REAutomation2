from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import time


class AgentType(str, Enum):
    CONVERSATION = "conversation"
    QUALIFICATION = "qualification"
    OBJECTION_HANDLER = "objection_handler"
    SCHEDULER = "scheduler"
    ANALYTICS = "analytics"


class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class WorkflowState(str, Enum):
    INITIALIZING = "initializing"
    GREETING = "greeting"
    QUALIFYING = "qualifying"
    HANDLING_OBJECTION = "handling_objection"
    SCHEDULING = "scheduling"
    CLOSING = "closing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentMessage(BaseModel):
    agent_type: AgentType
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentDecision(BaseModel):
    agent_type: AgentType
    action: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    next_agent: Optional[AgentType] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class WorkflowContext(BaseModel):
    call_id: str
    workflow_state: WorkflowState = WorkflowState.INITIALIZING
    current_agent: Optional[AgentType] = None
    conversation_history: List[AgentMessage] = Field(default_factory=list)
    lead_data: Dict[str, Any] = Field(default_factory=dict)
    qualification_score: float = 0.0
    objection_count: int = 0
    scheduling_attempts: int = 0
    tier_escalated: bool = False
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    agent_type: AgentType
    response_text: Optional[str] = None
    decision: Optional[AgentDecision] = None
    state_updates: Dict[str, Any] = Field(default_factory=dict)
    should_escalate_tier: bool = False
    should_end_call: bool = False
    processing_time_ms: float = 0


class QualificationFactors(BaseModel):
    intent: float = Field(ge=0.0, le=1.0)
    budget: float = Field(ge=0.0, le=1.0)
    timeline: float = Field(ge=0.0, le=1.0)
    authority: float = Field(ge=0.0, le=1.0)
    needs: float = Field(ge=0.0, le=1.0)

    @property
    def overall_score(self) -> float:
        return (self.intent + self.budget + self.timeline + self.authority + self.needs) / 5.0


class ObjectionType(str, Enum):
    PRICE = "price"
    TIME = "time"
    NEED = "need"
    TRUST = "trust"
    COMPETITION = "competition"
    AUTHORITY = "authority"
    OTHER = "other"


class Objection(BaseModel):
    type: ObjectionType
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: int = Field(ge=1, le=5)
    handled: bool = False
    response: Optional[str] = None


class SchedulingSlot(BaseModel):
    datetime: str  # ISO format
    duration_minutes: int = 30
    timezone: str = "UTC"
    available: bool = True
    priority: int = 1


class AnalyticsMetrics(BaseModel):
    call_id: str
    workflow_duration_ms: float
    agent_transitions: List[Dict[str, Any]] = Field(default_factory=list)
    decision_points: List[AgentDecision] = Field(default_factory=list)
    qualification_progression: List[float] = Field(default_factory=list)
    objection_handling_success: float = 0.0
    tier_escalation_triggered: bool = False
    outcome: str = "unknown"  # qualified, disqualified, scheduled, callback
    success_factors: Dict[str, float] = Field(default_factory=dict)