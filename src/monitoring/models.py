from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


# Enums for monitoring system
class WorkflowStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    ERROR = "error"
    THRESHOLD = "threshold"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class MetricCategory(str, Enum):
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    COST = "cost"
    QUALITY = "quality"


class AggregationLevel(str, Enum):
    CALL = "call"
    AGENT = "agent"
    WORKFLOW = "workflow"
    SYSTEM = "system"


class TimeWindow(str, Enum):
    REAL_TIME = "real_time"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class CostType(str, Enum):
    LLM = "llm"
    TTS = "tts"
    STT = "stt"
    API = "api"
    INFRASTRUCTURE = "infrastructure"


class ServiceProvider(str, Enum):
    OLLAMA = "ollama"
    ELEVENLABS = "elevenlabs"
    TWILIO = "twilio"
    LANGSMITH = "langsmith"
    PIPECAT = "pipecat"
    LOCAL = "local"


# Base monitoring models
class MonitoringBaseModel(BaseModel):
    """Base model for all monitoring data structures"""

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


# Workflow tracing models
class WorkflowTraceCreate(MonitoringBaseModel):
    """Model for creating a workflow trace"""
    call_id: str
    workflow_name: str
    workflow_version: Optional[str] = None
    parent_trace_id: Optional[str] = None
    initial_context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WorkflowTraceUpdate(MonitoringBaseModel):
    """Model for updating a workflow trace"""
    status: Optional[WorkflowStatus] = None
    final_context: Optional[Dict[str, Any]] = None
    workflow_state: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    total_agents_executed: Optional[int] = None
    total_llm_calls: Optional[int] = None
    total_cost: Optional[float] = None


class WorkflowTrace(MonitoringBaseModel):
    """Complete workflow trace model"""
    trace_id: str
    call_id: str
    workflow_name: str
    workflow_version: Optional[str] = None
    parent_trace_id: Optional[str] = None
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    initial_context: Optional[Dict[str, Any]] = None
    final_context: Optional[Dict[str, Any]] = None
    workflow_state: Optional[Dict[str, Any]] = None
    error_occurred: bool = False
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    total_agents_executed: int = 0
    total_llm_calls: int = 0
    total_cost: float = 0.0
    langsmith_run_id: Optional[str] = None
    langsmith_project: Optional[str] = None
    langsmith_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


# Agent execution models
class AgentExecutionCreate(MonitoringBaseModel):
    """Model for creating an agent execution"""
    trace_id: str
    agent_type: str
    agent_name: str
    execution_order: int
    agent_version: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    agent_state_before: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AgentExecutionUpdate(MonitoringBaseModel):
    """Model for updating an agent execution"""
    output_data: Optional[Dict[str, Any]] = None
    agent_state_after: Optional[Dict[str, Any]] = None
    decision_rationale: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    routing_decision: Optional[str] = None
    llm_calls_made: Optional[int] = Field(None, ge=0)
    tokens_consumed: Optional[int] = Field(None, ge=0)
    processing_cost: Optional[float] = Field(None, ge=0.0)
    error_message: Optional[str] = None
    retry_count: Optional[int] = Field(None, ge=0)


class AgentExecution(MonitoringBaseModel):
    """Complete agent execution model"""
    execution_id: str
    trace_id: str
    agent_type: str
    agent_name: str
    agent_version: Optional[str] = None
    execution_order: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    agent_state_before: Optional[Dict[str, Any]] = None
    agent_state_after: Optional[Dict[str, Any]] = None
    decision_rationale: Optional[str] = None
    confidence_score: Optional[float] = None
    routing_decision: Optional[str] = None
    llm_calls_made: int = 0
    tokens_consumed: int = 0
    processing_cost: float = 0.0
    error_occurred: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    langsmith_run_id: Optional[str] = None
    created_at: datetime


# Performance metrics models
class MetricValue(MonitoringBaseModel):
    """Statistical metric value with percentiles"""
    value: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None


class PerformanceMetricCreate(MonitoringBaseModel):
    """Model for creating a performance metric"""
    metric_category: MetricCategory
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None
    call_id: Optional[str] = None
    trace_id: Optional[str] = None
    agent_type: Optional[str] = None
    workflow_name: Optional[str] = None
    aggregation_level: AggregationLevel
    time_window: TimeWindow
    time_bucket: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class PerformanceMetric(MonitoringBaseModel):
    """Complete performance metric model"""
    metric_id: str
    metric_category: MetricCategory
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None
    call_id: Optional[str] = None
    trace_id: Optional[str] = None
    agent_type: Optional[str] = None
    workflow_name: Optional[str] = None
    aggregation_level: AggregationLevel
    time_window: TimeWindow
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None
    percentile_50: Optional[float] = None
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    threshold_breached: bool = False
    recorded_at: datetime
    time_bucket: datetime
    metadata: Optional[Dict[str, Any]] = None


# Cost tracking models
class CostBreakdownCreate(MonitoringBaseModel):
    """Model for creating cost breakdown"""
    call_id: Optional[str] = None
    trace_id: Optional[str] = None
    execution_id: Optional[str] = None
    cost_type: CostType
    service_provider: ServiceProvider
    cost_amount: float = Field(ge=0.0)
    cost_currency: str = "USD"
    resource_type: Optional[str] = None  # tokens, characters, minutes, requests
    resource_quantity: Optional[float] = Field(None, ge=0.0)
    unit_cost: Optional[float] = Field(None, ge=0.0)
    agent_type: Optional[str] = None
    workflow_step: Optional[str] = None
    tier_used: Optional[str] = None
    billing_period: str  # YYYY-MM-DD format
    budget_category: Optional[str] = None


class CostBreakdown(MonitoringBaseModel):
    """Complete cost breakdown model"""
    cost_id: str
    call_id: Optional[str] = None
    trace_id: Optional[str] = None
    execution_id: Optional[str] = None
    cost_type: CostType
    service_provider: ServiceProvider
    cost_amount: float
    cost_currency: str = "USD"
    resource_type: Optional[str] = None
    resource_quantity: Optional[float] = None
    unit_cost: Optional[float] = None
    agent_type: Optional[str] = None
    workflow_step: Optional[str] = None
    tier_used: Optional[str] = None
    cost_incurred_at: datetime
    billing_period: str
    daily_budget_impact: Optional[float] = None
    monthly_budget_impact: Optional[float] = None
    budget_category: Optional[str] = None
    cost_efficiency_score: Optional[float] = None
    optimization_opportunity: Optional[str] = None
    created_at: datetime


# Alert system models
class AlertRule(MonitoringBaseModel):
    """Alert rule definition"""
    rule_name: str
    metric_name: str
    condition: str  # e.g., "> 1000", "< 0.95"
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    alert_frequency: str = "immediate"  # immediate, hourly, daily
    enabled: bool = True


class AlertCreate(MonitoringBaseModel):
    """Model for creating an alert"""
    alert_type: AlertType
    alert_level: AlertLevel
    alert_title: str
    alert_message: str
    call_id: Optional[str] = None
    trace_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    rule_name: Optional[str] = None
    rule_condition: Optional[str] = None
    alert_frequency: Optional[str] = None
    notification_channels: Optional[List[str]] = Field(default_factory=list)


class AlertUpdate(MonitoringBaseModel):
    """Model for updating an alert"""
    status: Optional[AlertStatus] = None
    acknowledged_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    auto_resolved: Optional[bool] = None


class Alert(MonitoringBaseModel):
    """Complete alert model"""
    alert_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    alert_title: str
    alert_message: str
    call_id: Optional[str] = None
    trace_id: Optional[str] = None
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    notification_channels: Optional[List[str]] = None
    notifications_sent: int = 0
    last_notification_at: Optional[datetime] = None
    rule_name: Optional[str] = None
    rule_condition: Optional[str] = None
    alert_frequency: Optional[str] = None
    resolution_notes: Optional[str] = None
    auto_resolved: bool = False
    triggered_at: datetime
    updated_at: Optional[datetime] = None


# Dashboard and analytics models
class MetricSummary(MonitoringBaseModel):
    """Summary statistics for a metric"""
    metric_name: str
    metric_category: MetricCategory
    total_data_points: int
    time_range_start: datetime
    time_range_end: datetime
    current_value: Optional[float] = None
    average_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    percentile_95: Optional[float] = None
    trend_direction: Optional[str] = None  # "up", "down", "stable"
    trend_percentage: Optional[float] = None


class WorkflowAnalytics(MonitoringBaseModel):
    """Analytics for workflow performance"""
    workflow_name: str
    total_executions: int
    success_rate: float = Field(ge=0.0, le=1.0)
    average_duration_ms: Optional[float] = None
    average_cost: Optional[float] = None
    total_cost: float = 0.0
    most_common_failure_type: Optional[str] = None
    agent_performance: List[Dict[str, Any]] = Field(default_factory=list)
    time_range_start: datetime
    time_range_end: datetime


class SystemHealthStatus(MonitoringBaseModel):
    """Overall system health status"""
    overall_status: str  # "healthy", "degraded", "critical"
    langsmith_client_status: Dict[str, Any]
    active_alerts_count: int
    critical_alerts_count: int
    recent_error_rate: float
    average_response_time_ms: Optional[float] = None
    cost_burn_rate: Optional[float] = None
    daily_budget_utilization: Optional[float] = None
    last_updated: datetime


# Request/Response models for API
class MonitoringDashboardRequest(MonitoringBaseModel):
    """Request model for dashboard data"""
    time_range_start: datetime
    time_range_end: datetime
    workflow_names: Optional[List[str]] = None
    agent_types: Optional[List[str]] = None
    call_ids: Optional[List[str]] = None
    include_costs: bool = True
    include_alerts: bool = True


class MonitoringDashboardResponse(MonitoringBaseModel):
    """Response model for dashboard data"""
    system_health: SystemHealthStatus
    metric_summaries: List[MetricSummary]
    workflow_analytics: List[WorkflowAnalytics]
    recent_alerts: List[Alert]
    cost_summary: Dict[str, float]
    time_range: Dict[str, datetime]


# Trace query models
class TraceQueryRequest(MonitoringBaseModel):
    """Request model for trace queries"""
    call_ids: Optional[List[str]] = None
    workflow_names: Optional[List[str]] = None
    status: Optional[List[WorkflowStatus]] = None
    start_time_after: Optional[datetime] = None
    start_time_before: Optional[datetime] = None
    include_executions: bool = True
    include_metrics: bool = False
    include_costs: bool = False
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class TraceQueryResponse(MonitoringBaseModel):
    """Response model for trace queries"""
    traces: List[WorkflowTrace]
    total_count: int
    has_more: bool
    query_time_ms: float


# Validation functions
def validate_billing_period(period: str) -> bool:
    """Validate billing period format (YYYY-MM-DD)"""
    try:
        datetime.strptime(period, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def validate_confidence_score(score: Optional[float]) -> Optional[float]:
    """Validate confidence score is between 0 and 1"""
    if score is not None and (score < 0.0 or score > 1.0):
        raise ValueError("Confidence score must be between 0.0 and 1.0")
    return score


# Utility functions for model creation
def create_workflow_trace_id() -> str:
    """Generate a unique workflow trace ID"""
    return f"trace_{uuid.uuid4().hex[:16]}"


def create_execution_id() -> str:
    """Generate a unique execution ID"""
    return f"exec_{uuid.uuid4().hex[:12]}"


def create_metric_id(category: str, name: str) -> str:
    """Generate a unique metric ID"""
    return f"metric_{category}_{name}_{uuid.uuid4().hex[:8]}"


def create_cost_id(cost_type: str, provider: str) -> str:
    """Generate a unique cost ID"""
    return f"cost_{cost_type}_{provider}_{uuid.uuid4().hex[:8]}"


def create_alert_id(alert_type: str) -> str:
    """Generate a unique alert ID"""
    return f"alert_{alert_type}_{uuid.uuid4().hex[:8]}"