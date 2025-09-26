from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from .models import Base


# LangSmith Monitoring Tables

class WorkflowTrace(Base):
    """LangSmith workflow traces for agent orchestration monitoring"""
    __tablename__ = "workflow_traces"

    id = Column(Integer, primary_key=True, index=True)
    trace_id = Column(String(255), unique=True, nullable=False, index=True)
    call_id = Column(String(255), ForeignKey("calls.call_id"), nullable=False, index=True)

    # Workflow details
    workflow_name = Column(String(100), nullable=False, index=True)
    workflow_version = Column(String(20), nullable=True)
    parent_trace_id = Column(String(255), nullable=True, index=True)

    # Execution status
    status = Column(String(50), nullable=False, index=True)  # running, completed, failed, cancelled
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)

    # Context and state
    initial_context = Column(JSON, nullable=True)
    final_context = Column(JSON, nullable=True)
    workflow_state = Column(JSON, nullable=True)

    # Error handling
    error_occurred = Column(Boolean, default=False, index=True)
    error_message = Column(Text, nullable=True)
    error_type = Column(String(100), nullable=True)

    # Performance metrics
    total_agents_executed = Column(Integer, default=0)
    total_llm_calls = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)

    # LangSmith integration
    langsmith_run_id = Column(String(255), nullable=True, unique=True, index=True)
    langsmith_project = Column(String(100), nullable=True)
    langsmith_url = Column(String(500), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    agent_executions = relationship("AgentExecution", back_populates="workflow_trace")

    # Indexes
    __table_args__ = (
        Index('ix_workflow_traces_call_status', 'call_id', 'status'),
        Index('ix_workflow_traces_workflow_time', 'workflow_name', 'start_time'),
        Index('ix_workflow_traces_status_time', 'status', 'start_time'),
    )


class AgentExecution(Base):
    """Individual agent execution records within workflows"""
    __tablename__ = "agent_executions"

    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(String(255), unique=True, nullable=False, index=True)
    trace_id = Column(String(255), ForeignKey("workflow_traces.trace_id"), nullable=False, index=True)

    # Agent details
    agent_type = Column(String(100), nullable=False, index=True)  # conversation, qualification, objection_handler, etc.
    agent_name = Column(String(100), nullable=False)
    agent_version = Column(String(20), nullable=True)
    execution_order = Column(Integer, nullable=False)

    # Execution timing
    start_time = Column(DateTime(timezone=True), nullable=False, index=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Float, nullable=True)

    # Agent state
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    agent_state_before = Column(JSON, nullable=True)
    agent_state_after = Column(JSON, nullable=True)

    # Decision making
    decision_rationale = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    routing_decision = Column(String(100), nullable=True)

    # Performance metrics
    llm_calls_made = Column(Integer, default=0)
    tokens_consumed = Column(Integer, default=0)
    processing_cost = Column(Float, default=0.0)

    # Error handling
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # LangSmith integration
    langsmith_run_id = Column(String(255), nullable=True, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    workflow_trace = relationship("WorkflowTrace", back_populates="agent_executions")

    # Indexes
    __table_args__ = (
        Index('ix_agent_exec_trace_order', 'trace_id', 'execution_order'),
        Index('ix_agent_exec_type_time', 'agent_type', 'start_time'),
        Index('ix_agent_exec_duration', 'duration_ms'),
    )


class PerformanceMetrics(Base):
    """Real-time performance metrics and monitoring data"""
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_id = Column(String(255), unique=True, nullable=False, index=True)

    # Metric identification
    metric_category = Column(String(50), nullable=False, index=True)  # response_time, throughput, error_rate, cost
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)

    # Context
    call_id = Column(String(255), nullable=True, index=True)
    trace_id = Column(String(255), nullable=True, index=True)
    agent_type = Column(String(100), nullable=True, index=True)
    workflow_name = Column(String(100), nullable=True)

    # Aggregation level
    aggregation_level = Column(String(50), nullable=False, index=True)  # call, agent, workflow, system
    time_window = Column(String(20), nullable=False, index=True)  # real_time, minute, hour, day

    # Statistical data
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    avg_value = Column(Float, nullable=True)
    percentile_50 = Column(Float, nullable=True)
    percentile_95 = Column(Float, nullable=True)
    percentile_99 = Column(Float, nullable=True)

    # Threshold monitoring
    threshold_warning = Column(Float, nullable=True)
    threshold_critical = Column(Float, nullable=True)
    threshold_breached = Column(Boolean, default=False, index=True)

    # Time series data
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    time_bucket = Column(DateTime(timezone=True), nullable=False, index=True)  # Rounded time for aggregation

    # Additional metadata
    additional_metadata = Column(JSON, nullable=True)

    # Indexes for time-series queries
    __table_args__ = (
        Index('ix_perf_metrics_category_time', 'metric_category', 'time_bucket'),
        Index('ix_perf_metrics_name_time', 'metric_name', 'time_bucket'),
        Index('ix_perf_metrics_agent_time', 'agent_type', 'time_bucket'),
        Index('ix_perf_metrics_call_time', 'call_id', 'recorded_at'),
    )


class CostBreakdown(Base):
    """Detailed cost breakdown with LangSmith attribution"""
    __tablename__ = "cost_breakdown"

    id = Column(Integer, primary_key=True, index=True)
    cost_id = Column(String(255), unique=True, nullable=False, index=True)

    # Cost identification
    call_id = Column(String(255), nullable=True, index=True)
    trace_id = Column(String(255), nullable=True, index=True)
    execution_id = Column(String(255), nullable=True, index=True)

    # Cost details
    cost_type = Column(String(50), nullable=False, index=True)  # llm, tts, stt, api, infrastructure
    service_provider = Column(String(100), nullable=False, index=True)  # ollama, elevenlabs, twilio, langsmith
    cost_amount = Column(Float, nullable=False)
    cost_currency = Column(String(3), default='USD')

    # Resource consumption
    resource_type = Column(String(50), nullable=True)  # tokens, characters, minutes, requests
    resource_quantity = Column(Float, nullable=True)
    unit_cost = Column(Float, nullable=True)

    # Attribution
    agent_type = Column(String(100), nullable=True, index=True)
    workflow_step = Column(String(100), nullable=True)
    tier_used = Column(String(50), nullable=True, index=True)

    # Time tracking
    cost_incurred_at = Column(DateTime(timezone=True), nullable=False, index=True)
    billing_period = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD format

    # Budget tracking
    daily_budget_impact = Column(Float, nullable=True)
    monthly_budget_impact = Column(Float, nullable=True)
    budget_category = Column(String(50), nullable=True, index=True)

    # Cost optimization
    cost_efficiency_score = Column(Float, nullable=True)  # Cost per successful outcome
    optimization_opportunity = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Indexes
    __table_args__ = (
        Index('ix_cost_breakdown_type_period', 'cost_type', 'billing_period'),
        Index('ix_cost_breakdown_provider_period', 'service_provider', 'billing_period'),
        Index('ix_cost_breakdown_agent_period', 'agent_type', 'billing_period'),
        Index('ix_cost_breakdown_call_time', 'call_id', 'cost_incurred_at'),
    )


class AlertHistory(Base):
    """Alert history and notification tracking"""
    __tablename__ = "alert_history"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(255), unique=True, nullable=False, index=True)

    # Alert details
    alert_type = Column(String(50), nullable=False, index=True)  # performance, cost, error, threshold
    alert_level = Column(String(20), nullable=False, index=True)  # info, warning, critical
    alert_title = Column(String(255), nullable=False)
    alert_message = Column(Text, nullable=False)

    # Context
    call_id = Column(String(255), nullable=True, index=True)
    trace_id = Column(String(255), nullable=True, index=True)
    metric_name = Column(String(100), nullable=True)
    threshold_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=True)

    # Alert state
    status = Column(String(20), nullable=False, index=True)  # active, acknowledged, resolved, ignored
    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Notification tracking
    notification_channels = Column(JSON, nullable=True)  # email, slack, webhook
    notifications_sent = Column(Integer, default=0)
    last_notification_at = Column(DateTime(timezone=True), nullable=True)

    # Alert rules
    rule_name = Column(String(100), nullable=True)
    rule_condition = Column(Text, nullable=True)
    alert_frequency = Column(String(20), nullable=True)  # immediate, hourly, daily

    # Resolution
    resolution_notes = Column(Text, nullable=True)
    auto_resolved = Column(Boolean, default=False)

    # Timestamps
    triggered_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Indexes
    __table_args__ = (
        Index('ix_alert_history_type_level', 'alert_type', 'alert_level'),
        Index('ix_alert_history_status_time', 'status', 'triggered_at'),
        Index('ix_alert_history_call_time', 'call_id', 'triggered_at'),
    )


# Monitoring utility functions

def create_trace_id() -> str:
    """Generate a unique trace ID for workflow tracking"""
    return f"trace_{uuid.uuid4().hex[:16]}_{int(datetime.now().timestamp())}"


def create_execution_id() -> str:
    """Generate a unique execution ID for agent tracking"""
    return f"exec_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"


def create_metric_id(category: str, name: str, timestamp: datetime) -> str:
    """Generate a unique metric ID"""
    ts = int(timestamp.timestamp())
    return f"metric_{category}_{name}_{ts}_{uuid.uuid4().hex[:8]}"


def create_cost_id(cost_type: str, provider: str, timestamp: datetime) -> str:
    """Generate a unique cost ID"""
    ts = int(timestamp.timestamp())
    return f"cost_{cost_type}_{provider}_{ts}_{uuid.uuid4().hex[:8]}"


def create_alert_id(alert_type: str, timestamp: datetime) -> str:
    """Generate a unique alert ID"""
    ts = int(timestamp.timestamp())
    return f"alert_{alert_type}_{ts}_{uuid.uuid4().hex[:8]}"
