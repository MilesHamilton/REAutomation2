from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

Base = declarative_base()


class CallRecord(Base):
    """Call records table for storing call information and metrics"""
    __tablename__ = "calls"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String(255), unique=True, nullable=False, index=True)
    phone_number = Column(String(20), nullable=False, index=True)

    # Call status and timing
    status = Column(String(50), nullable=False, index=True)  # ringing, connected, completed, failed
    started_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Lead information
    lead_data = Column(JSON, nullable=True)
    qualification_score = Column(Float, nullable=True, index=True)
    qualified = Column(Boolean, default=False, index=True)

    # TTS tier information
    initial_tier = Column(String(50), nullable=False)  # local_piper, elevenlabs
    final_tier = Column(String(50), nullable=True)
    tier_switches = Column(Integer, default=0)
    escalation_trigger = Column(String(100), nullable=True)

    # Cost and performance metrics
    total_cost = Column(Float, default=0.0)
    llm_cost = Column(Float, default=0.0)
    tts_cost = Column(Float, default=0.0)
    stt_cost = Column(Float, default=0.0)

    # Performance metrics
    avg_response_time_ms = Column(Float, nullable=True)
    llm_latency_ms = Column(Float, nullable=True)
    tts_latency_ms = Column(Float, nullable=True)
    stt_latency_ms = Column(Float, nullable=True)

    # Error information
    error_occurred = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True)

    # Conversation summary
    conversation_summary = Column(Text, nullable=True)
    total_messages = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    conversations = relationship("ConversationHistory", back_populates="call")
    tier_switches_history = relationship("TierSwitchHistory", back_populates="call")

    # Indexes for performance
    __table_args__ = (
        Index('ix_calls_status_created', 'status', 'created_at'),
        Index('ix_calls_qualified_created', 'qualified', 'created_at'),
        Index('ix_calls_phone_created', 'phone_number', 'created_at'),
        Index('ix_calls_cost_created', 'total_cost', 'created_at'),
    )


class ConversationHistory(Base):
    """Conversation history for calls"""
    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String(255), ForeignKey("calls.call_id"), nullable=False, index=True)

    # Message details
    message_order = Column(Integer, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)

    # Processing metrics
    processing_time_ms = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)

    # Agent information
    agent_type = Column(String(50), nullable=True)  # conversation, qualification, objection_handler
    agent_state = Column(JSON, nullable=True)

    # Cost breakdown
    llm_tokens_used = Column(Integer, nullable=True)
    tts_characters = Column(Integer, nullable=True)
    processing_cost = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    call = relationship("CallRecord", back_populates="conversations")

    # Indexes
    __table_args__ = (
        Index('ix_conv_call_order', 'call_id', 'message_order'),
        Index('ix_conv_call_role', 'call_id', 'role'),
    )


class TierSwitchHistory(Base):
    """History of tier switches during calls"""
    __tablename__ = "tier_switches"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String(255), ForeignKey("calls.call_id"), nullable=False, index=True)

    # Switch details
    from_tier = Column(String(50), nullable=False)
    to_tier = Column(String(50), nullable=False)
    trigger = Column(String(100), nullable=False)  # qualification, manual, cost_control
    qualification_score_at_switch = Column(Float, nullable=True)

    # Timing
    switched_at = Column(DateTime(timezone=True), server_default=func.now())

    # Cost impact
    cost_before_switch = Column(Float, nullable=True)
    estimated_cost_savings = Column(Float, nullable=True)

    # Relationships
    call = relationship("CallRecord", back_populates="tier_switches_history")


class ContactRecord(Base):
    """Contact information and lead data"""
    __tablename__ = "contacts"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, nullable=False, index=True)

    # Contact details
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)

    # Lead scoring
    lead_score = Column(Float, nullable=True, index=True)
    lead_source = Column(String(100), nullable=True)
    lead_status = Column(String(50), nullable=True, index=True)

    # Contact preferences
    preferred_contact_time = Column(String(50), nullable=True)
    timezone = Column(String(50), nullable=True)
    do_not_call = Column(Boolean, default=False, index=True)

    # Additional data
    custom_fields = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)

    # Campaign tracking
    campaign_id = Column(String(100), nullable=True, index=True)
    campaign_name = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_contacted = Column(DateTime(timezone=True), nullable=True)

    # Call history count (denormalized for performance)
    total_calls = Column(Integer, default=0)
    successful_calls = Column(Integer, default=0)
    qualified_calls = Column(Integer, default=0)


class SystemMetrics(Base):
    """System performance and usage metrics"""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)

    # Metric details
    metric_type = Column(String(50), nullable=False, index=True)  # daily, hourly, call
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)

    # Dimensions
    call_id = Column(String(255), nullable=True, index=True)
    phone_number = Column(String(20), nullable=True)
    agent_type = Column(String(50), nullable=True)
    tier = Column(String(50), nullable=True)

    # Additional context
    meta_data = Column(JSON, nullable=True)

    # Timestamps
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Time-series indexes
    __table_args__ = (
        Index('ix_metrics_type_time', 'metric_type', 'recorded_at'),
        Index('ix_metrics_name_time', 'metric_name', 'recorded_at'),
        Index('ix_metrics_call_time', 'call_id', 'recorded_at'),
    )


class CostTracking(Base):
    """Detailed cost tracking and budget management"""
    __tablename__ = "cost_tracking"

    id = Column(Integer, primary_key=True, index=True)

    # Cost details
    cost_type = Column(String(50), nullable=False, index=True)  # llm, tts, stt, twilio
    cost_amount = Column(Float, nullable=False)
    cost_currency = Column(String(3), default='USD')

    # Resource usage
    units_consumed = Column(Float, nullable=True)
    unit_type = Column(String(50), nullable=True)  # tokens, characters, minutes
    unit_cost = Column(Float, nullable=True)

    # Context
    call_id = Column(String(255), nullable=True, index=True)
    service_provider = Column(String(100), nullable=True)
    tier = Column(String(50), nullable=True, index=True)

    # Budget tracking
    daily_date = Column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    monthly_period = Column(String(7), nullable=False, index=True)  # YYYY-MM

    # Timestamps
    incurred_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Cost optimization indexes
    __table_args__ = (
        Index('ix_cost_type_date', 'cost_type', 'daily_date'),
        Index('ix_cost_tier_date', 'tier', 'daily_date'),
        Index('ix_cost_call_type', 'call_id', 'cost_type'),
    )


class ScheduledCalls(Base):
    """Scheduled outbound calls queue"""
    __tablename__ = "scheduled_calls"

    id = Column(Integer, primary_key=True, index=True)

    # Call details
    phone_number = Column(String(20), nullable=False, index=True)
    lead_data = Column(JSON, nullable=True)
    priority = Column(Integer, default=1, index=True)

    # Scheduling
    scheduled_for = Column(DateTime(timezone=True), nullable=False, index=True)
    timezone = Column(String(50), nullable=True)

    # Campaign info
    campaign_id = Column(String(100), nullable=True, index=True)
    campaign_name = Column(String(255), nullable=True)

    # Status tracking
    status = Column(String(50), default='scheduled', nullable=False, index=True)  # scheduled, in_progress, completed, failed, cancelled
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    # Retry logic
    next_retry_at = Column(DateTime(timezone=True), nullable=True, index=True)
    retry_reason = Column(String(255), nullable=True)

    # Result tracking
    call_id = Column(String(255), nullable=True)  # Set when call is actually made
    completion_status = Column(String(50), nullable=True)
    qualification_result = Column(Boolean, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Scheduling indexes
    __table_args__ = (
        Index('ix_scheduled_status_time', 'status', 'scheduled_for'),
        Index('ix_scheduled_priority_time', 'priority', 'scheduled_for'),
        Index('ix_scheduled_retry_time', 'next_retry_at'),
        Index('ix_scheduled_campaign_status', 'campaign_id', 'status'),
    )


# Utility functions for model operations
def create_call_id() -> str:
    """Generate a unique call ID"""
    return f"call_{uuid.uuid4().hex[:12]}_{int(datetime.now().timestamp())}"


def get_daily_key(date: Optional[datetime] = None) -> str:
    """Get daily key for cost tracking"""
    if date is None:
        date = datetime.now()
    return date.strftime("%Y-%m-%d")


def get_monthly_key(date: Optional[datetime] = None) -> str:
    """Get monthly key for cost tracking"""
    if date is None:
        date = datetime.now()
    return date.strftime("%Y-%m")
