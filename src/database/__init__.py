"""
Database module for REAutomation2

This module provides database models, connections, and repositories
for managing call data, conversation history, cost tracking, and more.
"""

from .connection import db_manager, get_database_session
from .models import (
    Base, CallRecord, ConversationHistory, TierSwitchHistory,
    ContactRecord, SystemMetrics, CostTracking, ScheduledCalls,
    create_call_id, get_daily_key, get_monthly_key
)
from .repositories import (
    BaseRepository, CallRepository, ConversationRepository,
    ContactRepository, CostTrackingRepository, ScheduledCallRepository,
    MetricsRepository
)

__all__ = [
    # Connection management
    "db_manager",
    "get_database_session",

    # Models
    "Base",
    "CallRecord",
    "ConversationHistory",
    "TierSwitchHistory",
    "ContactRecord",
    "SystemMetrics",
    "CostTracking",
    "ScheduledCalls",
    "create_call_id",
    "get_daily_key",
    "get_monthly_key",

    # Repositories
    "BaseRepository",
    "CallRepository",
    "ConversationRepository",
    "ContactRepository",
    "CostTrackingRepository",
    "ScheduledCallRepository",
    "MetricsRepository",
]