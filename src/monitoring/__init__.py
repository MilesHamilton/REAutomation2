"""
REAutomation2 Monitoring System

Comprehensive monitoring and observability package for the REAutomation2 system.
Includes LangSmith integration, performance monitoring, alerting, and dashboard capabilities.
"""

from .langsmith_client import LangSmithClient, get_langsmith_client
from .models import (
    WorkflowStatus, AlertSeverity, AlertType, MetricType,
    WorkflowTrace, AgentExecution, PerformanceData,
    AlertRule, AlertNotification
)
from .tracing import (
    trace_workflow, trace_agent_execution,
    TracingContext, enable_orchestrator_tracing
)
from .integration import (
    enable_monitoring, disable_monitoring, get_monitoring_status
)
from .performance import (
    PerformanceMonitor, performance_monitor,
    monitor_performance, PerformanceTimer
)
from .alerts import (
    AlertManager, alert_manager,
    trigger_custom_alert, check_threshold_alert
)
from .dashboard import (
    DashboardManager, dashboard_manager,
    get_dashboard_html
)

__version__ = "0.1.0"
__author__ = "REAutomation2 Team"

# Public API
__all__ = [
    # Client and core
    "LangSmithClient",
    "get_langsmith_client",

    # Models and types
    "WorkflowStatus",
    "AlertSeverity",
    "AlertType",
    "MetricType",
    "WorkflowTrace",
    "AgentExecution",
    "PerformanceData",
    "AlertRule",
    "AlertNotification",

    # Tracing
    "trace_workflow",
    "trace_agent_execution",
    "TracingContext",
    "enable_orchestrator_tracing",

    # Integration
    "enable_monitoring",
    "disable_monitoring",
    "get_monitoring_status",

    # Performance monitoring
    "PerformanceMonitor",
    "performance_monitor",
    "monitor_performance",
    "PerformanceTimer",

    # Alerting
    "AlertManager",
    "alert_manager",
    "trigger_custom_alert",
    "check_threshold_alert",

    # Dashboard
    "DashboardManager",
    "dashboard_manager",
    "get_dashboard_html",
]

# Module metadata
MONITORING_FEATURES = [
    "LangSmith Integration",
    "Workflow Tracing",
    "Agent Execution Monitoring",
    "Performance Metrics Collection",
    "Real-time Alerting",
    "Cost Tracking",
    "Interactive Dashboard",
    "WebSocket Real-time Updates",
    "Circuit Breaker Patterns",
    "Fallback Logging"
]

def get_monitoring_info():
    """Get information about the monitoring system"""
    return {
        "version": __version__,
        "author": __author__,
        "features": MONITORING_FEATURES,
        "components": {
            "langsmith_client": "LangSmith API integration with circuit breaker",
            "tracing": "Decorator-based workflow and agent tracing",
            "performance": "System and application performance monitoring",
            "alerts": "Rule-based alerting with multiple notification channels",
            "dashboard": "Real-time monitoring dashboard with WebSocket updates",
            "integration": "Seamless integration with AgentOrchestrator"
        }
    }