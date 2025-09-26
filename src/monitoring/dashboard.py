"""
Basic monitoring dashboard for REAutomation2
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .performance import performance_monitor
from .alerts import alert_manager
from .langsmith_client import get_langsmith_client
from ..database.core import get_db
from ..database.monitoring_models import WorkflowTrace, AgentExecution, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class DashboardData:
    """Dashboard data structure"""
    system_health: Dict[str, Any]
    performance_summary: Dict[str, Any]
    active_alerts: List[Dict[str, Any]]
    recent_traces: List[Dict[str, Any]]
    cost_summary: Dict[str, Any]
    agent_metrics: Dict[str, Any]


class DashboardManager:
    """Manages dashboard data and real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_update: Optional[datetime] = None
        self.cached_data: Optional[DashboardData] = None
        self.cache_ttl = 30  # Cache for 30 seconds

    async def get_dashboard_data(self, force_refresh: bool = False) -> DashboardData:
        """Get comprehensive dashboard data"""
        current_time = datetime.utcnow()

        # Check cache
        if (not force_refresh and self.cached_data and self.last_update and
            (current_time - self.last_update).total_seconds() < self.cache_ttl):
            return self.cached_data

        try:
            # Gather all dashboard data
            dashboard_data = DashboardData(
                system_health=await self._get_system_health(),
                performance_summary=await self._get_performance_summary(),
                active_alerts=await self._get_active_alerts(),
                recent_traces=await self._get_recent_traces(),
                cost_summary=await self._get_cost_summary(),
                agent_metrics=await self._get_agent_metrics()
            )

            # Update cache
            self.cached_data = dashboard_data
            self.last_update = current_time

            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            # Return empty dashboard data on error
            return DashboardData(
                system_health={"status": "error", "message": str(e)},
                performance_summary={},
                active_alerts=[],
                recent_traces=[],
                cost_summary={},
                agent_metrics={}
            )

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            # Performance monitor status
            perf_stats = performance_monitor.get_current_stats()

            # LangSmith client status
            langsmith_client = get_langsmith_client()
            langsmith_status = langsmith_client.get_health_status()

            # Alert manager status
            alert_stats = alert_manager.get_alert_stats()

            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "performance_monitor": {
                    "enabled": perf_stats["enabled"],
                    "active_timers": perf_stats["active_timers"],
                    "metric_buffers": sum(perf_stats["metric_buffers"].values())
                },
                "langsmith": {
                    "enabled": langsmith_status["enabled"],
                    "healthy": langsmith_status["healthy"],
                    "last_error": langsmith_status.get("last_error")
                },
                "alerts": {
                    "total_active": alert_stats["total_active"],
                    "critical_count": alert_stats["by_severity"].get("critical", 0),
                    "warning_count": alert_stats["by_severity"].get("warning", 0)
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    async def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        try:
            perf_stats = performance_monitor.get_current_stats()

            # Calculate averages and totals
            total_calls = sum(perf_stats.get("call_counters", {}).values())
            total_errors = sum(perf_stats.get("error_counters", {}).values())
            error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0

            avg_response_times = perf_stats.get("avg_response_times", {})
            overall_avg_response = (
                sum(avg_response_times.values()) / len(avg_response_times)
                if avg_response_times else 0
            )

            return {
                "total_calls": total_calls,
                "total_errors": total_errors,
                "error_rate_percent": round(error_rate, 2),
                "avg_response_time_ms": round(overall_avg_response, 2),
                "response_times_by_operation": {
                    op: round(time, 2) for op, time in avg_response_times.items()
                },
                "call_counts_by_operation": perf_stats.get("call_counters", {}),
                "error_counts_by_type": perf_stats.get("error_counters", {})
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}

    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts for dashboard"""
        try:
            active_alerts = alert_manager.get_active_alerts()

            return [
                {
                    "id": alert.alert_id,
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "status": alert.status.value,
                    "call_id": alert.call_id,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "acknowledged": len(alert.acknowledgments) > 0
                }
                for alert in active_alerts
            ]

        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []

    async def _get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow traces"""
        try:
            async for db in get_db():
                traces = db.query(WorkflowTrace).order_by(
                    WorkflowTrace.started_at.desc()
                ).limit(limit).all()

                return [
                    {
                        "trace_id": trace.trace_id,
                        "call_id": trace.call_id,
                        "workflow_name": trace.workflow_name,
                        "status": trace.status,
                        "started_at": trace.started_at.isoformat() if trace.started_at else None,
                        "ended_at": trace.ended_at.isoformat() if trace.ended_at else None,
                        "duration_ms": trace.duration_ms,
                        "total_cost": float(trace.total_cost) if trace.total_cost else 0,
                        "agent_executions_count": trace.agent_executions_count or 0,
                        "error_message": trace.error_message
                    }
                    for trace in traces
                ]

        except Exception as e:
            logger.error(f"Error getting recent traces: {e}")
            return []

    async def _get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary for dashboard"""
        try:
            # Get cost data for last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)

            async for db in get_db():
                # Total cost from traces
                total_cost_result = db.query(
                    db.func.sum(WorkflowTrace.total_cost)
                ).filter(
                    WorkflowTrace.started_at >= since
                ).scalar()

                total_cost = float(total_cost_result) if total_cost_result else 0.0

                # Cost by workflow type
                cost_by_workflow = db.query(
                    WorkflowTrace.workflow_name,
                    db.func.sum(WorkflowTrace.total_cost).label('total')
                ).filter(
                    WorkflowTrace.started_at >= since
                ).group_by(WorkflowTrace.workflow_name).all()

                # Average cost per workflow
                avg_cost_result = db.query(
                    db.func.avg(WorkflowTrace.total_cost)
                ).filter(
                    WorkflowTrace.started_at >= since
                ).scalar()

                avg_cost = float(avg_cost_result) if avg_cost_result else 0.0

                return {
                    "total_cost_24h": round(total_cost, 4),
                    "avg_cost_per_workflow": round(avg_cost, 4),
                    "cost_by_workflow": {
                        workflow: round(float(cost), 4)
                        for workflow, cost in cost_by_workflow
                    },
                    "projected_monthly_cost": round(total_cost * 30, 2)
                }

        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {"error": str(e)}

    async def _get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        try:
            # Get agent execution data for last hour
            since = datetime.utcnow() - timedelta(hours=1)

            async for db in get_db():
                # Agent execution counts
                agent_counts = db.query(
                    AgentExecution.agent_type,
                    db.func.count(AgentExecution.id).label('count')
                ).filter(
                    AgentExecution.started_at >= since
                ).group_by(AgentExecution.agent_type).all()

                # Average execution times
                agent_avg_times = db.query(
                    AgentExecution.agent_type,
                    db.func.avg(AgentExecution.duration_ms).label('avg_duration')
                ).filter(
                    AgentExecution.started_at >= since,
                    AgentExecution.duration_ms.isnot(None)
                ).group_by(AgentExecution.agent_type).all()

                # Success rates
                agent_success_rates = db.query(
                    AgentExecution.agent_type,
                    db.func.sum(
                        db.case(
                            [(AgentExecution.status == 'completed', 1)],
                            else_=0
                        )
                    ).label('success_count'),
                    db.func.count(AgentExecution.id).label('total_count')
                ).filter(
                    AgentExecution.started_at >= since
                ).group_by(AgentExecution.agent_type).all()

                return {
                    "execution_counts": {
                        agent_type: count for agent_type, count in agent_counts
                    },
                    "avg_duration_ms": {
                        agent_type: round(float(avg_duration), 2)
                        for agent_type, avg_duration in agent_avg_times
                    },
                    "success_rates": {
                        agent_type: round((success_count / total_count) * 100, 1)
                        for agent_type, success_count, total_count in agent_success_rates
                        if total_count > 0
                    }
                }

        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return {"error": str(e)}

    # WebSocket management
    async def connect_websocket(self, websocket: WebSocket):
        """Connect new WebSocket client"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Dashboard WebSocket connected. Active connections: {len(self.active_connections)}")

    def disconnect_websocket(self, websocket: WebSocket):
        """Disconnect WebSocket client"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Dashboard WebSocket disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if not self.active_connections:
            return

        message = json.dumps(data)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect_websocket(connection)

    async def start_real_time_updates(self):
        """Start real-time dashboard updates"""
        while True:
            try:
                if self.active_connections:
                    dashboard_data = await self.get_dashboard_data(force_refresh=True)

                    # Convert to dict for JSON serialization
                    update_data = {
                        "type": "dashboard_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {
                            "system_health": dashboard_data.system_health,
                            "performance_summary": dashboard_data.performance_summary,
                            "active_alerts": dashboard_data.active_alerts,
                            "recent_traces": dashboard_data.recent_traces,
                            "cost_summary": dashboard_data.cost_summary,
                            "agent_metrics": dashboard_data.agent_metrics
                        }
                    }

                    await self.broadcast_update(update_data)

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Error in real-time updates: {e}")
                await asyncio.sleep(60)


# Global dashboard manager instance
dashboard_manager = DashboardManager()


# HTML Dashboard Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>REAutomation2 Monitoring Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .dashboard {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }

        .header h1 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .status-indicator {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
        }

        .status-healthy { background-color: #d4edda; color: #155724; }
        .status-warning { background-color: #fff3cd; color: #856404; }
        .status-error { background-color: #f8d7da; color: #721c24; }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .card h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 18px;
        }

        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }

        .metric:last-child {
            border-bottom: none;
        }

        .metric-label {
            font-weight: 500;
        }

        .metric-value {
            font-weight: 600;
            color: #2c3e50;
        }

        .alert {
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 10px;
            border-left: 4px solid;
        }

        .alert-critical {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }

        .alert-warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }

        .alert-info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }

        .trace {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .trace-success { border-left: 4px solid #28a745; }
        .trace-running { border-left: 4px solid #ffc107; }
        .trace-failed { border-left: 4px solid #dc3545; }

        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }

        .connected {
            background-color: #d4edda;
            color: #155724;
        }

        .disconnected {
            background-color: #f8d7da;
            color: #721c24;
        }

        .last-updated {
            font-size: 12px;
            color: #666;
            text-align: center;
            margin-top: 20px;
        }

        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>REAutomation2 Monitoring Dashboard</h1>
            <div id="system-status" class="status-indicator">Loading...</div>
            <div class="connection-status disconnected" id="connection-status">Connecting...</div>
        </div>

        <div class="grid">
            <!-- System Health -->
            <div class="card">
                <h2>System Health</h2>
                <div id="system-health">
                    <div class="metric">
                        <span class="metric-label">Status</span>
                        <span class="metric-value" id="health-status">Loading...</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Performance Monitor</span>
                        <span class="metric-value" id="perf-monitor-status">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">LangSmith</span>
                        <span class="metric-value" id="langsmith-status">-</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Timers</span>
                        <span class="metric-value" id="active-timers">0</span>
                    </div>
                </div>
            </div>

            <!-- Performance Summary -->
            <div class="card">
                <h2>Performance Summary</h2>
                <div id="performance-summary">
                    <div class="metric">
                        <span class="metric-label">Total Calls</span>
                        <span class="metric-value" id="total-calls">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Error Rate</span>
                        <span class="metric-value" id="error-rate">0%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Response Time</span>
                        <span class="metric-value" id="avg-response-time">0ms</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Errors</span>
                        <span class="metric-value" id="total-errors">0</span>
                    </div>
                </div>
            </div>

            <!-- Cost Summary -->
            <div class="card">
                <h2>Cost Summary (24h)</h2>
                <div id="cost-summary">
                    <div class="metric">
                        <span class="metric-label">Total Cost</span>
                        <span class="metric-value" id="total-cost">$0.00</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg per Workflow</span>
                        <span class="metric-value" id="avg-cost">$0.00</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Projected Monthly</span>
                        <span class="metric-value" id="projected-cost">$0.00</span>
                    </div>
                </div>
            </div>

            <!-- Agent Metrics -->
            <div class="card">
                <h2>Agent Performance (1h)</h2>
                <div id="agent-metrics">
                    <div class="metric">
                        <span class="metric-label">Executions</span>
                        <span class="metric-value" id="agent-executions">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Avg Duration</span>
                        <span class="metric-value" id="agent-avg-duration">0ms</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Success Rate</span>
                        <span class="metric-value" id="agent-success-rate">0%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Active Alerts -->
        <div class="card">
            <h2>Active Alerts</h2>
            <div id="active-alerts">
                <p>No active alerts</p>
            </div>
        </div>

        <!-- Recent Traces -->
        <div class="card">
            <h2>Recent Workflow Traces</h2>
            <div id="recent-traces">
                <p>No recent traces</p>
            </div>
        </div>

        <div class="last-updated" id="last-updated">
            Last updated: Never
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 10;

        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/monitoring/ws`;

            ws = new WebSocket(wsUrl);

            ws.onopen = function(event) {
                console.log('WebSocket connected');
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').className = 'connection-status connected';
                reconnectAttempts = 0;
            };

            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'dashboard_update') {
                    updateDashboard(message.data);
                }
            };

            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').className = 'connection-status disconnected';

                // Attempt to reconnect
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, 5000);
                }
            };

            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }

        function updateDashboard(data) {
            // Update system health
            if (data.system_health) {
                const health = data.system_health;
                document.getElementById('health-status').textContent = health.status;
                document.getElementById('system-status').textContent = health.status.toUpperCase();
                document.getElementById('system-status').className = `status-indicator status-${health.status}`;

                if (health.performance_monitor) {
                    document.getElementById('perf-monitor-status').textContent =
                        health.performance_monitor.enabled ? 'Enabled' : 'Disabled';
                    document.getElementById('active-timers').textContent =
                        health.performance_monitor.active_timers;
                }

                if (health.langsmith) {
                    document.getElementById('langsmith-status').textContent =
                        health.langsmith.enabled && health.langsmith.healthy ? 'Healthy' : 'Issues';
                }
            }

            // Update performance summary
            if (data.performance_summary) {
                const perf = data.performance_summary;
                document.getElementById('total-calls').textContent = perf.total_calls || 0;
                document.getElementById('error-rate').textContent = `${perf.error_rate_percent || 0}%`;
                document.getElementById('avg-response-time').textContent = `${perf.avg_response_time_ms || 0}ms`;
                document.getElementById('total-errors').textContent = perf.total_errors || 0;
            }

            // Update cost summary
            if (data.cost_summary) {
                const cost = data.cost_summary;
                document.getElementById('total-cost').textContent = `$${cost.total_cost_24h || 0}`;
                document.getElementById('avg-cost').textContent = `$${cost.avg_cost_per_workflow || 0}`;
                document.getElementById('projected-cost').textContent = `$${cost.projected_monthly_cost || 0}`;
            }

            // Update agent metrics
            if (data.agent_metrics) {
                const agents = data.agent_metrics;
                const totalExecutions = Object.values(agents.execution_counts || {}).reduce((a, b) => a + b, 0);
                const avgDurations = Object.values(agents.avg_duration_ms || {});
                const avgDuration = avgDurations.length ? avgDurations.reduce((a, b) => a + b, 0) / avgDurations.length : 0;
                const successRates = Object.values(agents.success_rates || {});
                const avgSuccessRate = successRates.length ? successRates.reduce((a, b) => a + b, 0) / successRates.length : 0;

                document.getElementById('agent-executions').textContent = totalExecutions;
                document.getElementById('agent-avg-duration').textContent = `${Math.round(avgDuration)}ms`;
                document.getElementById('agent-success-rate').textContent = `${Math.round(avgSuccessRate)}%`;
            }

            // Update active alerts
            if (data.active_alerts) {
                const alertsContainer = document.getElementById('active-alerts');
                if (data.active_alerts.length === 0) {
                    alertsContainer.innerHTML = '<p>No active alerts</p>';
                } else {
                    alertsContainer.innerHTML = data.active_alerts.map(alert => `
                        <div class="alert alert-${alert.severity}">
                            <strong>${alert.rule_name}</strong> - ${alert.message}
                            <br><small>Triggered: ${new Date(alert.triggered_at).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
            }

            // Update recent traces
            if (data.recent_traces) {
                const tracesContainer = document.getElementById('recent-traces');
                if (data.recent_traces.length === 0) {
                    tracesContainer.innerHTML = '<p>No recent traces</p>';
                } else {
                    tracesContainer.innerHTML = data.recent_traces.map(trace => `
                        <div class="trace trace-${trace.status}">
                            <strong>${trace.workflow_name}</strong> (${trace.trace_id})
                            <br>Status: ${trace.status} | Duration: ${trace.duration_ms || 0}ms | Cost: $${trace.total_cost || 0}
                            <br><small>${new Date(trace.started_at).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
            }

            // Update timestamp
            document.getElementById('last-updated').textContent =
                `Last updated: ${new Date().toLocaleString()}`;
        }

        // Initialize
        connectWebSocket();

        // Fallback: refresh every 60 seconds if WebSocket isn't working
        setInterval(function() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                window.location.reload();
            }
        }, 60000);
    </script>
</body>
</html>
"""


def get_dashboard_html() -> str:
    """Get the dashboard HTML"""
    return DASHBOARD_HTML