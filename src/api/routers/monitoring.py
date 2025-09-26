from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
import logging

from ...monitoring.models import (
    MonitoringDashboardRequest,
    MonitoringDashboardResponse,
    TraceQueryRequest,
    TraceQueryResponse,
    WorkflowTrace,
    AgentExecution,
    PerformanceMetric,
    CostBreakdown,
    Alert,
    SystemHealthStatus,
    WorkflowAnalytics,
    MetricSummary
)
from ...monitoring.langsmith_client import get_langsmith_client
from ...monitoring.integration import get_monitoring_status
from ...monitoring.dashboard import dashboard_manager, get_dashboard_html
from ...agents.orchestrator import agent_orchestrator

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
logger = logging.getLogger(__name__)


# WebSocket connections for real-time monitoring
active_websocket_connections: List[WebSocket] = []


async def get_current_user():
    """Placeholder for authentication - implement based on your auth system"""
    # In a real implementation, you'd validate JWT tokens or API keys here
    return {"user_id": "monitoring_user", "role": "admin"}


# Dashboard Endpoints
@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the monitoring dashboard HTML"""
    return get_dashboard_html()


@router.get("/dashboard/data", response_model=Dict[str, Any])
async def get_dashboard_data():
    """Get dashboard data as JSON"""
    try:
        dashboard_data = await dashboard_manager.get_dashboard_data()

        return {
            "system_health": dashboard_data.system_health,
            "performance_summary": dashboard_data.performance_summary,
            "active_alerts": dashboard_data.active_alerts,
            "recent_traces": dashboard_data.recent_traces,
            "cost_summary": dashboard_data.cost_summary,
            "agent_metrics": dashboard_data.agent_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")


# Health and Status Endpoints
@router.get("/health", response_model=Dict[str, Any])
async def get_monitoring_health():
    """Get monitoring system health status"""
    try:
        langsmith_client = get_langsmith_client()
        monitoring_status = get_monitoring_status()
        orchestrator_health = await agent_orchestrator.health_check()

        health_status = {
            "status": "healthy" if monitoring_status.get("enabled") else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "langsmith": {
                "enabled": monitoring_status.get("enabled", False),
                "client_health": langsmith_client.get_health_status(),
                "connection_status": "connected" if langsmith_client.enabled else "disabled"
            },
            "orchestrator": orchestrator_health,
            "monitoring": monitoring_status
        }

        return health_status

    except Exception as e:
        logger.error(f"Error getting monitoring health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/status", response_model=SystemHealthStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        langsmith_client = get_langsmith_client()

        # Get orchestrator health
        orchestrator_health = await agent_orchestrator.health_check()

        # Calculate metrics (simplified - in real implementation, query database)
        active_workflows = orchestrator_health.get("active_workflows", 0)

        # Mock some metrics for demonstration
        system_status = SystemHealthStatus(
            overall_status="healthy" if langsmith_client.enabled else "degraded",
            langsmith_client_status=langsmith_client.get_health_status(),
            active_alerts_count=0,  # Would query from database
            critical_alerts_count=0,
            recent_error_rate=0.02,  # 2% error rate
            average_response_time_ms=245.0,
            cost_burn_rate=0.05,  # $0.05 per hour
            daily_budget_utilization=0.25,  # 25% of daily budget used
            last_updated=datetime.utcnow()
        )

        return system_status

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"System status failed: {str(e)}")


# Dashboard Endpoints
@router.post("/dashboard", response_model=MonitoringDashboardResponse)
async def get_dashboard_data(
    request: MonitoringDashboardRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive dashboard data"""
    try:
        # In a real implementation, this would query the database
        # For now, return mock data structure

        system_health = SystemHealthStatus(
            overall_status="healthy",
            langsmith_client_status=get_langsmith_client().get_health_status(),
            active_alerts_count=2,
            critical_alerts_count=0,
            recent_error_rate=0.01,
            average_response_time_ms=342.5,
            cost_burn_rate=0.08,
            daily_budget_utilization=0.42,
            last_updated=datetime.utcnow()
        )

        # Mock metric summaries
        metric_summaries = [
            MetricSummary(
                metric_name="response_time",
                metric_category="response_time",
                total_data_points=1543,
                time_range_start=request.time_range_start,
                time_range_end=request.time_range_end,
                current_value=342.5,
                average_value=298.3,
                min_value=125.0,
                max_value=1240.0,
                percentile_95=856.2,
                trend_direction="stable",
                trend_percentage=2.3
            ),
            MetricSummary(
                metric_name="cost_per_call",
                metric_category="cost",
                total_data_points=1543,
                time_range_start=request.time_range_start,
                time_range_end=request.time_range_end,
                current_value=0.085,
                average_value=0.078,
                min_value=0.045,
                max_value=0.156,
                percentile_95=0.124,
                trend_direction="down",
                trend_percentage=-5.2
            )
        ]

        # Mock workflow analytics
        workflow_analytics = [
            WorkflowAnalytics(
                workflow_name="agent_orchestration",
                total_executions=1543,
                success_rate=0.987,
                average_duration_ms=2340.5,
                average_cost=0.078,
                total_cost=120.35,
                most_common_failure_type="timeout",
                agent_performance=[
                    {"agent": "conversation", "avg_time": 145.2, "success_rate": 0.995},
                    {"agent": "qualification", "avg_time": 234.1, "success_rate": 0.978},
                    {"agent": "scheduler", "avg_time": 156.8, "success_rate": 0.989}
                ],
                time_range_start=request.time_range_start,
                time_range_end=request.time_range_end
            )
        ]

        cost_summary = {
            "total_cost": 120.35,
            "llm_cost": 85.42,
            "tts_cost": 23.15,
            "api_cost": 11.78,
            "daily_budget_remaining": 28.90
        }

        dashboard_response = MonitoringDashboardResponse(
            system_health=system_health,
            metric_summaries=metric_summaries,
            workflow_analytics=workflow_analytics,
            recent_alerts=[],  # Would query from database
            cost_summary=cost_summary,
            time_range={
                "start": request.time_range_start,
                "end": request.time_range_end
            }
        )

        return dashboard_response

    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")


# Trace and Workflow Endpoints
@router.post("/traces", response_model=TraceQueryResponse)
async def query_traces(
    request: TraceQueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Query workflow traces with filtering"""
    try:
        # In real implementation, this would query the database
        # Mock response for now

        traces = []
        # Add mock traces if needed for testing

        response = TraceQueryResponse(
            traces=traces,
            total_count=len(traces),
            has_more=False,
            query_time_ms=45.2
        )

        return response

    except Exception as e:
        logger.error(f"Error querying traces: {e}")
        raise HTTPException(status_code=500, detail=f"Trace query failed: {str(e)}")


@router.get("/traces/{trace_id}", response_model=WorkflowTrace)
async def get_trace_by_id(
    trace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get specific workflow trace by ID"""
    try:
        # In real implementation, query database for trace
        # Mock for now
        raise HTTPException(status_code=404, detail="Trace not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Get trace failed: {str(e)}")


@router.get("/traces/{trace_id}/executions", response_model=List[AgentExecution])
async def get_trace_executions(
    trace_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get agent executions for a specific trace"""
    try:
        # In real implementation, query database for executions
        executions = []
        return executions

    except Exception as e:
        logger.error(f"Error getting executions for trace {trace_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Get executions failed: {str(e)}")


# Metrics Endpoints
@router.get("/metrics", response_model=List[PerformanceMetric])
async def get_metrics(
    metric_category: Optional[str] = Query(None, description="Filter by metric category"),
    metric_name: Optional[str] = Query(None, description="Filter by metric name"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    current_user: dict = Depends(get_current_user)
):
    """Get performance metrics with filtering"""
    try:
        # In real implementation, query database with filters
        metrics = []
        return metrics

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Get metrics failed: {str(e)}")


@router.get("/metrics/summary")
async def get_metrics_summary(
    time_window: str = Query("1h", description="Time window: 5m, 15m, 1h, 6h, 24h"),
    current_user: dict = Depends(get_current_user)
):
    """Get metrics summary for specified time window"""
    try:
        # Parse time window
        time_delta_map = {
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24)
        }

        if time_window not in time_delta_map:
            raise HTTPException(status_code=400, detail="Invalid time window")

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - time_delta_map[time_window]

        # Mock summary data
        summary = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "window": time_window
            },
            "metrics": {
                "total_calls": 247,
                "avg_response_time_ms": 342.5,
                "success_rate": 0.987,
                "error_rate": 0.013,
                "total_cost": 19.43,
                "avg_cost_per_call": 0.079
            },
            "alerts": {
                "active": 2,
                "critical": 0,
                "warning": 2
            }
        }

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics summary failed: {str(e)}")


# Cost Endpoints
@router.get("/costs", response_model=List[CostBreakdown])
async def get_costs(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    cost_type: Optional[str] = Query(None, description="Filter by cost type"),
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    limit: int = Query(100, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """Get cost breakdown data"""
    try:
        # In real implementation, query database with filters
        costs = []
        return costs

    except Exception as e:
        logger.error(f"Error getting costs: {e}")
        raise HTTPException(status_code=500, detail=f"Get costs failed: {str(e)}")


@router.get("/costs/summary")
async def get_cost_summary(
    period: str = Query("today", description="Period: today, week, month"),
    current_user: dict = Depends(get_current_user)
):
    """Get cost summary for specified period"""
    try:
        # Mock cost summary
        summary = {
            "period": period,
            "total_cost": 45.67,
            "breakdown": {
                "llm": 32.15,
                "tts": 8.94,
                "stt": 2.33,
                "api": 2.25
            },
            "daily_budget": 50.0,
            "budget_used_percentage": 91.34,
            "projection": {
                "daily": 45.67,
                "monthly": 1370.1
            },
            "top_cost_drivers": [
                {"name": "conversation_agent", "cost": 18.45},
                {"name": "qualification_agent", "cost": 12.33},
                {"name": "elevenlabs_tts", "cost": 8.94}
            ]
        }

        return summary

    except Exception as e:
        logger.error(f"Error getting cost summary: {e}")
        raise HTTPException(status_code=500, detail=f"Cost summary failed: {str(e)}")


# Alert Endpoints
@router.get("/alerts", response_model=List[Alert])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by alert status"),
    level: Optional[str] = Query(None, description="Filter by alert level"),
    limit: int = Query(100, ge=1, le=500),
    current_user: dict = Depends(get_current_user)
):
    """Get alerts with filtering"""
    try:
        # In real implementation, query database
        alerts = []
        return alerts

    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Get alerts failed: {str(e)}")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    notes: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge an alert"""
    try:
        # In real implementation, update alert in database
        return {
            "alert_id": alert_id,
            "status": "acknowledged",
            "acknowledged_by": current_user["user_id"],
            "acknowledged_at": datetime.utcnow().isoformat(),
            "notes": notes
        }

    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Acknowledge alert failed: {str(e)}")


# Real-time WebSocket Endpoint
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring data"""
    await dashboard_manager.connect_websocket(websocket)

    try:
        # Send initial dashboard data
        dashboard_data = await dashboard_manager.get_dashboard_data()
        initial_data = {
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
        await websocket.send_json(initial_data)

        # Keep connection alive and send heartbeats
        while True:
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
            heartbeat = {
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send_json(heartbeat)

    except WebSocketDisconnect:
        dashboard_manager.disconnect_websocket(websocket)
        if websocket in active_websocket_connections:
            active_websocket_connections.remove(websocket)
        logger.info("Dashboard WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_websocket_connections:
            active_websocket_connections.remove(websocket)


# Utility function to broadcast updates to all connected WebSockets
async def broadcast_update(update_data: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients"""
    if not active_websocket_connections:
        return

    disconnected_connections = []

    for websocket in active_websocket_connections:
        try:
            await websocket.send_json(update_data)
        except Exception as e:
            logger.error(f"Error broadcasting to WebSocket: {e}")
            disconnected_connections.append(websocket)

    # Remove disconnected connections
    for websocket in disconnected_connections:
        active_websocket_connections.remove(websocket)


# Export Data Endpoints
@router.get("/export/traces")
async def export_traces(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    format: str = Query("json", description="Export format: json, csv"),
    current_user: dict = Depends(get_current_user)
):
    """Export trace data"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Invalid export format")

        # In real implementation, query database and format data
        if format == "json":
            def generate_json():
                yield '{"traces": ['
                # Stream trace data
                yield ']}'

            return StreamingResponse(
                generate_json(),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=traces.json"}
            )

        # CSV format would be implemented similarly
        return {"message": "CSV export not implemented yet"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting traces: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/export/metrics")
async def export_metrics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    format: str = Query("json", description="Export format: json, csv"),
    current_user: dict = Depends(get_current_user)
):
    """Export metrics data"""
    try:
        if format not in ["json", "csv"]:
            raise HTTPException(status_code=400, detail="Invalid export format")

        # Mock export for now
        return {"message": f"Metrics export in {format} format - implementation pending"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Configuration Endpoints
@router.get("/config")
async def get_monitoring_config(current_user: dict = Depends(get_current_user)):
    """Get current monitoring configuration"""
    try:
        from ...config.settings import settings

        config = {
            "langsmith": {
                "enabled": settings.langsmith_enabled,
                "project": settings.langsmith_project,
                "endpoint": settings.langsmith_endpoint,
                "batch_size": settings.langsmith_batch_size,
                "flush_interval": settings.langsmith_flush_interval,
                "fallback_enabled": settings.langsmith_fallback_enabled
            },
            "monitoring_status": get_monitoring_status()
        }

        return config

    except Exception as e:
        logger.error(f"Error getting monitoring config: {e}")
        raise HTTPException(status_code=500, detail=f"Get config failed: {str(e)}")


@router.post("/flush")
async def flush_monitoring_data(current_user: dict = Depends(get_current_user)):
    """Manually flush pending monitoring data"""
    try:
        langsmith_client = get_langsmith_client()
        await langsmith_client.flush_batch()

        return {
            "status": "success",
            "message": "Monitoring data flushed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error flushing monitoring data: {e}")
        raise HTTPException(status_code=500, detail=f"Flush failed: {str(e)}")