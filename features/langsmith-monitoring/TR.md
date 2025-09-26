# Technical Requirements (TR)

# LangSmith Monitoring Integration

## Document Information

- **Feature Name**: LangSmith Monitoring Integration
- **Version**: 1.0
- **Date**: January 26, 2025
- **Author**: Development Team
- **Status**: Planning

## Technical Overview

This document defines the technical implementation requirements for integrating LangSmith monitoring into the REAutomation2 system. It covers architecture decisions, implementation details, API specifications, and technical constraints.

## Architecture Design

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph     │    │   LangSmith     │
│   Application   │◄──►│   Orchestrator  │◄──►│   Cloud Service │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Trace         │    │   Analytics     │
│   Dashboard     │◄──►│   Collector     │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Alert         │    │   Cost          │    │   Redis         │
│   Manager       │    │   Tracker       │    │   Cache         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

#### Core Components

1. **LangSmith Client Manager**

   - Handles LangSmith API connections
   - Manages authentication and rate limiting
   - Provides fallback mechanisms

2. **Trace Collector Service**

   - Captures workflow execution data
   - Formats data for LangSmith ingestion
   - Handles local buffering and retry logic

3. **Monitoring Dashboard Service**

   - Serves real-time monitoring interfaces
   - Aggregates data from multiple sources
   - Provides WebSocket connections for live updates

4. **Alert Manager**

   - Processes monitoring data for alert conditions
   - Manages alert routing and escalation
   - Integrates with notification systems

5. **Cost Attribution Engine**
   - Tracks resource usage and costs
   - Integrates with existing cost controls
   - Provides cost optimization insights

## Implementation Specifications

### TR-001: LangSmith Integration Layer

#### TR-001.1: Client Configuration

```python
# Configuration class for LangSmith integration
class LangSmithConfig:
    api_key: str
    project_name: str = "reautomation2-monitoring"
    endpoint: str = "https://api.smith.langchain.com"
    timeout: int = 30
    retry_attempts: int = 3
    batch_size: int = 100
    flush_interval: int = 5  # seconds
    fallback_enabled: bool = True
```

#### TR-001.2: Instrumentation Decorator

```python
# Decorator for automatic LangGraph instrumentation
@langsmith_trace
class AgentOrchestrator:
    def __init__(self):
        self.tracer = LangSmithTracer()

    @trace_workflow
    async def process_input(self, call_id: str, user_input: str):
        # Existing implementation with automatic tracing
        pass
```

#### TR-001.3: Trace Data Model

```python
# Data model for trace information
class WorkflowTrace(BaseModel):
    trace_id: str
    call_id: str
    workflow_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: TraceStatus
    agent_transitions: List[AgentTransition]
    performance_metrics: PerformanceMetrics
    cost_data: CostData
    error_info: Optional[ErrorInfo]
```

### TR-002: Performance Monitoring System

#### TR-002.1: Metrics Collection

```python
# Performance metrics collector
class PerformanceCollector:
    def __init__(self):
        self.metrics_buffer = []
        self.redis_client = Redis()

    async def collect_agent_metrics(self, agent_type: AgentType,
                                  execution_time: float,
                                  success: bool):
        metric = AgentMetric(
            agent_type=agent_type,
            execution_time=execution_time,
            timestamp=time.time(),
            success=success
        )
        await self._buffer_metric(metric)

    async def collect_workflow_metrics(self, workflow_trace: WorkflowTrace):
        # Aggregate workflow-level metrics
        pass
```

#### TR-002.2: Real-time Monitoring

```python
# WebSocket handler for real-time monitoring
class MonitoringWebSocket:
    def __init__(self):
        self.active_connections = set()

    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        for connection in self.active_connections:
            await connection.send_json(metrics)

    async def handle_connection(self, websocket: WebSocket):
        # Handle real-time monitoring connections
        pass
```

### TR-003: Cost Tracking Integration

#### TR-003.1: Cost Calculator Enhancement

```python
# Enhanced cost calculator with LangSmith integration
class EnhancedCostCalculator:
    def __init__(self, langsmith_client: LangSmithClient):
        self.langsmith_client = langsmith_client
        self.cost_tracker = CostTracker()

    async def calculate_workflow_cost(self, trace: WorkflowTrace) -> CostBreakdown:
        # Calculate costs with detailed attribution
        llm_cost = self._calculate_llm_cost(trace.agent_transitions)
        tts_cost = self._calculate_tts_cost(trace.voice_usage)
        api_cost = self._calculate_api_cost(trace.external_calls)

        return CostBreakdown(
            total_cost=llm_cost + tts_cost + api_cost,
            llm_cost=llm_cost,
            tts_cost=tts_cost,
            api_cost=api_cost,
            cost_per_agent=self._calculate_agent_costs(trace)
        )
```

#### TR-003.2: Budget Alert Integration

```python
# Budget alert system integration
class BudgetAlertManager:
    def __init__(self, cost_controller: CostController):
        self.cost_controller = cost_controller
        self.alert_thresholds = {
            'daily_80_percent': 0.8,
            'call_limit_exceeded': 0.10,
            'unusual_spike': 2.0  # 2x normal cost
        }

    async def check_cost_alerts(self, cost_data: CostData):
        # Check various cost alert conditions
        pass
```

### TR-004: Analytics and Reporting

#### TR-004.1: Data Aggregation Service

```python
# Analytics data aggregation
class AnalyticsAggregator:
    def __init__(self):
        self.db_session = get_db_session()
        self.redis_client = Redis()

    async def aggregate_conversation_metrics(self,
                                           time_period: TimePeriod) -> ConversationMetrics:
        # Aggregate conversation flow data
        qualification_rates = await self._calculate_qualification_rates(time_period)
        objection_handling = await self._analyze_objection_handling(time_period)
        tier_escalations = await self._analyze_tier_escalations(time_period)

        return ConversationMetrics(
            qualification_success_rate=qualification_rates,
            objection_handling_effectiveness=objection_handling,
            tier_escalation_patterns=tier_escalations
        )
```

#### TR-004.2: Dashboard API Endpoints

```python
# FastAPI endpoints for monitoring dashboard
@router.get("/monitoring/dashboard")
async def get_dashboard_data(time_range: str = "1h"):
    # Return dashboard data
    pass

@router.get("/monitoring/traces/{call_id}")
async def get_call_trace(call_id: str):
    # Return detailed trace for specific call
    pass

@router.get("/monitoring/performance")
async def get_performance_metrics(agent_type: Optional[AgentType] = None):
    # Return performance metrics
    pass

@router.get("/monitoring/costs")
async def get_cost_analysis(time_range: str = "24h"):
    # Return cost analysis data
    pass
```

### TR-005: Alert and Notification System

#### TR-005.1: Alert Processing Engine

```python
# Alert processing and routing
class AlertProcessor:
    def __init__(self):
        self.alert_rules = self._load_alert_rules()
        self.notification_channels = self._setup_notification_channels()

    async def process_metrics(self, metrics: Dict[str, Any]):
        for rule in self.alert_rules:
            if rule.evaluate(metrics):
                alert = Alert(
                    rule_id=rule.id,
                    severity=rule.severity,
                    message=rule.generate_message(metrics),
                    timestamp=datetime.utcnow()
                )
                await self._send_alert(alert)
```

#### TR-005.2: Notification Channels

```python
# Notification channel implementations
class NotificationChannel(ABC):
    @abstractmethod
    async def send_notification(self, alert: Alert):
        pass

class EmailNotificationChannel(NotificationChannel):
    async def send_notification(self, alert: Alert):
        # Send email notification
        pass

class SlackNotificationChannel(NotificationChannel):
    async def send_notification(self, alert: Alert):
        # Send Slack notification
        pass
```

## Database Schema Extensions

### Monitoring Tables

```sql
-- Workflow execution traces
CREATE TABLE workflow_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id VARCHAR(255) UNIQUE NOT NULL,
    call_id VARCHAR(255) NOT NULL,
    workflow_name VARCHAR(100) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) NOT NULL,
    total_cost DECIMAL(10,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent execution details
CREATE TABLE agent_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id VARCHAR(255) REFERENCES workflow_traces(trace_id),
    agent_type VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    execution_time_ms INTEGER,
    success BOOLEAN NOT NULL,
    input_data JSONB,
    output_data JSONB,
    cost DECIMAL(10,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id VARCHAR(255) REFERENCES workflow_traces(trace_id),
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metadata JSONB
);

-- Cost tracking details
CREATE TABLE cost_details (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id VARCHAR(255) REFERENCES workflow_traces(trace_id),
    cost_type VARCHAR(50) NOT NULL, -- 'llm', 'tts', 'api'
    cost_amount DECIMAL(10,4) NOT NULL,
    usage_units INTEGER,
    unit_cost DECIMAL(10,6),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Alert history
CREATE TABLE alert_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);
```

### Indexes for Performance

```sql
-- Indexes for efficient querying
CREATE INDEX idx_workflow_traces_call_id ON workflow_traces(call_id);
CREATE INDEX idx_workflow_traces_start_time ON workflow_traces(start_time);
CREATE INDEX idx_agent_executions_trace_id ON agent_executions(trace_id);
CREATE INDEX idx_agent_executions_agent_type ON agent_executions(agent_type);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_cost_details_timestamp ON cost_details(timestamp);
CREATE INDEX idx_alert_history_triggered_at ON alert_history(triggered_at);
```

## API Specifications

### LangSmith Integration API

```python
# LangSmith client interface
class LangSmithClient:
    def __init__(self, config: LangSmithConfig):
        self.config = config
        self.session = aiohttp.ClientSession()

    async def create_trace(self, trace_data: WorkflowTrace) -> str:
        """Create a new trace in LangSmith"""
        pass

    async def update_trace(self, trace_id: str, updates: Dict[str, Any]):
        """Update existing trace with new data"""
        pass

    async def add_trace_step(self, trace_id: str, step_data: AgentTransition):
        """Add a step to an existing trace"""
        pass

    async def get_trace(self, trace_id: str) -> WorkflowTrace:
        """Retrieve trace data from LangSmith"""
        pass

    async def query_traces(self, filters: Dict[str, Any]) -> List[WorkflowTrace]:
        """Query traces with filters"""
        pass
```

### Monitoring Dashboard API

```python
# Dashboard API endpoints
class MonitoringAPI:
    @router.websocket("/ws/monitoring")
    async def monitoring_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time monitoring"""
        pass

    @router.get("/api/v1/monitoring/health")
    async def get_system_health() -> SystemHealthResponse:
        """Get overall system health status"""
        pass

    @router.get("/api/v1/monitoring/metrics")
    async def get_metrics(
        time_range: str = Query("1h"),
        agent_type: Optional[AgentType] = None
    ) -> MetricsResponse:
        """Get performance metrics"""
        pass

    @router.get("/api/v1/monitoring/costs")
    async def get_cost_analysis(
        time_range: str = Query("24h"),
        breakdown: bool = Query(False)
    ) -> CostAnalysisResponse:
        """Get cost analysis data"""
        pass

    @router.get("/api/v1/monitoring/alerts")
    async def get_alerts(
        status: Optional[AlertStatus] = None,
        limit: int = Query(50)
    ) -> AlertsResponse:
        """Get alert history"""
        pass
```

## Configuration Management

### Environment Variables

```bash
# LangSmith Configuration
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT_NAME=reautomation2-monitoring
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_TIMEOUT=30
LANGSMITH_BATCH_SIZE=100
LANGSMITH_FLUSH_INTERVAL=5

# Monitoring Configuration
MONITORING_ENABLED=true
MONITORING_DASHBOARD_PORT=8001
MONITORING_WEBSOCKET_ENABLED=true
MONITORING_RETENTION_DAYS=30

# Alert Configuration
ALERTS_ENABLED=true
ALERT_EMAIL_RECIPIENTS=dev-team@company.com,ops-team@company.com
ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/...
ALERT_ESCALATION_TIMEOUT=300

# Performance Thresholds
PERF_AGENT_WARNING_MS=500
PERF_AGENT_CRITICAL_MS=1000
PERF_WORKFLOW_ALERT_MS=2000

# Cost Thresholds
COST_DAILY_BUDGET_ALERT_PERCENT=80
COST_CALL_LIMIT_DOLLARS=0.10
COST_SPIKE_MULTIPLIER=2.0
```

### Configuration Classes

```python
# Configuration management
class MonitoringSettings(BaseSettings):
    # LangSmith settings
    langsmith_api_key: str
    langsmith_project_name: str = "reautomation2-monitoring"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_timeout: int = 30
    langsmith_batch_size: int = 100
    langsmith_flush_interval: int = 5

    # Monitoring settings
    monitoring_enabled: bool = True
    monitoring_dashboard_port: int = 8001
    monitoring_websocket_enabled: bool = True
    monitoring_retention_days: int = 30

    # Alert settings
    alerts_enabled: bool = True
    alert_email_recipients: List[str] = []
    alert_slack_webhook: Optional[str] = None
    alert_escalation_timeout: int = 300

    # Performance thresholds
    perf_agent_warning_ms: int = 500
    perf_agent_critical_ms: int = 1000
    perf_workflow_alert_ms: int = 2000

    # Cost thresholds
    cost_daily_budget_alert_percent: float = 0.8
    cost_call_limit_dollars: float = 0.10
    cost_spike_multiplier: float = 2.0

    class Config:
        env_file = ".env"
```

## Security Requirements

### Authentication and Authorization

```python
# Security configuration for monitoring endpoints
class MonitoringAuth:
    def __init__(self):
        self.api_key_header = "X-Monitoring-API-Key"
        self.jwt_secret = os.getenv("MONITORING_JWT_SECRET")

    async def verify_api_key(self, api_key: str) -> bool:
        """Verify API key for monitoring access"""
        pass

    async def verify_jwt_token(self, token: str) -> Optional[User]:
        """Verify JWT token for dashboard access"""
        pass

    def require_monitoring_access(self):
        """Dependency for monitoring endpoint access"""
        pass
```

### Data Privacy and Encryption

```python
# Data privacy and encryption utilities
class DataPrivacy:
    @staticmethod
    def sanitize_trace_data(trace: WorkflowTrace) -> WorkflowTrace:
        """Remove PII from trace data"""
        # Remove sensitive information from conversation data
        pass

    @staticmethod
    def encrypt_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in trace data"""
        pass

    @staticmethod
    def anonymize_user_data(user_input: str) -> str:
        """Anonymize user input for monitoring"""
        pass
```

## Performance Optimization

### Caching Strategy

```python
# Redis caching for monitoring data
class MonitoringCache:
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.cache_ttl = {
            'metrics': 60,      # 1 minute
            'dashboard': 30,    # 30 seconds
            'alerts': 300,      # 5 minutes
            'costs': 120        # 2 minutes
        }

    async def cache_metrics(self, key: str, data: Dict[str, Any]):
        """Cache performance metrics"""
        pass

    async def get_cached_metrics(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached metrics"""
        pass
```

### Batch Processing

```python
# Batch processing for trace data
class TraceBatchProcessor:
    def __init__(self, batch_size: int = 100, flush_interval: int = 5):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.trace_buffer = []

    async def add_trace(self, trace: WorkflowTrace):
        """Add trace to batch buffer"""
        self.trace_buffer.append(trace)
        if len(self.trace_buffer) >= self.batch_size:
            await self._flush_batch()

    async def _flush_batch(self):
        """Flush batch to LangSmith"""
        pass
```

## Testing Requirements

### Unit Testing

```python
# Test cases for monitoring components
class TestLangSmithIntegration:
    async def test_trace_creation(self):
        """Test trace creation in LangSmith"""
        pass

    async def test_cost_calculation(self):
        """Test cost calculation accuracy"""
        pass

    async def test_alert_processing(self):
        """Test alert rule processing"""
        pass

class TestPerformanceMonitoring:
    async def test_metrics_collection(self):
        """Test metrics collection accuracy"""
        pass

    async def test_real_time_updates(self):
        """Test WebSocket real-time updates"""
        pass
```

### Integration Testing

```python
# Integration test scenarios
class TestMonitoringIntegration:
    async def test_end_to_end_workflow_tracing(self):
        """Test complete workflow tracing from start to finish"""
        pass

    async def test_cost_integration_with_budget_controls(self):
        """Test cost tracking integration with existing budget system"""
        pass

    async def test_alert_notification_delivery(self):
        """Test alert notification delivery through various channels"""
        pass
```

## Deployment Specifications

### Docker Configuration

```dockerfile
# Dockerfile additions for monitoring
FROM python:3.11-slim

# Install monitoring dependencies
COPY requirements-monitoring.txt .
RUN pip install -r requirements-monitoring.txt

# Copy monitoring configuration
COPY monitoring/ /app/monitoring/

# Expose monitoring dashboard port
EXPOSE 8001

# Health check for monitoring services
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1
```

### Kubernetes Deployment

```yaml
# Kubernetes deployment for monitoring
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reautomation2-monitoring
spec:
  replicas: 2
  selector:
    matchLabels:
      app: reautomation2-monitoring
  template:
    metadata:
      labels:
        app: reautomation2-monitoring
    spec:
      containers:
        - name: monitoring
          image: reautomation2:monitoring-latest
          ports:
            - containerPort: 8001
          env:
            - name: LANGSMITH_API_KEY
              valueFrom:
                secretKeyRef:
                  name: langsmith-secret
                  key: api-key
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
```

## Monitoring and Observability

### Health Checks

```python
# Health check endpoints for monitoring system
@router.get("/health/monitoring")
async def monitoring_health_check():
    """Health check for monitoring system components"""
    health_status = {
        "langsmith_connection": await check_langsmith_connection(),
        "database_connection": await check_database_connection(),
        "redis_connection": await check_redis_connection(),
        "alert_system": await check_alert_system(),
        "dashboard_service": await check_dashboard_service()
    }

    overall_status = "healthy" if all(health_status.values()) else "unhealthy"

    return {
        "status": overall_status,
        "components": health_status,
        "timestamp": datetime.utcnow()
    }
```

### Metrics and Logging

```python
# Structured logging for monitoring system
import structlog

logger = structlog.get_logger(__name__)

class MonitoringLogger:
    @staticmethod
    def log_trace_created(trace_id: str, call_id: str):
        logger.info("trace_created", trace_id=trace_id, call_id=call_id)

    @staticmethod
    def log_alert_triggered(alert_type: str, severity: str, message: str):
        logger.warning("alert_triggered",
                      alert_type=alert_type,
                      severity=severity,
                      message=message)

    @staticmethod
    def log_performance_issue(component: str, metric: str, value: float, threshold: float):
        logger.error("performance_issue",
                    component=component,
                    metric=metric,
                    value=value,
                    threshold=threshold)
```

---

**Next Steps**: Proceed to dependencies and risks documentation, followed by task breakdown and implementation planning.
