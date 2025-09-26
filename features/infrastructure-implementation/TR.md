# Technical Requirements: REAutomation2 Infrastructure Implementation

## Overview

This document defines the technical architecture, implementation patterns, and technology stack for the REAutomation2 infrastructure implementation. It provides detailed technical specifications for each component and their interactions.

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Voice Layer   │    │  Agent Layer    │    │  Data Layer     │
│                 │    │                 │    │                 │
│ • Pipecat       │◄──►│ • LangGraph     │◄──►│ • PostgreSQL    │
│ • Twilio        │    │ • Multi-Agents  │    │ • Redis Cache   │
│ • STT/TTS       │    │ • Orchestrator  │    │ • Google Sheets │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   API Layer     │
                    │                 │
                    │ • FastAPI       │
                    │ • WebSocket     │
                    │ • REST APIs     │
                    └─────────────────┘
```

### Component Interaction Flow

```
Call Initiation → Twilio → Pipecat → STT → Agent Orchestrator
                                              │
                                              ▼
                                         LLM Service
                                              │
                                              ▼
                                         TTS Manager → Audio Output
                                              │
                                              ▼
                                         Database Storage
```

## Technology Stack

### Core Technologies

| Component            | Technology | Version   | Purpose                          |
| -------------------- | ---------- | --------- | -------------------------------- |
| **Runtime**          | Python     | 3.11+     | Primary development language     |
| **Web Framework**    | FastAPI    | 0.104+    | API server and WebSocket support |
| **Voice Processing** | Pipecat    | Latest    | Real-time audio pipeline         |
| **Telephony**        | Twilio     | API v2010 | Voice calls and WebRTC           |
| **LLM Service**      | Ollama     | Latest    | Local LLM inference              |
| **Database**         | PostgreSQL | 15+       | Primary data storage             |
| **Cache**            | Redis      | 7+        | Session and response caching     |
| **Agent Framework**  | LangGraph  | Latest    | Multi-agent orchestration        |

### External Services

| Service           | Purpose            | Integration Method |
| ----------------- | ------------------ | ------------------ |
| **11Labs**        | Premium TTS        | REST API           |
| **Whisper**       | Speech-to-Text     | Local/API          |
| **Piper/Coqui**   | Local TTS          | Local inference    |
| **Google Sheets** | Contact management | Google API         |

## Phase 1: Foundation Components

### TASK-001: Voice Pipeline Implementation

#### Technical Architecture

```python
# Voice Pipeline Architecture
class VoicePipeline:
    """
    Real-time voice processing pipeline using Pipecat framework
    """
    def __init__(self):
        self.audio_processor = PipecatProcessor()
        self.stt_service = WhisperSTTService()
        self.tts_manager = TTSManager()
        self.twilio_client = TwilioWebRTCClient()

    async def process_audio_stream(self, audio_stream):
        # Process incoming audio with <200ms latency
        pass
```

#### Implementation Requirements

1. **Pipecat Integration**

   - Install and configure Pipecat framework
   - Set up real-time audio processing pipelines
   - Implement audio buffer management
   - Configure audio format handling (16kHz, 16-bit, mono)

2. **Twilio WebRTC Setup**

   ```python
   # Twilio WebRTC Configuration
   TWILIO_CONFIG = {
       "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
       "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
       "twiml_app_sid": os.getenv("TWILIO_TWIML_APP_SID"),
       "websocket_url": "wss://media-streams.twilio.com"
   }
   ```

3. **STT Integration**

   - Whisper model selection (base/small for speed vs. accuracy)
   - Real-time transcription with streaming
   - Confidence score thresholding
   - Language detection and handling

4. **TTS Management**
   ```python
   class TTSManager:
       def __init__(self):
           self.local_tts = PiperTTSService()
           self.premium_tts = ElevenLabsTTSService()
           self.cost_controller = CostController()

       async def synthesize_speech(self, text: str, lead_score: float):
           if self.cost_controller.should_use_premium(lead_score):
               return await self.premium_tts.synthesize(text)
           return await self.local_tts.synthesize(text)
   ```

#### Performance Requirements

- **Latency Target:** <200ms end-to-end
- **Concurrent Sessions:** 5+ simultaneous calls
- **Audio Quality:** 16kHz, 16-bit, mono PCM
- **Buffer Size:** 20ms audio chunks for real-time processing

### TASK-002: LLM Service Completion

#### Technical Architecture

```python
# LLM Service Architecture
class LLMService:
    """
    Optimized Ollama client with caching and queue management
    """
    def __init__(self):
        self.ollama_client = OptimizedOllamaClient()
        self.cache_service = RedisCache()
        self.queue_manager = RequestQueueManager()
        self.context_manager = ContextWindowManager()
```

#### Implementation Requirements

1. **Ollama Client Optimization**

   ```python
   # Connection Pool Configuration
   OLLAMA_CONFIG = {
       "base_url": "http://localhost:11434",
       "model": "llama3.1:8b",
       "max_connections": 10,
       "timeout": 30,
       "gpu_layers": -1,  # Use all GPU layers
       "context_length": 4096
   }
   ```

2. **Redis Caching System**

   ```python
   class ResponseCache:
       def __init__(self):
           self.redis_client = redis.Redis(
               host=os.getenv("REDIS_HOST", "localhost"),
               port=int(os.getenv("REDIS_PORT", 6379)),
               db=0,
               decode_responses=True
           )

       async def get_cached_response(self, prompt_hash: str):
           return await self.redis_client.get(f"llm_response:{prompt_hash}")
   ```

3. **Context Window Management**

   - Token counting and management
   - Context compression algorithms
   - Conversation history truncation
   - Agent context switching

4. **Request Queue Management**
   - Priority-based request queuing
   - Load balancing across model instances
   - Request timeout handling
   - GPU memory optimization

#### Performance Requirements

- **Response Time:** <2 seconds for LLM inference
- **Concurrent Requests:** 10+ simultaneous
- **Cache Hit Rate:** >70% for common patterns
- **GPU Utilization:** >80% efficiency

### TASK-003: Database Layer Setup

#### Technical Architecture

```python
# Database Architecture
class DatabaseLayer:
    """
    PostgreSQL with SQLAlchemy ORM and Alembic migrations
    """
    def __init__(self):
        self.engine = create_async_engine(DATABASE_URL)
        self.session_factory = async_sessionmaker(self.engine)
```

#### Schema Design

```sql
-- Core Tables
CREATE TABLE calls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID REFERENCES contacts(id),
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    duration INTEGER,
    cost DECIMAL(10,4),
    recording_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id UUID REFERENCES calls(id),
    agent_type VARCHAR(50) NOT NULL,
    message_type VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

CREATE TABLE lead_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    contact_id UUID REFERENCES contacts(id),
    call_id UUID REFERENCES calls(id),
    qualification_score DECIMAL(3,2),
    interest_level VARCHAR(20),
    budget_qualified BOOLEAN,
    timeline VARCHAR(50),
    decision_maker BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

#### Implementation Requirements

1. **Alembic Migration Setup**

   ```python
   # alembic/env.py configuration
   from src.database.models import Base
   target_metadata = Base.metadata

   def run_migrations_online():
       connectable = create_engine(DATABASE_URL)
       with connectable.connect() as connection:
           context.configure(
               connection=connection,
               target_metadata=target_metadata
           )
   ```

2. **SQLAlchemy Models**

   ```python
   class Call(Base):
       __tablename__ = "calls"

       id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
       contact_id = Column(UUID(as_uuid=True), ForeignKey("contacts.id"))
       status = Column(String(50), nullable=False)
       start_time = Column(DateTime(timezone=True))
       end_time = Column(DateTime(timezone=True))
       cost = Column(Numeric(10, 4))

       # Relationships
       contact = relationship("Contact", back_populates="calls")
       conversations = relationship("Conversation", back_populates="call")
   ```

3. **Repository Pattern**
   ```python
   class CallRepository:
       def __init__(self, session: AsyncSession):
           self.session = session

       async def create_call(self, call_data: CallCreate) -> Call:
           call = Call(**call_data.dict())
           self.session.add(call)
           await self.session.commit()
           return call
   ```

#### Performance Requirements

- **Connection Pool:** 20 connections max
- **Query Performance:** <100ms for standard queries
- **Transaction Support:** ACID compliance
- **Indexing Strategy:** Optimized for time-series queries

## Phase 2: Core Features

### TASK-004: Cost Control System

#### Technical Architecture

```python
class CostController:
    """
    Real-time cost tracking and budget enforcement
    """
    def __init__(self):
        self.cost_calculator = CostCalculator()
        self.budget_enforcer = BudgetEnforcer()
        self.tier_decision_engine = TierDecisionEngine()
```

#### Implementation Requirements

1. **Cost Calculation Engine**

   ```python
   class CostCalculator:
       PRICING = {
           "whisper_stt": 0.006,  # per minute
           "piper_tts": 0.0,      # free local
           "elevenlabs_tts": 0.30, # per 1K characters
           "ollama_llm": 0.0      # free local
       }

       def calculate_call_cost(self, call_metrics: CallMetrics) -> Decimal:
           stt_cost = call_metrics.duration_minutes * self.PRICING["whisper_stt"]
           tts_cost = self.calculate_tts_cost(call_metrics)
           return stt_cost + tts_cost
   ```

2. **Budget Enforcement**
   ```python
   class BudgetEnforcer:
       def __init__(self):
           self.daily_limit = Decimal(os.getenv("DAILY_BUDGET_LIMIT", "50.00"))
           self.monthly_limit = Decimal(os.getenv("MONTHLY_BUDGET_LIMIT", "1000.00"))

       async def check_budget_limits(self) -> BudgetStatus:
           current_spend = await self.get_current_spend()
           return BudgetStatus(
               can_proceed=current_spend < self.daily_limit,
               remaining_budget=self.daily_limit - current_spend
           )
   ```

### TASK-005: Integration Services

#### Technical Architecture

```python
class IntegrationService:
    """
    External service integrations and session management
    """
    def __init__(self):
        self.elevenlabs_client = ElevenLabsClient()
        self.redis_session_manager = RedisSessionManager()
        self.webhook_manager = WebhookManager()
```

#### Implementation Requirements

1. **11Labs Integration**

   ```python
   class ElevenLabsClient:
       def __init__(self):
           self.api_key = os.getenv("ELEVENLABS_API_KEY")
           self.base_url = "https://api.elevenlabs.io/v1"
           self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")

       async def synthesize_speech(self, text: str) -> bytes:
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   f"{self.base_url}/text-to-speech/{self.voice_id}",
                   headers={"xi-api-key": self.api_key},
                   json={"text": text, "model_id": "eleven_monolingual_v1"}
               )
               return response.content
   ```

2. **Redis Session Management**
   ```python
   class RedisSessionManager:
       def __init__(self):
           self.redis = redis.Redis(
               host=os.getenv("REDIS_HOST"),
               port=int(os.getenv("REDIS_PORT", 6379)),
               decode_responses=True
           )

       async def store_conversation_state(self, call_id: str, state: dict):
           await self.redis.setex(
               f"conversation:{call_id}",
               3600,  # 1 hour expiration
               json.dumps(state)
           )
   ```

### TASK-006: Error Handling & Recovery

#### Technical Architecture

```python
class ErrorHandler:
    """
    Comprehensive error handling and recovery system
    """
    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.retry_manager = RetryManager()
        self.fallback_manager = FallbackManager()
```

#### Implementation Requirements

1. **Circuit Breaker Pattern**

   ```python
   class CircuitBreaker:
       def __init__(self, failure_threshold: int = 5, timeout: int = 60):
           self.failure_threshold = failure_threshold
           self.timeout = timeout
           self.failure_count = 0
           self.last_failure_time = None
           self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

       async def call(self, func, *args, **kwargs):
           if self.state == "OPEN":
               if time.time() - self.last_failure_time > self.timeout:
                   self.state = "HALF_OPEN"
               else:
                   raise CircuitBreakerOpenError()

           try:
               result = await func(*args, **kwargs)
               self.reset()
               return result
           except Exception as e:
               self.record_failure()
               raise e
   ```

2. **Retry Logic**
   ```python
   class RetryManager:
       @retry(
           stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=4, max=10),
           retry=retry_if_exception_type(httpx.RequestError)
       )
       async def make_request(self, func, *args, **kwargs):
           return await func(*args, **kwargs)
   ```

## Phase 3: Production Ready

### TASK-007: Testing Infrastructure

#### Testing Architecture

```python
# Test Structure
tests/
├── unit/
│   ├── agents/
│   ├── voice/
│   ├── llm/
│   └── database/
├── integration/
│   ├── test_voice_pipeline.py
│   ├── test_agent_orchestration.py
│   └── test_database_operations.py
├── performance/
│   ├── test_concurrent_calls.py
│   └── test_latency_benchmarks.py
└── conftest.py
```

#### Implementation Requirements

1. **Unit Testing Framework**

   ```python
   # pytest configuration
   @pytest.fixture
   async def mock_llm_service():
       with patch('src.llm.service.LLMService') as mock:
           mock.generate_response.return_value = "Test response"
           yield mock

   @pytest.mark.asyncio
   async def test_agent_conversation_flow(mock_llm_service):
       agent = ConversationAgent()
       response = await agent.process_message("Hello")
       assert response.content == "Test response"
   ```

2. **Load Testing**
   ```python
   # locust load testing
   class VoiceCallUser(HttpUser):
       wait_time = between(1, 3)

       @task
       def simulate_voice_call(self):
           # Simulate concurrent voice calls
           pass
   ```

### TASK-008: Monitoring & Analytics

#### Monitoring Architecture

```python
class MonitoringService:
    """
    Prometheus metrics and real-time monitoring
    """
    def __init__(self):
        self.metrics_collector = PrometheusMetrics()
        self.dashboard = GrafanaDashboard()
        self.alerting = AlertManager()
```

#### Implementation Requirements

1. **Prometheus Metrics**

   ```python
   from prometheus_client import Counter, Histogram, Gauge

   # Define metrics
   CALL_COUNTER = Counter('calls_total', 'Total number of calls', ['status'])
   CALL_DURATION = Histogram('call_duration_seconds', 'Call duration')
   ACTIVE_CALLS = Gauge('active_calls', 'Number of active calls')
   COST_GAUGE = Gauge('cost_per_call_dollars', 'Cost per call in dollars')
   ```

2. **Real-time Dashboard**
   - Grafana dashboard configuration
   - Real-time call monitoring
   - Cost tracking visualization
   - Performance metrics display

### TASK-009: Documentation & Deployment

#### Deployment Architecture

```yaml
# docker-compose.yml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/reautomation
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: reautomation
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

#### Implementation Requirements

1. **Docker Configuration**

   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY src/ ./src/
   COPY alembic/ ./alembic/
   COPY alembic.ini .

   CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Production Setup Scripts**

   ```bash
   #!/bin/bash
   # deploy.sh

   # Build and deploy containers
   docker-compose build
   docker-compose up -d

   # Run database migrations
   docker-compose exec api alembic upgrade head

   # Health check
   curl -f http://localhost:8000/health || exit 1
   ```

## Environment Configuration

### Development Environment

```bash
# .env.development
DATABASE_URL=postgresql://dev:dev@localhost:5432/reautomation_dev
REDIS_URL=redis://localhost:6379
OLLAMA_BASE_URL=http://localhost:11434
TWILIO_ACCOUNT_SID=your_dev_sid
TWILIO_AUTH_TOKEN=your_dev_token
ELEVENLABS_API_KEY=your_dev_key
LOG_LEVEL=DEBUG
```

### Production Environment

```bash
# .env.production
DATABASE_URL=postgresql://prod_user:secure_pass@prod_db:5432/reautomation
REDIS_URL=redis://prod_redis:6379
OLLAMA_BASE_URL=http://ollama_server:11434
TWILIO_ACCOUNT_SID=your_prod_sid
TWILIO_AUTH_TOKEN=your_prod_token
ELEVENLABS_API_KEY=your_prod_key
LOG_LEVEL=INFO
DAILY_BUDGET_LIMIT=100.00
MONTHLY_BUDGET_LIMIT=2000.00
```

## Security Considerations

### Authentication & Authorization

```python
# JWT Authentication
class JWTAuth:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"

    def create_access_token(self, data: dict):
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=30)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
```

### Data Protection

- Encryption at rest for sensitive data
- TLS 1.3 for all external communications
- API key rotation and management
- Call recording consent and compliance

## Performance Optimization

### Caching Strategy

```python
# Multi-level caching
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = redis.Redis()  # Redis cache

    async def get(self, key: str):
        # Check L1 cache first
        if key in self.l1_cache:
            return self.l1_cache[key]

        # Check L2 cache
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value

        return None
```

### Database Optimization

```sql
-- Performance indexes
CREATE INDEX CONCURRENTLY idx_calls_start_time ON calls(start_time);
CREATE INDEX CONCURRENTLY idx_conversations_call_id ON conversations(call_id);
CREATE INDEX CONCURRENTLY idx_lead_scores_contact_id ON lead_scores(contact_id);

-- Partitioning for large tables
CREATE TABLE calls_2024 PARTITION OF calls
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

## Scalability Considerations

### Horizontal Scaling

- Stateless API design for load balancing
- Database read replicas for query scaling
- Redis clustering for cache scaling
- Container orchestration with Kubernetes

### Resource Management

```python
# Resource limits and monitoring
RESOURCE_LIMITS = {
    "max_concurrent_calls": 10,
    "max_memory_usage": "2GB",
    "max_cpu_usage": "80%",
    "max_gpu_memory": "4GB"
}
```

This technical requirements document provides the detailed implementation specifications needed to build the REAutomation2 infrastructure. Each component includes specific code examples, configuration details, and performance requirements to guide the development process.
