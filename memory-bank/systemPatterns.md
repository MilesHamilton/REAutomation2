# System Patterns: REAutomation2

## System Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   LangGraph     │    │   Ollama        │
│   Web Server    │◄──►│   Orchestrator  │◄──►│   Local LLM     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Twilio        │    │   Pipecat       │    │   11Labs        │
│   WebRTC        │◄──►│   Voice Pipeline│◄──►│   Premium TTS   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Redis         │    │   Analytics     │
│   Database      │    │   Cache/Queue   │    │   Dashboard     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Relationships

#### Core Components

1. **AgentOrchestrator**: Central workflow management using LangGraph
2. **BaseAgent**: Abstract base class for all conversation agents
3. **Voice Pipeline**: Pipecat-based real-time audio processing
4. **LLM Service**: Ollama client for local language model inference
5. **Cost Control System**: Comprehensive budget management with real-time alerts
6. **LangSmith Monitoring**: Production-ready monitoring with circuit breaker protection
7. **Integration Services**: Google Sheets, CRM, and external API connectors

#### Agent Hierarchy

```
BaseAgent (Abstract)
├── ConversationAgent (Greeting & General Chat)
├── QualificationAgent (Lead Scoring & Questions)
├── ObjectionHandlerAgent (Objection Resolution)
├── SchedulerAgent (Appointment Booking)
└── AnalyticsAgent (Call Analysis & Metrics)
```

## Key Technical Decisions

### Multi-Agent Architecture Pattern

- **Decision**: Use LangGraph for agent orchestration instead of simple state machines
- **Rationale**: Enables complex conversation flows with dynamic routing
- **Implementation**: Each agent handles specific conversation phases with clear transitions
- **Benefits**: Modular design, easier testing, scalable conversation logic

### Dual-Tier Voice Strategy

- **Decision**: Local TTS (Pipecat) for pre-screening, 11Labs for qualified leads
- **Rationale**: Cost optimization while maintaining quality for important interactions
- **Implementation**: Tier escalation based on qualification scores and conversation context
- **Benefits**: <$0.10 per call target while preserving conversion quality

### Local LLM Integration

- **Decision**: Ollama with Llama 3.1 8B instead of cloud-only solutions
- **Rationale**: Cost control and reduced latency for high-volume operations
- **Implementation**: GPU-accelerated inference with concurrent request management
- **Benefits**: Predictable costs, data privacy, reduced API dependencies

### Real-Time Voice Processing

- **Decision**: Pipecat + Twilio WebRTC for voice pipeline
- **Rationale**: Low-latency requirements for natural conversation flow
- **Implementation**: Streaming audio processing with real-time transcription
- **Benefits**: Natural conversation experience, immediate response capability

## Design Patterns in Use

### Agent Pattern

```python
class BaseAgent(ABC):
    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        # Standard processing pipeline for all agents

    def can_handle(self, context: WorkflowContext) -> bool:
        # Agent selection logic

    def get_system_prompt(self, context: WorkflowContext) -> str:
        # Context-aware prompt generation
```

### State Management Pattern

```python
class WorkflowContext(BaseModel):
    call_id: str
    current_state: WorkflowState
    conversation_history: List[AgentMessage]
    qualification_factors: QualificationFactors
    cost_tracking: CostMetrics
```

### Factory Pattern for Agent Creation

```python
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: AgentType) -> BaseAgent:
        # Dynamic agent instantiation based on workflow needs
```

### Observer Pattern for Events

```python
class AgentOrchestrator:
    def on_agent_transition(self, callback: Callable):
        # Event-driven architecture for monitoring and analytics

    def on_tier_escalation(self, callback: Callable):
        # Cost and quality management events
```

## Critical Implementation Paths

### Call Initialization Flow

1. **API Request** → FastAPI endpoint receives call request
2. **Context Creation** → WorkflowContext initialized with lead data
3. **Agent Selection** → ConversationAgent selected for greeting phase
4. **Voice Setup** → Pipecat pipeline established with Twilio
5. **LLM Preparation** → Ollama model loaded and ready for inference

### Conversation Processing Loop

1. **Audio Input** → Twilio WebRTC streams audio to Pipecat
2. **Speech-to-Text** → Whisper processes audio to text
3. **Agent Processing** → Current agent processes input and generates response
4. **LLM Inference** → Ollama generates contextual response
5. **Text-to-Speech** → Pipecat converts response to audio
6. **Audio Output** → Response streamed back through Twilio

### Tier Escalation Decision

1. **Qualification Scoring** → QualificationAgent calculates lead score
2. **Threshold Check** → Score compared against escalation threshold (0.7)
3. **Cost Analysis** → Current call cost evaluated against budget
4. **Tier Switch** → If qualified, switch from local TTS to 11Labs
5. **Context Preservation** → Conversation context maintained across tiers

### Error Handling Strategy

1. **Graceful Degradation** → System continues with reduced functionality on component failure
2. **Retry Logic** → Automatic retry for transient failures (network, API limits)
3. **Fallback Mechanisms** → Local TTS fallback if 11Labs unavailable
4. **Circuit Breaker** → Prevent cascade failures in high-load scenarios

## Data Flow Architecture

### Request Flow

```
User Request → FastAPI → AgentOrchestrator → LangGraph → Agent → LLM → Response
```

### Voice Flow

```
Twilio → Pipecat → STT → Agent Processing → LLM → TTS → Pipecat → Twilio
```

### Analytics Flow

```
Agent Events → AnalyticsAgent → Metrics Calculation → Database → Dashboard
```

## Scalability Patterns

### Horizontal Scaling

- **Stateless Agents**: All agents designed to be stateless for easy scaling
- **Context Persistence**: WorkflowContext stored in Redis for multi-instance access
- **Load Balancing**: FastAPI instances can be load-balanced behind proxy

### Resource Management

- **Connection Pooling**: Database and Redis connections pooled and reused
- **LLM Queue Management**: Concurrent request limits prevent resource exhaustion
- **Memory Management**: Conversation history pruned to prevent memory leaks

### Performance Optimization

- **Caching Strategy**: Frequent LLM responses cached in Redis
- **Async Processing**: All I/O operations use async/await patterns
- **Batch Processing**: Analytics and metrics calculated in batches
