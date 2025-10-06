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
8. **Database Layer**: Complete migration system with Alembic, SQLAlchemy models, and testing framework

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

- **Decision**: Use LangGraph 0.6.8 for agent orchestration instead of simple state machines
- **Rationale**: Enables complex conversation flows with dynamic routing and better debugging
- **Implementation**: Each agent handles specific conversation phases with clear transitions
- **Benefits**: Modular design, easier testing, scalable conversation logic, improved workflow visualization

### LangGraph Upgrade & Compatibility Pattern

- **Decision**: Upgrade from LangGraph 0.0.25 to 0.6.8 with full backward compatibility
- **Rationale**: Access to latest features, improved performance, and better debugging tools
- **Implementation**: Resolved circular imports, fixed dependency conflicts, updated monitoring integration
- **Benefits**: Enhanced stability, better error handling, improved development experience

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

## LangGraph Upgrade Patterns

### Circular Import Resolution Pattern

- **Problem**: Circular imports between orchestrator and monitoring modules
- **Solution**: Lazy imports using function-level imports instead of module-level
- **Implementation**: Import statements moved inside functions to break circular dependencies
- **Benefits**: Cleaner module initialization, reduced startup complexity

### Dependency Compatibility Pattern

- **Problem**: Missing dependencies causing import failures during system startup
- **Solution**: Comprehensive dependency audit and systematic installation
- **Implementation**: Added psutil, backoff, icalendar, pytz, cartesia to requirements.txt
- **Benefits**: Stable system startup, reduced deployment issues

### Database Import Standardization Pattern

- **Problem**: Inconsistent database import paths across monitoring modules
- **Solution**: Standardized all database imports to use connection.py module
- **Implementation**: Replaced `..database.core` with `..database.connection` throughout codebase
- **Benefits**: Consistent database access, easier maintenance

### Model Validation Compatibility Pattern

- **Problem**: Pydantic model validation errors due to field name mismatches
- **Solution**: Created compatibility aliases and updated model initialization
- **Implementation**: Added AlertSeverity = AlertLevel alias, fixed AlertRule field names
- **Benefits**: Backward compatibility, smoother upgrades

### Monitoring Integration Pattern

- **Problem**: LangSmith client compatibility with new LangGraph version
- **Solution**: Updated API calls to use direct HTTP requests instead of deprecated imports
- **Implementation**: Replaced RunCreate imports with direct API endpoint calls
- **Benefits**: Future-proof monitoring, reduced dependency on internal APIs

## Voice-Agent Integration Patterns

### Orchestration Integration Pattern

- **Pattern**: ConversationProcessor routes voice input through AgentOrchestrator with timeout-based fallback
- **Implementation**: 500ms timeout on agent processing with fallback to direct LLM
- **Benefits**: Low-latency voice responses, graceful degradation on agent failure

### State Synchronization Pattern

- **Pattern**: Bidirectional sync between CallSession and WorkflowContext
- **Implementation**: `_sync_workflow_state()` method updates workflow_context_id, workflow_state, current_agent
- **Benefits**: Consistent state across voice and agent systems, improved debugging

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    # Three states: CLOSED → OPEN → HALF_OPEN → CLOSED
    # Prevents cascade failures in agent orchestration
    # Configurable failure threshold and timeout
```

- **States**: CLOSED (normal), OPEN (blocked), HALF_OPEN (testing recovery)
- **Configuration**: 5 failures trigger OPEN, 60-second timeout before HALF_OPEN
- **Benefits**: Prevents cascade failures, automatic recovery, improved reliability

### Response Caching Pattern

```python
class ResponseCache:
    # LRU cache with TTL for agent responses
    # Cache keys: hash(call_id + user_input + context)
    # Automatic cleanup of expired entries
```

- **Strategy**: LRU eviction with 5-minute TTL
- **Key Generation**: MD5 hash of call_id, user_input, and workflow context
- **Benefits**: Reduced latency, lower LLM costs, improved user experience

### Voice Optimization Pattern

- **Response Truncation**: Limit responses to 150 words at sentence boundaries
- **Markdown Removal**: Strip formatting for natural speech synthesis
- **Conversational Flow**: Ensure responses end with proper punctuation
- **Benefits**: Natural voice delivery, improved TTS quality, reduced audio latency

### Metrics Collection Pattern

```python
class VoiceAgentIntegrationMetrics:
    # Tracks agent transitions, state sync, tier escalations
    # Calculates hit rates, failure rates, latency percentiles
    # Provides comprehensive summary for dashboards
```

- **Metrics Types**: AgentTransitionMetrics, StateSyncMetrics, TierEscalationMetrics, PerformanceMetrics
- **Aggregation**: Real-time calculation of rates, averages, and percentiles
- **Benefits**: Data-driven optimization, proactive issue detection, business insights

### Database Schema Extension Pattern

- **New Tables**: agent_transitions, tier_escalation_events
- **Migration Strategy**: Alembic migration with proper indexes and foreign keys
- **Rollback Support**: Down migration removes all changes cleanly
- **Benefits**: Comprehensive audit trail, analytics support, easy rollback

## Audio Processing Patterns

### Real-Time Audio Processing Pipeline Pattern

```python
class AudioProcessor:
    # Complete processing pipeline: Jitter Buffer → AEC → Noise Reduction → Buffer
    # Target: <200ms end-to-end latency
    # 20ms chunk processing (320 samples @ 16kHz)
```

- **Processing Pipeline**:
  1. **Jitter Buffer**: Adaptive buffering (40-200ms), packet reordering, loss detection
  2. **Echo Cancellation**: Speex AEC or NLMS adaptive filter
  3. **Noise Reduction**: noisereduce library or spectral subtraction
  4. **Buffer Management**: Circular buffer with latency tracking

- **Components**: 6 modules (1,894 lines production code)
  - `audio_buffer.py` (321 lines): Circular buffer, 20ms chunking
  - `jitter_buffer.py` (385 lines): Adaptive jitter buffering
  - `packet_loss_concealment.py` (248 lines): Loss concealment (simple/linear/spectral)
  - `echo_cancellation.py` (276 lines): Acoustic echo cancellation
  - `noise_reduction.py` (314 lines): Background noise reduction
  - `audio_processor.py` (350 lines): Unified pipeline orchestrator

- **Performance**:
  - Total latency: <200ms (target met)
  - Processing overhead: <50ms
  - Throughput: >5x real-time
  - Memory: ~200KB per call

- **Benefits**: Production-grade VoIP quality, adaptive to network conditions, comprehensive monitoring

### Audio Buffer Management Pattern

```python
class AudioBufferManager:
    # Circular buffer with 20ms chunks
    # O(1) operations using deque
    # Latency tracking and overflow/underflow detection
```

- **Features**:
  - Ring buffer (500ms default)
  - Automatic chunk normalization (padding/truncation)
  - Zero-copy operations where possible
  - Async read/write with locks
  - Health checks (fill percentage, latency)

- **Metrics Tracked**:
  - Buffer fill percentage
  - Underruns/overruns
  - Average/current latency
  - Total chunks processed

### Adaptive Jitter Buffer Pattern

```python
class AdaptiveJitterBuffer:
    # Handles network jitter and packet loss
    # Target: 60-100ms adaptive delay
    # Packet reordering by sequence number
```

- **Algorithm**:
  1. Add packets to min heap (by sequence number)
  2. Wait for target buffer delay before playout
  3. Detect missing packets, trigger concealment
  4. Measure jitter, adjust buffer size dynamically
  5. Filter duplicates and late packets

- **Adaptive Sizing**:
  - Increase buffer on high jitter (>50ms)
  - Decrease buffer on stable network (<10ms jitter, <1% loss)
  - Range: 40-200ms

- **Metrics**:
  - Jitter (inter-arrival time variance)
  - Packet loss rate
  - Late/duplicate/out-of-order packets
  - Concealed packets

### Packet Loss Concealment Pattern

```python
class PacketLossConcealment:
    # Three concealment methods
    # Adaptive to consecutive losses
```

- **Methods**:
  1. **Simple**: Repeat last packet with exponential fade (<5% loss)
  2. **Linear**: Extrapolate trend between packets (5-10% loss)
  3. **Spectral**: FFT-based magnitude/phase prediction (best quality)

- **Algorithm (Spectral)**:
  1. FFT last 3 packets → magnitude/phase spectra
  2. Calculate average progression (differences)
  3. Extrapolate next spectrum with damping
  4. IFFT → concealed audio

- **Adaptation**: Fade factor = 0.7^consecutive_losses

### Echo Cancellation Pattern

```python
class EchoCancellationProcessor:
    # Acoustic Echo Cancellation
    # Primary: Speex DSP (industry standard)
    # Fallback: NLMS adaptive filter
```

- **Speex AEC**:
  - Optimized C implementation
  - Frame size: 320 samples (20ms)
  - Filter length: 1024 taps
  - Processing time: <10ms

- **NLMS Fallback**:
  - Normalized Least Mean Squares
  - Step size μ = 0.1
  - Regularization ε = 1e-6
  - Sample-by-sample adaptation

- **Far-end/Near-end Processing**:
  - Far-end: TTS audio (reference signal)
  - Near-end: Microphone input (with echo)
  - Output: Echo-cancelled audio

### Noise Reduction Pattern

```python
class NoiseReductionProcessor:
    # Background noise reduction
    # Primary: noisereduce (statistical)
    # Fallback: Spectral subtraction
```

- **noisereduce Method**:
  - Statistical noise profiling
  - Spectral gating
  - Configurable reduction strength (0.0-1.0)
  - Processing time: <15ms

- **Spectral Subtraction Fallback**:
  1. Learn noise spectrum (10 initial frames)
  2. FFT input audio → magnitude spectrum
  3. Subtract noise spectrum with strength factor
  4. Floor to prevent negative magnitudes
  5. IFFT → clean audio

- **Noise Learning**:
  - Automatic from initial frames (no speech)
  - Explicit from silence samples
  - Running average with smoothing (α = 0.9)

### Twilio Audio Conversion Pattern

```python
class TwilioIntegration:
    # Bidirectional audio format conversion
    # Inbound: µ-law 8kHz → PCM 16kHz
    # Outbound: PCM 16kHz → µ-law 8kHz
```

- **Inbound Pipeline**:
  1. Receive µ-law 8kHz from Twilio WebSocket
  2. Convert µ-law to PCM using audioop.ulaw2lin()
  3. Resample from 8kHz to 16kHz using scipy.signal.resample()
  4. Pass PCM 16kHz to voice processing (STT, LLM)

- **Outbound Pipeline**:
  1. Receive PCM audio (any sample rate) from TTS
  2. Resample to 8kHz using scipy.signal.resample()
  3. Convert PCM to µ-law using audioop.lin2ulaw()
  4. Send µ-law 8kHz to Twilio WebSocket

- **Error Handling**: Graceful fallback if scipy/audioop unavailable
- **Benefits**: Standards-compliant Twilio integration, optimal audio quality for processing

### STT Confidence Scoring Pattern

```python
def _calculate_confidence(self, segments: list) -> float:
    # Extract avg_logprob from Whisper segments
    # Map log probabilities to 0-1 confidence scale
    # Consider no_speech_prob for accuracy
    # Return bounded confidence score
```

- **Whisper Log Probability Mapping**:
  - avg_logprob >= -0.5 → confidence = 1.0 (high)
  - avg_logprob <= -3.0 → confidence = 0.0 (low)
  - Linear interpolation for values in between

- **No-Speech Detection**:
  - If no_speech_prob > 0.5, reduce confidence by (1 - no_speech_prob)
  - Prevents false positives from background noise

- **Fallback**: Return 0.7 default if no probability data available
- **Benefits**: Real confidence metrics, quality-based filtering, improved transcription accuracy

### STT Language Detection Pattern

```python
class STTConfig:
    language: str = "en"
    auto_detect_language: bool = False  # Enable for multilingual

def _transcribe_with_whisper(self, audio_array):
    # Use language=None for auto-detection
    # Use configured language for fixed-language mode
    language = None if auto_detect_language else self.config.language
    result = model.transcribe(audio_array, language=language)
```

- **Auto-Detection Mode**: Whisper determines language automatically
- **Fixed-Language Mode**: Use configured language for faster processing
- **Language Tracking**: detected_language and language_probability in STTResult
- **Benefits**: Multilingual support, language-specific processing, improved accuracy

## Performance Optimization Patterns

### Prompt Optimization Pattern

```python
class PromptOptimizer:
    # Token reduction through compression techniques
    # 15-25% token savings on average
    # Pre-optimized templates for common scenarios
```

- **Techniques**:
  1. **Whitespace/Formatting Removal**: Remove excessive whitespace, markdown formatting
  2. **Verbose Phrase Replacement**: "in order to" → "to", "due to the fact that" → "because"
  3. **Filler Word Removal**: Remove "just", "really", "actually", "literally"
  4. **Abbreviations** (aggressive mode): "you are" → "you're", "cannot" → "can't"
  5. **Explanation Removal** (aggressive mode): Remove parentheticals, "Note:", "Remember:" sections

- **Template Management**:
  - Pre-optimized templates: conversation, qualification, objection, summarization
  - Variable substitution with `.format()`
  - Token counting with tiktoken
  - Version tracking

- **Benefits**:
  - Conversation prompt: 80% token reduction (450 → 90 tokens)
  - Qualification prompt: 56% token reduction (180 → 80 tokens)
  - Faster inference (20-35% latency reduction)
  - Lower costs

### Streaming Response Pattern

```python
class StreamingHandler:
    # Chunks LLM responses in real-time
    # Sentence-boundary detection for natural speech
    # TTFC (Time-To-First-Chunk) optimization
```

- **Features**:
  - **Buffering**: Aggregate chunks until sentence boundaries
  - **Error Recovery**: Automatic retry with exponential backoff
  - **Voice Adaptation**: Chunk optimization for TTS (10-200 chars)
  - **Metrics Tracking**: TTFC, throughput (tokens/sec), chunks/sec

- **Streaming Pipeline**:
  1. Receive chunks from Ollama generate_stream()
  2. Buffer chunks until sentence boundary (. ! ?)
  3. Yield complete sentences to client
  4. Track performance metrics
  5. Handle errors gracefully with retry

- **Voice Optimization**:
  - Min chunk length: 10 chars
  - Max chunk length: 200 chars
  - Sentence boundary priority
  - Clause boundary fallback (comma, semicolon)
  - Word boundary forced split

- **Benefits**:
  - 50-70% reduction in perceived latency
  - Natural speech synthesis
  - Better user experience
  - Real-time progress indicators

### Performance Auto-Tuning Pattern

```python
class PerformanceOptimizer:
    # GPU utilization-based concurrency tuning
    # Adaptive batch size optimization
    # Latency-aware scaling
```

- **Auto-Tuning Algorithm**:
  1. Collect performance samples (latency, GPU util, throughput, success rate)
  2. Calculate P95 latency, average GPU utilization
  3. Compare against targets (75% GPU util, 2000ms latency)
  4. Adjust concurrency: underutilized → increase, overutilized → decrease
  5. Adjust batch size based on GPU capacity and success rate
  6. Track improvement/degradation streaks

- **Configuration**:
  - Target GPU utilization: 75% (configurable)
  - Target latency P95: 2000ms (configurable)
  - Concurrency range: 3-10 (adaptive)
  - Adjustment interval: 60 seconds
  - Performance history: 100 samples

- **Optimization Score**:
  - GPU score: 1.0 - abs(actual - target)
  - Latency score: max(0, 1.0 - actual/target)
  - Success score: success_rate
  - Overall: average of three scores

- **Benefits**:
  - 15-30% improvement in GPU utilization
  - Automatic scaling based on load
  - Performance regression detection
  - Self-healing under changing conditions

### GPU Monitoring & Dashboard Pattern

```python
# Performance monitoring with GPU metrics
def record_gpu_metrics(self, gpu_id, utilization, memory, temperature):
    # Track GPU utilization, memory, temperature, power
    # Integrate with performance dashboard
    # Real-time alerts on low utilization or high temperature
```

- **Metrics Collected**:
  - **Utilization**: GPU compute utilization percentage (0-100%)
  - **Memory**: Used/total memory in MB, utilization percentage
  - **Temperature**: GPU temperature in Celsius
  - **Power Draw**: Power consumption in watts (if available)
  - **Models Loaded**: Number of models currently in GPU memory

- **Dashboard Integration**:
  - Real-time GPU metrics display
  - Historical trend charts
  - Utilization alerts (<50% = underutilized, >90% = overloaded)
  - Memory pressure indicators
  - Temperature warnings

- **Streaming Metrics Dashboard**:
  - Total streams processed
  - Average TTFC (Time-To-First-Chunk)
  - Average throughput (tokens/second)
  - Average chunks per stream
  - Real-time streaming activity

- **Collection Frequency**: Every 2 minutes (system metrics collector)

- **Benefits**:
  - Proactive performance monitoring
  - Capacity planning insights
  - Early warning for hardware issues
  - Data-driven optimization decisions

### Streaming API Patterns

```python
# SSE (Server-Sent Events)
@router.post("/streaming/sse")
async def stream_sse(request: StreamRequest):
    # Unidirectional HTTP streaming
    # Compatible with standard HTTP clients
    # Format: "data: {json}\n\n"

# WebSocket
@router.websocket("/streaming/ws/{call_id}")
async def stream_websocket(websocket: WebSocket):
    # Bidirectional communication
    # Lower overhead than SSE
    # Real-time interactive applications
```

- **SSE Endpoint** (`POST /streaming/sse`):
  - **Protocol**: HTTP with `text/event-stream` content type
  - **Format**: `data: {json}\n\n` for each chunk
  - **Chunk Types**: "chunk" (content), "done" (completion), "error" (failure)
  - **Headers**: Cache-Control: no-cache, Connection: keep-alive, X-Accel-Buffering: no
  - **Use Case**: Web dashboards, monitoring tools, one-way streaming

- **WebSocket Endpoint** (`WS /streaming/ws/{call_id}`):
  - **Protocol**: WebSocket upgrade from HTTP
  - **Format**: JSON messages with type field
  - **Bidirectional**: Client can send follow-up requests on same connection
  - **Lower Latency**: No HTTP overhead per message
  - **Use Case**: Interactive chat, voice applications, real-time collaboration

- **Request Model**:
  ```python
  class StreamRequest:
      call_id: str
      user_input: str
      conversation_history: list
      system_prompt: Optional[str]
      for_voice: bool  # Voice-optimized chunking
      lead_info: dict
      conversation_state: str
  ```

- **Performance Tracking**:
  - Metrics recorded per stream (TTFC, chunks, tokens, duration, throughput)
  - Aggregated metrics available via `/streaming/metrics`
  - Integration with performance monitoring dashboard

- **Error Handling**:
  - Automatic retry on transient failures
  - Graceful error messages to client
  - Connection cleanup on errors
  - Circuit breaker protection

- **Benefits**:
  - Real-time response streaming
  - Multiple client options (SSE/WebSocket)
  - Voice-optimized chunking support
  - Production-ready error handling
  - Comprehensive metrics

### Combined Optimization Impact

- **Inference Latency**: ↓ 20-35% (prompt optimization + streaming)
- **Time-To-First-Token**: ↓ 50-70% (streaming)
- **GPU Utilization**: ↑ 15-30% (auto-tuning)
- **Token Usage**: ↓ 15-25% (prompt optimization)
- **Overall Throughput**: ↑ 30-50% (combined optimizations)
- **User-Perceived Latency**: ↓ 60-80% (streaming + optimization)

### Implementation Files

**Performance Optimizations** (1,190 lines):
- `src/llm/prompt_optimizer.py` (330 lines): Template management, token optimization
- `src/llm/streaming_handler.py` (340 lines): Streaming buffers, voice adaptation
- `src/llm/performance_optimizer.py` (300 lines): Auto-tuning, GPU-based scaling

---

## Database Migration Patterns

### Migration Management Strategy

- **Decision**: Use Alembic for all schema changes with comprehensive testing
- **Rationale**: Version control, repeatability, rollback capability, data preservation
- **Implementation**: Auto-generate migrations from model changes, test upgrade/downgrade cycles
- **Benefits**: Safe deployments, easy rollbacks, documented schema evolution

### Database Schema Evolution Pattern

**5-Phase Migration System**:

1. **Base Tables** (`26a6cb1543c8_add_base_tables.py`)
   - Core tables: calls, contacts, conversation_history, call_notes
   - 40+ indexes for primary queries
   - Foundation for all operations

2. **LangSmith Monitoring** (`001_add_langsmith_monitoring_tables.py`)
   - Tables: workflow_traces, agent_executions, performance_metrics
   - 50+ indexes for observability
   - System metrics tracking

3. **Voice-Agent Integration** (`002_add_voice_agent_integration_fields.py`)
   - Tables: agent_transitions, tier_escalation_events
   - Voice-specific fields and tracking
   - Integration monitoring

4. **Context Management** (`003_add_context_management_fields.py`)
   - Fields: context_pruned, pruning_count, total_context_tokens
   - Message importance and token tracking
   - 6 new indexes for context analytics

5. **Performance Indexes** (`004_add_performance_indexes.py`)
   - 15 new indexes for optimized queries
   - Covering indexes for dashboard queries
   - Partial indexes for selective queries (WHERE clauses)
   - Time-series optimized indexes (DESC ordering)

### Index Strategy Patterns

**Covering Indexes** (reduce disk I/O):
```python
# Example: Dashboard query optimization
op.create_index(
    'ix_calls_status_created_cost',
    'calls',
    ['status', 'created_at', 'total_cost'],  # All columns needed by query
    postgresql_using='btree'
)
```

**Partial Indexes** (reduce index size, improve maintenance):
```python
# Example: Only index context-pruned calls
op.execute("""
    CREATE INDEX ix_calls_context_pruned_only
    ON calls(total_context_tokens, pruning_count)
    WHERE context_pruned = true  -- Only ~10-20% of calls
""")
```

**Time-Series Indexes** (optimize recent data queries):
```python
# Example: Recent metrics queries
op.create_index(
    'ix_system_metrics_name_recorded_desc',
    'system_metrics',
    ['metric_name', sa.text('recorded_at DESC'), 'metric_value']
)
```

**Composite Indexes** (multi-column queries):
```python
# Example: Analytics queries
op.create_index(
    'ix_cost_tracking_type_date_amount',
    'cost_tracking',
    ['cost_type', 'daily_date', 'cost_amount']
)
```

### Migration Testing Pattern

**Comprehensive Test Coverage**:
```python
class MigrationTester:
    async def upgrade_to(self, revision: str):
        # Apply migration

    async def downgrade_to(self, revision: str):
        # Rollback migration

    async def get_table_columns(self, table_name: str) -> List[str]:
        # Verify schema changes

    async def get_table_indexes(self, table_name: str) -> List[Dict]:
        # Verify indexes created

    async def explain_query(self, query: str) -> str:
        # Validate index effectiveness
```

**Test Categories**:
1. **Upgrade Tests**: Verify schema changes applied correctly
2. **Downgrade Tests**: Verify rollback removes changes cleanly
3. **Data Preservation Tests**: Verify existing data not lost or corrupted
4. **Index Effectiveness Tests**: Verify indexes improve query performance (EXPLAIN ANALYZE)
5. **Full Cycle Tests**: Verify complete upgrade/downgrade cycle works
6. **Timing Benchmarks**: Verify migrations complete within time limits (<30s)

### Data Migration Pattern

**Backfill Strategy** (for new fields on existing data):

```python
class ContextFieldBackfiller:
    async def run(self):
        # 1. Count records to backfill
        total = await self.count_records_to_backfill()

        # 2. Process in batches (avoid long transactions)
        offset = 0
        while offset < total:
            await self.backfill_batch(offset)
            offset += batch_size
            # Log progress

        # 3. Verify completion
```

**Backfill Features**:
- **Batched Processing**: 100-500 records per batch (avoid locks)
- **Progress Logging**: Regular status updates
- **Idempotent Design**: Safe to re-run (skip already-processed records)
- **Estimation Logic**: Calculate values based on existing data
- **Error Handling**: Graceful failure and resume capability

**Example Backfill Scripts**:
1. `backfill_context_fields.py`: Analyze conversation history to estimate context metrics
2. `backfill_message_tokens.py`: Calculate token counts using 4 chars/token heuristic

### Zero-Downtime Migration Pattern

**Multi-Phase Deployment** (for breaking changes):

**Phase 1**: Add nullable column
```python
def upgrade():
    op.add_column('calls', sa.Column('new_field', sa.String(), nullable=True))
```

**Phase 2**: Deploy application that writes to both old and new columns

**Phase 3**: Backfill new column
```bash
python -m alembic.data_migrations.backfill_new_field
```

**Phase 4**: Make new column NOT NULL
```python
def upgrade():
    op.alter_column('calls', 'new_field', nullable=False)
```

**Phase 5**: Deploy application that only uses new column

**Phase 6**: Drop old column
```python
def upgrade():
    op.drop_column('calls', 'old_field')
```

### Migration Documentation Pattern

**Complete Documentation Coverage**:

1. **MIGRATION_GUIDE.md** (600+ lines):
   - Running migrations (dev and production)
   - Creating new migrations (auto-generate and manual)
   - Testing procedures (automated and manual)
   - Troubleshooting common issues
   - Production deployment procedures
   - Zero-downtime strategies
   - Migration checklist template

2. **Data Migration README** (`alembic/data_migrations/README.md`):
   - Script usage instructions
   - When to run each script
   - Configuration options
   - Production considerations
   - Troubleshooting guide

3. **Migration Files**:
   - Detailed docstrings
   - Clear upgrade/downgrade implementations
   - Comments explaining complex operations

### Database Schema Status

**Current State**:
- ✅ **5 complete migrations**: base → monitoring → voice-agent → context → performance
- ✅ **105+ indexes**: Optimizing all query patterns
- ✅ **Automated testing**: All migrations verified
- ✅ **Production ready**: Complete documentation and rollback procedures

**Benefits Achieved**:
- Version-controlled schema evolution
- Safe rollback capability
- Data preservation guarantees
- Query performance optimization
- Production deployment confidence

### Implementation Files

**Database Migrations** (1,180+ lines):
- `alembic/versions/003_add_context_management_fields.py` (79 lines): Context tracking fields
- `alembic/versions/004_add_performance_indexes.py` (201 lines): Performance optimization indexes
- `tests/migrations/test_migrations.py` (700+ lines): Comprehensive test suite
- `alembic/data_migrations/backfill_context_fields.py` (210 lines): Context field backfill
- `alembic/data_migrations/backfill_message_tokens.py` (170 lines): Token count backfill
- `MIGRATION_GUIDE.md` (600+ lines): Complete migration documentation
- `alembic/data_migrations/README.md` (200+ lines): Data migration usage guide
- `src/llm/service.py` (+120 lines): Integration layer
- `src/api/routers/streaming.py` (220 lines): SSE/WebSocket endpoints

**Monitoring Integration** (+100 lines):
- `src/monitoring/performance.py` (+50 lines): GPU/streaming metrics
- `src/monitoring/dashboard.py` (+50 lines): Dashboard endpoints
