# Technical Requirements (TR)
## LangGraph Orchestration + Pipecat/Twilio Integration

### Architecture Overview

This document outlines the technical implementation requirements for integrating the LangGraph agent orchestration system with the Pipecat/Twilio voice pipeline.

### System Architecture Changes

#### Current Architecture
```
Twilio WebRTC → Pipecat → STT → ConversationProcessor → Direct LLM → TTS → Pipecat → Twilio
```

#### Target Architecture
```
Twilio WebRTC → Pipecat → STT → ConversationProcessor → AgentOrchestrator → LangGraph Workflow → Agent Response → TTS → Pipecat → Twilio
                                                                    ↓
                                                            State Synchronization
                                                                    ↓
                                                        CallSession ↔ WorkflowContext
```

### Technical Implementation Requirements

#### TR-001: ConversationProcessor Enhancement
**Component:** `src/voice/pipecat_integration.py`
**Class:** `ConversationProcessor`

**Required Changes:**
1. **Agent Orchestrator Integration**
   ```python
   class ConversationProcessor(FrameProcessor):
       def __init__(self, call_session: CallSession, agent_orchestrator: AgentOrchestrator):
           super().__init__()
           self.call_session = call_session
           self.agent_orchestrator = agent_orchestrator
           self.workflow_context = None
           self._fallback_to_direct_llm = False
   ```

2. **Enhanced Input Processing**
   ```python
   async def _handle_user_input(self, user_text: str):
       try:
           # Route through agent orchestration
           agent_response = await self.agent_orchestrator.process_input(
               call_id=self.call_session.call_id,
               user_input=user_text,
               lead_data=self.call_session.lead_data
           )
           
           if agent_response and agent_response.response_text:
               # Update call session with agent info
               self.call_session.current_agent = agent_response.agent_type
               
               # Sync state with workflow context
               await self._sync_workflow_state(agent_response)
               
               # Send response to TTS
               await self.push_frame(TextFrame(text=agent_response.response_text))
               
               # Check for tier escalation
               if agent_response.should_escalate_tier:
                   await self._handle_tier_escalation(agent_response)
           
       except Exception as e:
           logger.error(f"Agent orchestration failed: {e}")
           await self._fallback_to_direct_llm(user_text)
   ```

3. **State Synchronization Methods**
   ```python
   async def _sync_workflow_state(self, agent_response: AgentResponse):
       """Synchronize CallSession with WorkflowContext"""
       workflow_context = self.agent_orchestrator.get_context(self.call_session.call_id)
       if workflow_context:
           # Sync qualification score
           self.call_session.metrics.qualification_score = workflow_context.qualification_score
           
           # Sync workflow state
           self.call_session.workflow_state = workflow_context.workflow_state
           
           # Record agent transition
           if hasattr(agent_response, 'previous_agent'):
               transition = AgentTransition(
                   from_agent=agent_response.previous_agent,
                   to_agent=agent_response.agent_type,
                   timestamp=time.time(),
                   trigger=agent_response.transition_reason
               )
               self.call_session.agent_transition_history.append(transition)
   ```

#### TR-002: Enhanced Data Models
**Component:** `src/voice/models.py`

**Required Model Extensions:**
1. **CallSession Enhancement**
   ```python
   @dataclass
   class CallSession:
       # Existing fields...
       call_id: str
       phone_number: str
       current_tier: TTSProvider
       tts_config: TTSConfig
       metrics: VoiceMetrics
       lead_data: Dict[str, Any]
       
       # New fields for agent integration
       workflow_context_id: Optional[str] = None
       current_agent: Optional[AgentType] = None
       agent_transition_history: List[AgentTransition] = field(default_factory=list)
       workflow_state: Optional[WorkflowState] = None
       last_state_sync: Optional[float] = None
       integration_enabled: bool = True
   ```

2. **Agent Transition Model**
   ```python
   @dataclass
   class AgentTransition:
       from_agent: Optional[AgentType]
       to_agent: AgentType
       timestamp: float
       trigger: str
       context_preserved: bool = True
       transition_duration_ms: Optional[float] = None
   ```

3. **Integration Context Model**
   ```python
   @dataclass
   class VoiceAgentIntegrationContext:
       call_id: str
       voice_session: CallSession
       workflow_context: Optional[WorkflowContext]
       sync_status: SyncStatus
       last_sync_timestamp: float
       error_count: int = 0
       fallback_active: bool = False
   ```

#### TR-003: Agent Orchestrator Enhancements
**Component:** `src/agents/orchestrator.py`
**Class:** `AgentOrchestrator`

**Required Enhancements:**
1. **Voice-Specific Processing Method**
   ```python
   async def process_voice_input(
       self,
       call_id: str,
       user_input: str,
       call_session: CallSession,
       lead_data: Optional[Dict[str, Any]] = None
   ) -> Optional[AgentResponse]:
       """Process voice input with enhanced context"""
       try:
           # Get or create workflow context with voice-specific data
           context = self._get_or_create_voice_context(call_id, call_session, lead_data)
           
           # Add voice-specific metadata
           context.metadata.update({
               "input_modality": "voice",
               "current_tier": call_session.current_tier.value,
               "call_duration": time.time() - call_session.started_at,
               "previous_agent": call_session.current_agent
           })
           
           # Process through workflow
           result = await self.workflow_graph.ainvoke(context, {"configurable": {"thread_id": call_id}})
           
           # Extract and enhance response for voice
           if hasattr(result, 'metadata') and 'agent_response' in result.metadata:
               agent_response = result.metadata['agent_response']
               
               # Add voice-specific response enhancements
               agent_response.response_text = self._optimize_for_voice(agent_response.response_text)
               agent_response.should_escalate_tier = self._check_tier_escalation_needed(result)
               
               return agent_response
           
           return None
           
       except Exception as e:
           logger.error(f"Voice input processing failed for call {call_id}: {e}")
           return self._create_fallback_response(user_input)
   ```

2. **Voice Context Creation**
   ```python
   def _get_or_create_voice_context(
       self,
       call_id: str,
       call_session: CallSession,
       lead_data: Optional[Dict[str, Any]] = None
   ) -> WorkflowContext:
       """Create workflow context optimized for voice interactions"""
       if call_id in self.active_contexts:
           context = self.active_contexts[call_id]
           # Update with latest call session data
           context.metadata["call_session"] = call_session
           return context
       
       context = WorkflowContext(
           call_id=call_id,
           lead_data=lead_data or call_session.lead_data,
           workflow_state=WorkflowState.INITIALIZING,
           metadata={
               "call_session": call_session,
               "voice_enabled": True,
               "tier_escalation_enabled": True,
               "performance_mode": "real_time"
           }
       )
       
       self.active_contexts[call_id] = context
       return context
   ```

#### TR-004: Voice Pipeline Integration
**Component:** `src/voice/pipeline.py`
**Class:** `VoicePipeline`

**Required Integration Points:**
1. **Agent Orchestrator Injection**
   ```python
   class VoicePipeline:
       def __init__(self, use_pipecat: bool = True, enable_agent_integration: bool = True):
           # Existing initialization...
           self.enable_agent_integration = enable_agent_integration
           self.agent_orchestrator = None
           
       async def initialize(self):
           """Enhanced initialization with agent orchestrator"""
           # Existing initialization...
           
           if self.enable_agent_integration:
               from ..agents.orchestrator import agent_orchestrator
               self.agent_orchestrator = agent_orchestrator
               
               if not self.agent_orchestrator.is_initialized:
                   orchestrator_ready = await self.agent_orchestrator.initialize()
                   if not orchestrator_ready:
                       logger.warning("Agent orchestrator initialization failed, using fallback mode")
                       self.enable_agent_integration = False
   ```

2. **Enhanced Call Startup**
   ```python
   async def start_call(self, call_id: str, phone_number: str, **kwargs) -> bool:
       """Enhanced call startup with agent integration"""
       # Existing call startup logic...
       
       if self.enable_agent_integration and self.agent_orchestrator:
           # Set up tier escalation callback
           self.agent_orchestrator.on_tier_escalation(
               lambda cid, score: self._handle_agent_tier_escalation(cid, score)
           )
           
           # Set up agent transition callback
           self.agent_orchestrator.on_agent_transition(
               lambda cid, from_agent, to_agent: self._handle_agent_transition(cid, from_agent, to_agent)
           )
   ```

#### TR-005: Tier Escalation Integration
**Component:** Multiple components
**Integration Points:** Agent orchestration → Voice pipeline

**Implementation Requirements:**
1. **Callback Registration**
   ```python
   # In VoicePipeline initialization
   async def _setup_agent_callbacks(self):
       """Set up callbacks for agent-driven events"""
       if self.agent_orchestrator:
           self.agent_orchestrator.on_tier_escalation(self._handle_tier_escalation_request)
           self.agent_orchestrator.on_workflow_complete(self._handle_workflow_completion)
   
   async def _handle_tier_escalation_request(self, call_id: str, qualification_score: float):
       """Handle tier escalation request from agent orchestrator"""
       try:
           if call_id in self.active_calls:
               call_session = self.active_calls[call_id]
               
               # Check budget constraints
               if await self._check_escalation_budget(call_session):
                   success = await self.switch_tier(call_id, TTSProvider.ELEVENLABS, "agent_qualification")
                   
                   if success:
                       logger.info(f"Tier escalated for call {call_id} based on qualification score {qualification_score}")
                   else:
                       logger.error(f"Tier escalation failed for call {call_id}")
               else:
                   logger.warning(f"Tier escalation blocked for call {call_id} due to budget constraints")
                   
       except Exception as e:
           logger.error(f"Error handling tier escalation for call {call_id}: {e}")
   ```

2. **Budget Integration**
   ```python
   async def _check_escalation_budget(self, call_session: CallSession) -> bool:
       """Check if tier escalation is within budget"""
       from ..cost_control.budget_manager import budget_manager
       
       estimated_additional_cost = self._calculate_escalation_cost(call_session)
       return await budget_manager.can_afford_escalation(estimated_additional_cost)
   ```

#### TR-006: Performance Optimization
**Component:** Multiple components
**Focus:** Real-time performance requirements

**Optimization Requirements:**
1. **Async Processing Pipeline**
   ```python
   class OptimizedConversationProcessor(ConversationProcessor):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self._processing_queue = asyncio.Queue(maxsize=10)
           self._response_cache = {}
           
       async def _handle_user_input_optimized(self, user_text: str):
           """Optimized input handling with caching and parallel processing"""
           # Check response cache first
           cache_key = self._generate_cache_key(user_text, self.call_session.current_agent)
           if cache_key in self._response_cache:
               cached_response = self._response_cache[cache_key]
               if self._is_cache_valid(cached_response):
                   await self.push_frame(TextFrame(text=cached_response.text))
                   return
           
           # Process with timeout
           try:
               agent_response = await asyncio.wait_for(
                   self.agent_orchestrator.process_voice_input(
                       self.call_session.call_id,
                       user_text,
                       self.call_session
                   ),
                   timeout=1.5  # 1.5 second timeout for real-time requirements
               )
               
               if agent_response:
                   # Cache successful responses
                   self._cache_response(cache_key, agent_response)
                   await self.push_frame(TextFrame(text=agent_response.response_text))
               
           except asyncio.TimeoutError:
               logger.warning(f"Agent processing timeout for call {self.call_session.call_id}")
               await self._fallback_to_direct_llm(user_text)
   ```

2. **Connection Pooling and Resource Management**
   ```python
   class VoiceResourceManager:
       def __init__(self):
           self.agent_pool = asyncio.Queue(maxsize=50)  # Pre-initialized agent contexts
           self.tts_connection_pool = {}
           self.redis_pool = None
           
       async def get_agent_context(self, call_id: str) -> AgentContext:
           """Get pre-warmed agent context"""
           try:
               context = await asyncio.wait_for(self.agent_pool.get(), timeout=0.1)
               context.call_id = call_id
               return context
           except asyncio.TimeoutError:
               # Create new context if pool is empty
               return AgentContext(call_id=call_id)
   ```

#### TR-007: Error Handling and Resilience
**Component:** All integration components
**Pattern:** Circuit breaker and graceful degradation

**Implementation Requirements:**
1. **Circuit Breaker Pattern**
   ```python
   class AgentOrchestrationCircuitBreaker:
       def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
           self.failure_threshold = failure_threshold
           self.recovery_timeout = recovery_timeout
           self.failure_count = 0
           self.last_failure_time = 0
           self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
           
       async def call_with_circuit_breaker(self, func, *args, **kwargs):
           """Execute function with circuit breaker protection"""
           if self.state == "OPEN":
               if time.time() - self.last_failure_time > self.recovery_timeout:
                   self.state = "HALF_OPEN"
               else:
                   raise CircuitBreakerOpenError("Agent orchestration circuit breaker is OPEN")
           
           try:
               result = await func(*args, **kwargs)
               if self.state == "HALF_OPEN":
                   self.state = "CLOSED"
                   self.failure_count = 0
               return result
               
           except Exception as e:
               self.failure_count += 1
               self.last_failure_time = time.time()
               
               if self.failure_count >= self.failure_threshold:
                   self.state = "OPEN"
                   
               raise e
   ```

2. **Graceful Degradation**
   ```python
   class GracefulDegradationManager:
       def __init__(self):
           self.degradation_levels = {
               "FULL": {"agent_orchestration": True, "tier_escalation": True, "analytics": True},
               "PARTIAL": {"agent_orchestration": True, "tier_escalation": False, "analytics": False},
               "MINIMAL": {"agent_orchestration": False, "tier_escalation": False, "analytics": False}
           }
           self.current_level = "FULL"
           
       async def handle_component_failure(self, component: str, error: Exception):
           """Handle component failure with appropriate degradation"""
           if component == "agent_orchestrator":
               self.current_level = "MINIMAL"
               logger.error(f"Agent orchestrator failed, degrading to minimal mode: {error}")
           elif component == "tier_escalation":
               if self.current_level == "FULL":
                   self.current_level = "PARTIAL"
               logger.warning(f"Tier escalation failed, degrading service: {error}")
   ```

#### TR-008: Monitoring and Observability
**Component:** Integration monitoring
**Requirements:** Comprehensive metrics and tracing

**Implementation Requirements:**
1. **Integration Metrics**
   ```python
   class VoiceAgentIntegrationMetrics:
       def __init__(self):
           self.metrics = {
               "agent_processing_latency": [],
               "tier_escalation_rate": 0.0,
               "agent_transition_count": 0,
               "fallback_activation_count": 0,
               "state_sync_failures": 0,
               "circuit_breaker_trips": 0
           }
           
       async def record_agent_processing_time(self, call_id: str, processing_time_ms: float):
           """Record agent processing latency"""
           self.metrics["agent_processing_latency"].append({
               "call_id": call_id,
               "timestamp": time.time(),
               "latency_ms": processing_time_ms
           })
           
       async def record_tier_escalation(self, call_id: str, from_tier: str, to_tier: str, trigger: str):
           """Record tier escalation event"""
           escalation_event = {
               "call_id": call_id,
               "timestamp": time.time(),
               "from_tier": from_tier,
               "to_tier": to_tier,
               "trigger": trigger
           }
           # Send to monitoring system
           await self._send_to_monitoring(escalation_event)
   ```

2. **Distributed Tracing**
   ```python
   from opentelemetry import trace
   
   class VoiceAgentTracing:
       def __init__(self):
           self.tracer = trace.get_tracer(__name__)
           
       async def trace_agent_processing(self, call_id: str, user_input: str):
           """Trace agent processing with distributed tracing"""
           with self.tracer.start_as_current_span("agent_processing") as span:
               span.set_attribute("call_id", call_id)
               span.set_attribute("input_length", len(user_input))
               
               try:
                   # Agent processing logic
                   result = await self._process_with_agents(call_id, user_input)
                   span.set_attribute("success", True)
                   return result
               except Exception as e:
                   span.set_attribute("success", False)
                   span.set_attribute("error", str(e))
                   raise
   ```

#### TR-009: Configuration Management
**Component:** Configuration system
**Requirements:** Runtime configuration for integration parameters

**Configuration Schema:**
```python
@dataclass
class VoiceAgentIntegrationConfig:
    # Agent integration settings
    enable_agent_integration: bool = True
    agent_processing_timeout_ms: int = 1500
    fallback_to_direct_llm: bool = True
    
    # Tier escalation settings
    qualification_threshold: float = 0.7
    enable_tier_escalation: bool = True
    escalation_budget_check: bool = True
    
    # Performance settings
    max_concurrent_agent_calls: int = 50
    agent_response_cache_size: int = 1000
    cache_ttl_seconds: int = 300
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    
    # Monitoring settings
    enable_detailed_metrics: bool = True
    trace_agent_processing: bool = True
    metrics_export_interval: int = 30
```

#### TR-010: Database Schema Changes
**Component:** Database models
**Requirements:** Support for agent integration data

**Schema Extensions:**
```sql
-- Add agent integration fields to existing call records
ALTER TABLE calls ADD COLUMN workflow_context_id VARCHAR(255);
ALTER TABLE calls ADD COLUMN current_agent VARCHAR(50);
ALTER TABLE calls ADD COLUMN agent_transition_count INTEGER DEFAULT 0;
ALTER TABLE calls ADD COLUMN tier_escalation_trigger VARCHAR(100);

-- New table for agent transitions
CREATE TABLE agent_transitions (
    id SERIAL PRIMARY KEY,
    call_id VARCHAR(255) NOT NULL,
    from_agent VARCHAR(50),
    to_agent VARCHAR(50) NOT NULL,
    transition_timestamp TIMESTAMP NOT NULL,
    trigger_reason VARCHAR(255),
    context_preserved BOOLEAN DEFAULT TRUE,
    transition_duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- New table for tier escalation events
CREATE TABLE tier_escalation_events (
    id SERIAL PRIMARY KEY,
    call_id VARCHAR(255) NOT NULL,
    trigger_agent VARCHAR(50),
    qualification_score DECIMAL(3,2),
    from_tier VARCHAR(50) NOT NULL,
    to_tier VARCHAR(50) NOT NULL,
    escalation_timestamp TIMESTAMP NOT NULL,
    cost_impact DECIMAL(10,4),
    budget_check_passed BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_agent_transitions_call_id ON agent_transitions(call_id);
CREATE INDEX idx_tier_escalations_call_id ON tier_escalation_events(call_id);
CREATE INDEX idx_calls_workflow_context ON calls(workflow_context_id);
```

### Implementation Dependencies

#### Internal Dependencies
- **LangGraph 0.6.8+**: Agent orchestration framework
- **Pipecat**: Real-time audio processing pipeline
- **FastAPI**: API framework for integration endpoints
- **Redis**: State synchronization and caching
- **PostgreSQL**: Persistent data storage

#### External Dependencies
- **Twilio WebRTC**: Voice communication infrastructure
- **OpenAI Whisper**: Speech-to-text processing
- **11Labs API**: Premium text-to-speech service
- **Ollama**: Local LLM inference

### Performance Targets

#### Latency Requirements
- **Agent Processing**: <500ms per interaction
- **State Synchronization**: <100ms per update
- **Tier Escalation**: <200ms for TTS switch
- **Total Response Time**: <2000ms end-to-end

#### Throughput Requirements
- **Concurrent Calls**: Up to 50 simultaneous calls
- **Agent Transitions**: Up to 100 per second
- **State Updates**: Up to 200 per second

#### Resource Requirements
- **Memory Overhead**: <100MB per active call
- **CPU Overhead**: <10% per active call
- **Network Bandwidth**: <64kbps per call

### Security Requirements

#### Data Protection
- Encrypt all conversation data in transit using TLS 1.3
- Encrypt stored conversation data using AES-256
- Implement PII detection and masking
- Maintain audit logs for all data access

#### API Security
- Implement OAuth 2.0 for API authentication
- Use JWT tokens for session management
- Rate limiting for API endpoints
- Input validation and sanitization

#### Network Security
- Secure WebSocket connections with WSS
- VPN or private network for internal communication
- Firewall rules for component isolation
- Regular security audits and penetration testing

### Testing Strategy

#### Unit Testing
- Test individual integration components
- Mock external dependencies
- Validate error handling scenarios
- Performance benchmarking

#### Integration Testing
- End-to-end conversation flows
- Agent transition scenarios
- Tier escalation processes
- State synchronization validation

#### Load Testing
- Concurrent call processing
- Resource utilization under load
- Performance degradation thresholds
- Scalability limits

#### Chaos Engineering
- Component failure simulation
- Network partition testing
- Resource exhaustion scenarios
- Recovery time validation

### Deployment Requirements

#### Infrastructure
- Kubernetes cluster for container orchestration
- Redis cluster for state management
- PostgreSQL cluster for data persistence
- Load balancers for traffic distribution

#### Monitoring
- Prometheus for metrics collection
- Grafana for visualization
- Jaeger for distributed tracing
- ELK stack for log aggregation

#### CI/CD Pipeline
- Automated testing on code changes
- Staged deployment process
- Rollback capabilities
- Feature flag management

### Migration Strategy

#### Phase 1: Foundation (Week 1-2)
- Implement enhanced data models
- Set up monitoring and observability
- Create configuration management system

#### Phase 2: Core Integration (Week 3-4)
- Integrate ConversationProcessor with AgentOrchestrator
- Implement state synchronization
- Add error handling and circuit breakers

#### Phase 3: Advanced Features (Week 5-6)
- Implement tier escalation integration
- Add performance optimizations
- Complete monitoring and alerting

#### Phase 4: Testing and Deployment (Week 7-8)
- Comprehensive testing
- Performance tuning
- Production deployment with feature flags

### Rollback Plan

#### Immediate Rollback
- Feature flag to disable agent integration
- Automatic fallback to direct LLM processing
- Preserve existing voice pipeline functionality

#### Data Recovery
- Backup conversation state before migration
- Ability to restore previous system state
- Data consistency validation tools

#### Monitoring and Alerts
- Real-time system health monitoring
- Automated rollback triggers
- Operations team notification system
