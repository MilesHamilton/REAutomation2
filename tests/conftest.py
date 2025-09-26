"""
Pytest configuration and shared fixtures for REAutomation2 tests
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import time
from typing import Dict, Any, List

# Handle optional imports gracefully for testing
try:
    from src.agents.models import (
        AgentType, WorkflowState, WorkflowContext, AgentMessage, 
        AgentResponse, QualificationFactors
    )
except ImportError:
    # Create mock enums and classes for testing when dependencies are missing
    from enum import Enum
    from pydantic import BaseModel
    
    class AgentType(str, Enum):
        CONVERSATION = "conversation"
        QUALIFICATION = "qualification"
        OBJECTION_HANDLER = "objection_handler"
        SCHEDULER = "scheduler"
        ANALYTICS = "analytics"
    
    class WorkflowState(str, Enum):
        INITIALIZING = "initializing"
        GREETING = "greeting"
        QUALIFYING = "qualifying"
        HANDLING_OBJECTION = "handling_objection"
        SCHEDULING = "scheduling"
        CLOSING = "closing"
        COMPLETED = "completed"
        FAILED = "failed"
    
    class AgentMessage(BaseModel):
        agent_type: AgentType
        content: str
        timestamp: float
        confidence: float = 0.9
    
    class AgentResponse(BaseModel):
        agent_type: AgentType
        response_text: str = ""
        state_updates: Dict[str, Any] = {}
        should_escalate_tier: bool = False
        should_end_call: bool = False
        processing_time_ms: float = 0
    
    class WorkflowContext(BaseModel):
        call_id: str
        workflow_state: WorkflowState = WorkflowState.INITIALIZING
        conversation_history: List[AgentMessage] = []
        lead_data: Dict[str, Any] = {}
        qualification_score: float = 0.0
        created_at: float = 0.0
        updated_at: float = 0.0
        metadata: Dict[str, Any] = {}
    
    class QualificationFactors(BaseModel):
        intent: float = 0.0
        budget: float = 0.0
        timeline: float = 0.0
        authority: float = 0.0
        needs: float = 0.0
        
        @property
        def overall_score(self) -> float:
            return (self.intent + self.budget + self.timeline + self.authority + self.needs) / 5.0

try:
    from src.llm.models import LLMRequest, LLMResponse, Message, ConversationContext, MessageRole
except ImportError:
    # Create mock LLM models for testing
    from enum import Enum
    from pydantic import BaseModel
    
    class MessageRole(str, Enum):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
    
    class Message(BaseModel):
        role: MessageRole
        content: str
        timestamp: float = None
    
    class LLMRequest(BaseModel):
        messages: List[Message]
        max_tokens: int = 150
        temperature: float = 0.7
        system_prompt: str = None
        structured_output: bool = False
        response_format: Dict[str, Any] = None
    
    class LLMResponse(BaseModel):
        content: str
        usage_tokens: int = 0
        response_time_ms: float = 0
        model_used: str = ""
        confidence_score: float = None
        structured_data: Dict[str, Any] = None
    
    class ConversationContext(BaseModel):
        call_id: str
        messages: List[Message] = []
        lead_info: Dict[str, Any] = {}
        qualification_score: float = None
        conversation_state: str = "initial"

# Handle additional missing attributes for WorkflowContext
try:
    # Check if WorkflowContext has all required attributes
    sample_context = WorkflowContext(call_id="test")
    if not hasattr(sample_context, 'objection_count'):
        # Add missing attributes to the mock class
        WorkflowContext.model_fields.update({
            'objection_count': (int, 0),
            'scheduling_attempts': (int, 0),
            'tier_escalated': (bool, False)
        })
except:
    # If WorkflowContext doesn't exist or has issues, ensure our mock has all attributes
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing"""
    service = AsyncMock()
    
    # Default response for generate_response
    service.generate_response.return_value = LLMResponse(
        content="Hello! I'm calling from ABC Realty. How are you today?",
        usage_tokens=25,
        response_time_ms=150.0,
        model_used="llama3.1:8b",
        confidence_score=0.9
    )
    
    # Default response for generate_structured_response
    service.generate_structured_response.return_value = LLMResponse(
        content='{"qualification_score": 0.7, "confidence": 0.8}',
        usage_tokens=30,
        response_time_ms=200.0,
        model_used="llama3.1:8b",
        structured_data={
            "qualification_score": 0.7,
            "confidence": 0.8,
            "factors": {
                "intent": 0.8,
                "budget": 0.6,
                "timeline": 0.7,
                "authority": 0.8,
                "needs": 0.7
            },
            "reasoning": "Lead shows strong interest and has decision-making authority",
            "recommended_action": "continue"
        }
    )
    
    return service


@pytest.fixture
def sample_workflow_context():
    """Sample workflow context for testing"""
    return WorkflowContext(
        call_id="test-call-123",
        workflow_state=WorkflowState.GREETING,
        conversation_history=[
            AgentMessage(
                agent_type=AgentType.CONVERSATION,
                content="Hello! I'm calling from ABC Realty.",
                timestamp=time.time() - 10
            )
        ],
        lead_data={
            "name": "John Doe",
            "phone": "+1234567890",
            "company": "Test Corp",
            "industry": "Technology"
        },
        qualification_score=0.0,
        objection_count=0,
        scheduling_attempts=0,
        tier_escalated=False,
        created_at=time.time() - 60,
        updated_at=time.time() - 10,
        metadata={}
    )


@pytest.fixture
def sample_conversation_context():
    """Sample conversation context for LLM service"""
    return ConversationContext(
        call_id="test-call-123",
        messages=[
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi! How can I help you today?")
        ],
        lead_info={
            "name": "John Doe",
            "phone": "+1234567890"
        },
        qualification_score=0.5,
        conversation_state="greeting"
    )


@pytest.fixture
def sample_qualification_factors():
    """Sample qualification factors for testing"""
    return QualificationFactors(
        intent=0.8,
        budget=0.6,
        timeline=0.7,
        authority=0.8,
        needs=0.7
    )


@pytest.fixture
def mock_agent_response():
    """Mock agent response for testing"""
    return AgentResponse(
        agent_type=AgentType.CONVERSATION,
        response_text="Thank you for your interest! Can you tell me more about what you're looking for?",
        state_updates={
            "workflow_state": WorkflowState.QUALIFYING,
            "qualification_score": 0.3
        },
        should_escalate_tier=False,
        should_end_call=False,
        processing_time_ms=250.0
    )


@pytest.fixture
def mock_integration_service():
    """Mock integration service for testing"""
    service = AsyncMock()
    
    # Mock contact retrieval
    service.get_contacts.return_value = [
        {
            "name": "John Doe",
            "phone": "+1234567890",
            "status": "new",
            "company": "Test Corp"
        },
        {
            "name": "Jane Smith", 
            "phone": "+0987654321",
            "status": "contacted",
            "company": "Smith LLC"
        }
    ]
    
    # Mock status update
    service.update_contact_status.return_value = True
    
    # Mock result recording
    service.record_call_result.return_value = True
    
    return service


@pytest.fixture
def sample_lead_data():
    """Sample lead data for testing"""
    return {
        "name": "John Doe",
        "phone": "+1234567890",
        "email": "john.doe@example.com",
        "company": "Test Corporation",
        "industry": "Technology",
        "location": "San Francisco, CA",
        "source": "website_form",
        "notes": "Interested in commercial real estate"
    }


@pytest.fixture
def sample_objection_scenarios():
    """Sample objection scenarios for testing"""
    return [
        {
            "user_input": "I'm not interested right now",
            "expected_type": "need",
            "expected_severity": 2
        },
        {
            "user_input": "Your prices are too high",
            "expected_type": "price", 
            "expected_severity": 3
        },
        {
            "user_input": "I need to talk to my partner first",
            "expected_type": "authority",
            "expected_severity": 2
        },
        {
            "user_input": "I don't have time for this",
            "expected_type": "time",
            "expected_severity": 3
        },
        {
            "user_input": "I'm already working with another agent",
            "expected_type": "competition",
            "expected_severity": 4
        }
    ]


@pytest.fixture
def sample_scheduling_slots():
    """Sample scheduling slots for testing"""
    return [
        {
            "datetime": "2024-01-15T10:00:00Z",
            "duration_minutes": 30,
            "timezone": "America/New_York",
            "available": True,
            "priority": 1
        },
        {
            "datetime": "2024-01-15T14:00:00Z", 
            "duration_minutes": 30,
            "timezone": "America/New_York",
            "available": True,
            "priority": 2
        },
        {
            "datetime": "2024-01-16T09:00:00Z",
            "duration_minutes": 60,
            "timezone": "America/New_York", 
            "available": True,
            "priority": 1
        }
    ]


class MockLLMResponse:
    """Helper class for creating mock LLM responses"""
    
    @staticmethod
    def conversation_response(content: str, confidence: float = 0.9) -> LLMResponse:
        return LLMResponse(
            content=content,
            usage_tokens=len(content.split()) * 2,
            response_time_ms=150.0,
            model_used="llama3.1:8b",
            confidence_score=confidence
        )
    
    @staticmethod
    def qualification_response(score: float, factors: Dict[str, float]) -> LLMResponse:
        structured_data = {
            "qualification_score": score,
            "confidence": 0.8,
            "factors": factors,
            "reasoning": f"Lead qualification score: {score}",
            "recommended_action": "continue" if score > 0.5 else "disqualify"
        }
        
        return LLMResponse(
            content=str(structured_data),
            usage_tokens=50,
            response_time_ms=200.0,
            model_used="llama3.1:8b",
            structured_data=structured_data
        )
    
    @staticmethod
    def objection_response(objection_type: str, response_text: str) -> LLMResponse:
        structured_data = {
            "objection_detected": True,
            "objection_type": objection_type,
            "severity": 3,
            "response_strategy": "acknowledge_and_redirect",
            "suggested_response": response_text
        }
        
        return LLMResponse(
            content=response_text,
            usage_tokens=40,
            response_time_ms=180.0,
            model_used="llama3.1:8b",
            structured_data=structured_data
        )


# Test utilities
class TestHelpers:
    """Helper methods for tests"""
    
    @staticmethod
    def create_agent_message(agent_type: AgentType, content: str) -> AgentMessage:
        return AgentMessage(
            agent_type=agent_type,
            content=content,
            timestamp=time.time(),
            confidence=0.9
        )
    
    @staticmethod
    def create_workflow_context(
        call_id: str = "test-call",
        state: WorkflowState = WorkflowState.GREETING,
        messages: List[AgentMessage] = None
    ) -> WorkflowContext:
        return WorkflowContext(
            call_id=call_id,
            workflow_state=state,
            conversation_history=messages or [],
            lead_data={"name": "Test User", "phone": "+1234567890"},
            created_at=time.time(),
            updated_at=time.time()
        )


@pytest.fixture
async def db_session():
    """Create a test database session"""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        from src.database.models import Base
        
        # Use in-memory SQLite for testing
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False
        )
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Create session factory
        async_session_factory = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Create and yield session
        async with async_session_factory() as session:
            yield session
            
        # Cleanup
        await engine.dispose()
        
    except ImportError:
        # If database dependencies are missing, create a mock session
        from unittest.mock import AsyncMock
        mock_session = AsyncMock()
        mock_session.execute.return_value.scalar.return_value = 1
        mock_session.execute.return_value.fetchone.return_value = {"id": 1}
        mock_session.execute.return_value.fetchall.return_value = []
        yield mock_session


@pytest.fixture
def mock_audio_data():
    """Mock audio data for testing"""
    import numpy as np
    
    # Generate sample audio data
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, samples)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.1 * 32767).astype(np.int16)
    
    return {
        "raw_audio": audio.tobytes(),
        "wav_audio": audio.tobytes(),
        "sample_rate": sample_rate,
        "duration": duration,
        "samples": samples
    }


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing"""
    from unittest.mock import AsyncMock
    
    websocket = AsyncMock()
    websocket.send.return_value = None
    websocket.receive.return_value = {"type": "websocket.receive", "bytes": b"test"}
    websocket.close.return_value = None
    
    return websocket


@pytest.fixture
def test_helpers():
    """Provide test helper methods"""
    return TestHelpers
