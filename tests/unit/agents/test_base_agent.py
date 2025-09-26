"""
Unit tests for BaseAgent abstract class
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Handle optional imports gracefully for testing
try:
    from src.agents.base_agent import BaseAgent
    from src.agents.models import (
        AgentType, AgentState, AgentResponse, WorkflowContext, 
        WorkflowState, AgentDecision
    )
    from src.llm.models import LLMResponse, ConversationContext
except ImportError:
    # Skip tests if dependencies are not available
    pytest.skip("Agent dependencies not available", allow_module_level=True)


class TestableAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing"""
    
    def __init__(self):
        super().__init__(AgentType.CONVERSATION)
        self.process_called = False
        self.system_prompt_called = False
    
    async def process(self, context: WorkflowContext, user_input: str) -> AgentResponse:
        self.process_called = True
        return AgentResponse(
            agent_type=self.agent_type,
            response_text="Test response",
            processing_time_ms=100.0
        )
    
    def get_system_prompt(self, context: WorkflowContext) -> str:
        self.system_prompt_called = True
        return "Test system prompt"


class TestBaseAgent:
    """Test suite for BaseAgent"""

    @pytest.fixture
    def agent(self):
        """Create a testable agent instance"""
        return TestableAgent()

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response"""
        return LLMResponse(
            content="Generated response",
            usage_tokens=25,
            response_time_ms=150.0,
            model_used="llama3.1:8b",
            confidence_score=0.9
        )

    @pytest.fixture
    def mock_structured_response(self):
        """Mock structured LLM response"""
        return LLMResponse(
            content='{"action": "continue", "confidence": 0.8}',
            usage_tokens=30,
            response_time_ms=200.0,
            model_used="llama3.1:8b",
            structured_data={
                "action": "continue",
                "reasoning": "User shows interest",
                "confidence": 0.8,
                "parameters": {}
            }
        )

    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.agent_type == AgentType.CONVERSATION
        assert agent.state == AgentState.IDLE
        assert agent._processing_lock is not None

    def test_can_handle_default(self, agent, sample_workflow_context):
        """Test default can_handle implementation"""
        assert agent.can_handle(sample_workflow_context) is True

    def test_get_confidence_threshold_default(self, agent):
        """Test default confidence threshold"""
        assert agent.get_confidence_threshold() == 0.7

    @pytest.mark.asyncio
    async def test_execute_with_timeout_success(self, agent, sample_workflow_context):
        """Test successful execution with timeout"""
        user_input = "Hello"
        
        response = await agent.execute_with_timeout(
            context=sample_workflow_context,
            user_input=user_input,
            timeout_seconds=5.0
        )
        
        assert response.agent_type == AgentType.CONVERSATION
        assert response.response_text == "Test response"
        assert response.processing_time_ms > 0
        assert agent.state == AgentState.COMPLETED
        assert agent.process_called is True

    @pytest.mark.asyncio
    async def test_execute_with_timeout_timeout_error(self, agent, sample_workflow_context):
        """Test execution timeout handling"""
        # Mock process to take longer than timeout
        async def slow_process(context, user_input):
            await asyncio.sleep(2.0)
            return AgentResponse(agent_type=AgentType.CONVERSATION, response_text="Slow response")
        
        agent.process = slow_process
        
        response = await agent.execute_with_timeout(
            context=sample_workflow_context,
            user_input="Hello",
            timeout_seconds=0.1  # Very short timeout
        )
        
        assert response.agent_type == AgentType.CONVERSATION
        assert "moment to process" in response.response_text
        assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_execute_with_timeout_exception(self, agent, sample_workflow_context):
        """Test execution exception handling"""
        # Mock process to raise exception
        async def failing_process(context, user_input):
            raise ValueError("Test error")
        
        agent.process = failing_process
        
        response = await agent.execute_with_timeout(
            context=sample_workflow_context,
            user_input="Hello"
        )
        
        assert response.agent_type == AgentType.CONVERSATION
        assert "encountered an issue" in response.response_text
        assert agent.state == AgentState.ERROR

    @pytest.mark.asyncio
    async def test_generate_llm_response_success(self, agent, sample_workflow_context, mock_llm_response):
        """Test successful LLM response generation"""
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.return_value = mock_llm_response
            
            response = await agent._generate_llm_response(
                context=sample_workflow_context,
                user_input="Hello"
            )
            
            assert response == "Generated response"
            mock_service.generate_response.assert_called_once()
            
            # Verify conversation context was created properly
            call_args = mock_service.generate_response.call_args
            conv_context = call_args[1]['context']
            assert isinstance(conv_context, ConversationContext)
            assert conv_context.call_id == sample_workflow_context.call_id

    @pytest.mark.asyncio
    async def test_generate_llm_response_with_custom_prompt(self, agent, sample_workflow_context, mock_llm_response):
        """Test LLM response generation with custom system prompt"""
        custom_prompt = "Custom system prompt"
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.return_value = mock_llm_response
            
            response = await agent._generate_llm_response(
                context=sample_workflow_context,
                user_input="Hello",
                system_prompt=custom_prompt
            )
            
            assert response == "Generated response"
            
            # Verify custom prompt was used
            call_args = mock_service.generate_response.call_args
            assert call_args[1]['system_prompt'] == custom_prompt
            assert agent.system_prompt_called is False  # Should not call get_system_prompt

    @pytest.mark.asyncio
    async def test_generate_llm_response_failure(self, agent, sample_workflow_context):
        """Test LLM response generation failure"""
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.side_effect = Exception("LLM error")
            
            response = await agent._generate_llm_response(
                context=sample_workflow_context,
                user_input="Hello"
            )
            
            assert response is None

    @pytest.mark.asyncio
    async def test_generate_llm_response_no_response(self, agent, sample_workflow_context):
        """Test LLM response generation when service returns None"""
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.return_value = None
            
            response = await agent._generate_llm_response(
                context=sample_workflow_context,
                user_input="Hello"
            )
            
            assert response is None

    @pytest.mark.asyncio
    async def test_generate_structured_response_success(self, agent, sample_workflow_context, mock_structured_response):
        """Test successful structured response generation"""
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "confidence": {"type": "number"}
            }
        }
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_structured_response.return_value = mock_structured_response
            
            response = await agent._generate_structured_response(
                context=sample_workflow_context,
                user_input="Hello",
                response_schema=schema
            )
            
            assert response == mock_structured_response.structured_data
            mock_service.generate_structured_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_structured_response_failure(self, agent, sample_workflow_context):
        """Test structured response generation failure"""
        schema = {"type": "object"}
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_structured_response.side_effect = Exception("Structured error")
            
            response = await agent._generate_structured_response(
                context=sample_workflow_context,
                user_input="Hello",
                response_schema=schema
            )
            
            assert response is None

    @pytest.mark.asyncio
    async def test_make_decision_success(self, agent, sample_workflow_context, mock_structured_response):
        """Test successful decision making"""
        decision_options = {
            "continue": "Continue conversation",
            "escalate": "Escalate to human",
            "end": "End conversation"
        }
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_structured_response.return_value = mock_structured_response
            
            decision = await agent._make_decision(
                context=sample_workflow_context,
                user_input="I'm interested",
                decision_options=decision_options
            )
            
            assert isinstance(decision, AgentDecision)
            assert decision.agent_type == AgentType.CONVERSATION
            assert decision.action == "continue"
            assert decision.reasoning == "User shows interest"
            assert decision.confidence == 0.8

    @pytest.mark.asyncio
    async def test_make_decision_failure(self, agent, sample_workflow_context):
        """Test decision making failure"""
        decision_options = {"continue": "Continue"}
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_structured_response.side_effect = Exception("Decision error")
            
            decision = await agent._make_decision(
                context=sample_workflow_context,
                user_input="Hello",
                decision_options=decision_options
            )
            
            assert decision is None

    @pytest.mark.asyncio
    async def test_make_decision_no_response(self, agent, sample_workflow_context):
        """Test decision making when no structured response"""
        decision_options = {"continue": "Continue"}
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_structured_response.return_value = None
            
            decision = await agent._make_decision(
                context=sample_workflow_context,
                user_input="Hello",
                decision_options=decision_options
            )
            
            assert decision is None

    @pytest.mark.asyncio
    async def test_conversation_context_creation(self, agent, sample_workflow_context):
        """Test conversation context creation from workflow context"""
        # Add more messages to conversation history
        from src.agents.models import AgentMessage
        sample_workflow_context.conversation_history.extend([
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(15)  # More than the 10 message limit
        ])
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.return_value = LLMResponse(
                content="Response", usage_tokens=10, response_time_ms=100.0, model_used="test"
            )
            
            await agent._generate_llm_response(
                context=sample_workflow_context,
                user_input="Hello"
            )
            
            # Verify only last 10 messages were used
            call_args = mock_service.generate_response.call_args
            conv_context = call_args[1]['context']
            assert len(conv_context.messages) == 10

    @pytest.mark.asyncio
    async def test_cleanup(self, agent):
        """Test agent cleanup"""
        agent.state = AgentState.THINKING
        
        await agent.cleanup()
        
        assert agent.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_concurrent_execution_lock(self, agent, sample_workflow_context):
        """Test that processing lock prevents concurrent execution"""
        execution_order = []
        
        async def tracked_process(context, user_input):
            execution_order.append(f"start_{user_input}")
            await asyncio.sleep(0.1)  # Simulate processing time
            execution_order.append(f"end_{user_input}")
            return AgentResponse(agent_type=AgentType.CONVERSATION, response_text=f"Response to {user_input}")
        
        agent.process = tracked_process
        
        # Start two concurrent executions
        task1 = asyncio.create_task(agent.execute_with_timeout(sample_workflow_context, "input1"))
        task2 = asyncio.create_task(agent.execute_with_timeout(sample_workflow_context, "input2"))
        
        responses = await asyncio.gather(task1, task2)
        
        # Verify both completed successfully
        assert len(responses) == 2
        assert all(r.response_text.startswith("Response to") for r in responses)
        
        # Verify they executed sequentially (lock worked)
        assert len(execution_order) == 4
        # First execution should complete before second starts
        assert execution_order.index("end_input1") < execution_order.index("start_input2") or \
               execution_order.index("end_input2") < execution_order.index("start_input1")

    def test_system_prompt_called_when_none_provided(self, agent, sample_workflow_context):
        """Test that get_system_prompt is called when no custom prompt provided"""
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.return_value = LLMResponse(
                content="Response", usage_tokens=10, response_time_ms=100.0, model_used="test"
            )
            
            asyncio.run(agent._generate_llm_response(
                context=sample_workflow_context,
                user_input="Hello"
            ))
            
            assert agent.system_prompt_called is True


class TestBaseAgentEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def agent(self):
        return TestableAgent()

    @pytest.mark.asyncio
    async def test_empty_conversation_history(self, agent):
        """Test handling of empty conversation history"""
        context = WorkflowContext(
            call_id="test",
            workflow_state=WorkflowState.GREETING,
            conversation_history=[],  # Empty history
            lead_data={}
        )
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.return_value = LLMResponse(
                content="Response", usage_tokens=10, response_time_ms=100.0, model_used="test"
            )
            
            response = await agent._generate_llm_response(context, "Hello")
            
            assert response == "Response"
            # Verify empty messages list was handled
            call_args = mock_service.generate_response.call_args
            conv_context = call_args[1]['context']
            assert len(conv_context.messages) == 0

    @pytest.mark.asyncio
    async def test_very_short_timeout(self, agent, sample_workflow_context):
        """Test behavior with extremely short timeout"""
        response = await agent.execute_with_timeout(
            context=sample_workflow_context,
            user_input="Hello",
            timeout_seconds=0.001  # 1ms timeout
        )
        
        # Should handle gracefully even with very short timeout
        assert response.agent_type == AgentType.CONVERSATION
        assert response.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_large_conversation_history(self, agent):
        """Test handling of very large conversation history"""
        from src.agents.models import AgentMessage
        
        # Create context with 100 messages
        large_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(100)
        ]
        
        context = WorkflowContext(
            call_id="test",
            workflow_state=WorkflowState.GREETING,
            conversation_history=large_history,
            lead_data={}
        )
        
        with patch('src.agents.base_agent.llm_service') as mock_service:
            mock_service.generate_response.return_value = LLMResponse(
                content="Response", usage_tokens=10, response_time_ms=100.0, model_used="test"
            )
            
            response = await agent._generate_llm_response(context, "Hello")
            
            assert response == "Response"
            # Should only use last 10 messages
            call_args = mock_service.generate_response.call_args
            conv_context = call_args[1]['context']
            assert len(conv_context.messages) == 10


if __name__ == "__main__":
    pytest.main([__file__])
