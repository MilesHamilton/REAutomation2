"""
Unit tests for AgentOrchestrator
"""
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Handle optional imports gracefully for testing
try:
    from src.agents.orchestrator import AgentOrchestrator
    from src.agents.models import (
        AgentType, WorkflowState, WorkflowContext, AgentMessage,
        AgentResponse, AgentDecision
    )
    from src.agents.conversation_agent import ConversationAgent
    from src.agents.qualification_agent import QualificationAgent
    from src.agents.objection_handler import ObjectionHandlerAgent
    from src.agents.scheduler_agent import SchedulerAgent
    from src.agents.analytics_agent import AnalyticsAgent
except ImportError:
    # Skip tests if dependencies are not available
    pytest.skip("Agent dependencies not available", allow_module_level=True)


class TestAgentOrchestrator:
    """Test suite for AgentOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance for each test"""
        return AgentOrchestrator()

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        agents = {}
        for agent_type in AgentType:
            mock_agent = AsyncMock()
            mock_agent.execute_with_timeout.return_value = AgentResponse(
                agent_type=agent_type,
                response_text=f"Response from {agent_type.value}",
                processing_time_ms=100.0
            )
            mock_agent.cleanup.return_value = None
            agents[agent_type] = mock_agent
        return agents

    @pytest.fixture
    def sample_lead_data(self):
        """Sample lead data for testing"""
        return {
            "name": "John Doe",
            "phone": "+1234567890",
            "email": "john.doe@example.com",
            "company": "Test Corp",
            "source": "website"
        }

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.agents is not None
        assert len(orchestrator.agents) == 5
        assert AgentType.CONVERSATION in orchestrator.agents
        assert AgentType.QUALIFICATION in orchestrator.agents
        assert AgentType.OBJECTION_HANDLER in orchestrator.agents
        assert AgentType.SCHEDULER in orchestrator.agents
        assert AgentType.ANALYTICS in orchestrator.agents
        
        assert orchestrator.workflow_graph is None
        assert orchestrator.active_contexts == {}
        assert orchestrator.is_initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, orchestrator):
        """Test successful orchestrator initialization"""
        result = await orchestrator.initialize()
        
        assert result is True
        assert orchestrator.is_initialized is True
        assert orchestrator.workflow_graph is not None

    @pytest.mark.asyncio
    async def test_initialize_failure(self, orchestrator):
        """Test orchestrator initialization failure"""
        # Mock _build_workflow_graph to raise exception
        with patch.object(orchestrator, '_build_workflow_graph', side_effect=Exception("Build failed")):
            result = await orchestrator.initialize()
            
            assert result is False
            assert orchestrator.is_initialized is False

    @pytest.mark.asyncio
    async def test_process_input_not_initialized(self, orchestrator):
        """Test process_input when orchestrator not initialized"""
        response = await orchestrator.process_input("test-call", "Hello")
        
        assert response is None

    @pytest.mark.asyncio
    async def test_process_input_new_context(self, orchestrator, sample_lead_data):
        """Test process_input with new workflow context"""
        await orchestrator.initialize()
        
        # Mock the workflow graph
        mock_result = WorkflowContext(
            call_id="test-call",
            workflow_state=WorkflowState.GREETING,
            lead_data=sample_lead_data
        )
        mock_result.metadata = {
            "agent_response": AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="Hello! How can I help you today?",
                processing_time_ms=150.0
            )
        }
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke.return_value = mock_result
        
        response = await orchestrator.process_input(
            call_id="test-call",
            user_input="Hello",
            lead_data=sample_lead_data
        )
        
        assert response is not None
        assert response.agent_type == AgentType.CONVERSATION
        assert response.response_text == "Hello! How can I help you today?"
        assert "test-call" in orchestrator.active_contexts

    @pytest.mark.asyncio
    async def test_process_input_existing_context(self, orchestrator):
        """Test process_input with existing workflow context"""
        await orchestrator.initialize()
        
        # Create existing context
        existing_context = WorkflowContext(
            call_id="test-call",
            workflow_state=WorkflowState.QUALIFYING,
            conversation_history=[
                AgentMessage(
                    agent_type=AgentType.CONVERSATION,
                    content="Previous message",
                    timestamp=time.time() - 10
                )
            ]
        )
        orchestrator.active_contexts["test-call"] = existing_context
        
        # Mock workflow result
        mock_result = existing_context
        mock_result.metadata = {
            "agent_response": AgentResponse(
                agent_type=AgentType.QUALIFICATION,
                response_text="Can you tell me about your budget?",
                processing_time_ms=200.0
            )
        }
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke.return_value = mock_result
        
        response = await orchestrator.process_input("test-call", "I'm interested")
        
        assert response is not None
        assert response.agent_type == AgentType.QUALIFICATION
        # Verify conversation history was updated
        assert len(existing_context.conversation_history) >= 2

    @pytest.mark.asyncio
    async def test_process_input_tier_escalation(self, orchestrator):
        """Test process_input with tier escalation"""
        await orchestrator.initialize()
        
        # Set up tier escalation callback
        escalation_calls = []
        def mock_escalation_callback(call_id, score):
            escalation_calls.append((call_id, score))
        
        orchestrator.on_tier_escalation(mock_escalation_callback)
        
        # Mock workflow result with escalation
        mock_result = WorkflowContext(
            call_id="test-call",
            qualification_score=0.8
        )
        mock_result.metadata = {
            "agent_response": AgentResponse(
                agent_type=AgentType.QUALIFICATION,
                response_text="Great! Let me connect you with our premium service.",
                should_escalate_tier=True,
                processing_time_ms=180.0
            )
        }
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke.return_value = mock_result
        
        response = await orchestrator.process_input("test-call", "I'm very interested")
        
        assert response is not None
        assert response.should_escalate_tier is True
        assert len(escalation_calls) == 1
        assert escalation_calls[0] == ("test-call", 0.8)

    @pytest.mark.asyncio
    async def test_process_input_exception_handling(self, orchestrator):
        """Test process_input exception handling"""
        await orchestrator.initialize()
        
        # Mock workflow graph to raise exception
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke.side_effect = Exception("Workflow error")
        
        response = await orchestrator.process_input("test-call", "Hello")
        
        assert response is not None
        assert response.agent_type == AgentType.CONVERSATION
        assert "technical issue" in response.response_text

    def test_get_or_create_context_new(self, orchestrator, sample_lead_data):
        """Test creating new workflow context"""
        context = orchestrator._get_or_create_context("new-call", sample_lead_data)
        
        assert context.call_id == "new-call"
        assert context.lead_data == sample_lead_data
        assert context.workflow_state == WorkflowState.INITIALIZING
        assert "new-call" in orchestrator.active_contexts

    def test_get_or_create_context_existing(self, orchestrator):
        """Test getting existing workflow context"""
        # Create existing context
        existing_context = WorkflowContext(call_id="existing-call")
        orchestrator.active_contexts["existing-call"] = existing_context
        
        context = orchestrator._get_or_create_context("existing-call")
        
        assert context is existing_context
        assert context.call_id == "existing-call"

    @pytest.mark.asyncio
    async def test_conversation_node(self, orchestrator, sample_workflow_context):
        """Test conversation node processing"""
        # Mock conversation agent
        mock_agent = AsyncMock()
        mock_response = AgentResponse(
            agent_type=AgentType.CONVERSATION,
            response_text="Hello! How are you today?",
            state_updates={"workflow_state": WorkflowState.GREETING},
            processing_time_ms=120.0
        )
        mock_agent.execute_with_timeout.return_value = mock_response
        orchestrator.agents[AgentType.CONVERSATION] = mock_agent
        
        # Set current input in metadata
        sample_workflow_context.metadata["current_input"] = "Hello"
        
        result = await orchestrator._conversation_node(sample_workflow_context)
        
        assert result.current_agent == AgentType.CONVERSATION
        assert result.metadata["agent_response"] == mock_response
        assert result.workflow_state == WorkflowState.GREETING
        mock_agent.execute_with_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_qualification_node(self, orchestrator, sample_workflow_context):
        """Test qualification node processing"""
        mock_agent = AsyncMock()
        mock_response = AgentResponse(
            agent_type=AgentType.QUALIFICATION,
            response_text="Can you tell me about your budget range?",
            state_updates={"qualification_score": 0.6},
            processing_time_ms=200.0
        )
        mock_agent.execute_with_timeout.return_value = mock_response
        orchestrator.agents[AgentType.QUALIFICATION] = mock_agent
        
        sample_workflow_context.metadata["current_input"] = "I'm looking to buy"
        
        result = await orchestrator._qualification_node(sample_workflow_context)
        
        assert result.current_agent == AgentType.QUALIFICATION
        assert result.metadata["agent_response"] == mock_response
        assert result.qualification_score == 0.6

    @pytest.mark.asyncio
    async def test_objection_handler_node(self, orchestrator, sample_workflow_context):
        """Test objection handler node processing"""
        mock_agent = AsyncMock()
        mock_response = AgentResponse(
            agent_type=AgentType.OBJECTION_HANDLER,
            response_text="I understand your concern. Let me address that.",
            state_updates={"objection_count": 1},
            processing_time_ms=180.0
        )
        mock_agent.execute_with_timeout.return_value = mock_response
        orchestrator.agents[AgentType.OBJECTION_HANDLER] = mock_agent
        
        sample_workflow_context.metadata["current_input"] = "This is too expensive"
        
        result = await orchestrator._objection_handler_node(sample_workflow_context)
        
        assert result.current_agent == AgentType.OBJECTION_HANDLER
        assert result.metadata["agent_response"] == mock_response
        assert result.objection_count == 1

    @pytest.mark.asyncio
    async def test_scheduler_node(self, orchestrator, sample_workflow_context):
        """Test scheduler node processing"""
        mock_agent = AsyncMock()
        mock_response = AgentResponse(
            agent_type=AgentType.SCHEDULER,
            response_text="Great! I have availability tomorrow at 2 PM. Does that work?",
            state_updates={"scheduling_attempts": 1},
            processing_time_ms=160.0
        )
        mock_agent.execute_with_timeout.return_value = mock_response
        orchestrator.agents[AgentType.SCHEDULER] = mock_agent
        
        sample_workflow_context.metadata["current_input"] = "I'd like to schedule a meeting"
        
        result = await orchestrator._scheduler_node(sample_workflow_context)
        
        assert result.current_agent == AgentType.SCHEDULER
        assert result.metadata["agent_response"] == mock_response
        assert result.scheduling_attempts == 1

    @pytest.mark.asyncio
    async def test_analytics_node(self, orchestrator, sample_workflow_context):
        """Test analytics node processing"""
        mock_agent = AsyncMock()
        mock_response = AgentResponse(
            agent_type=AgentType.ANALYTICS,
            response_text="",  # Analytics typically doesn't generate user-facing text
            state_updates={"analytics_processed": True},
            processing_time_ms=50.0
        )
        mock_agent.execute_with_timeout.return_value = mock_response
        orchestrator.agents[AgentType.ANALYTICS] = mock_agent
        
        sample_workflow_context.metadata["current_input"] = "Any input"
        
        result = await orchestrator._analytics_node(sample_workflow_context)
        
        # Analytics updates metadata but doesn't change workflow flow
        assert result.metadata["analytics_processed"] is True

    def test_route_from_conversation(self, orchestrator, sample_workflow_context):
        """Test routing from conversation agent"""
        # Test routing to qualification
        sample_workflow_context.workflow_state = WorkflowState.QUALIFYING
        route = orchestrator._route_from_conversation(sample_workflow_context)
        assert route == "qualification"
        
        # Test routing to objection handler
        sample_workflow_context.workflow_state = WorkflowState.HANDLING_OBJECTION
        route = orchestrator._route_from_conversation(sample_workflow_context)
        assert route == "objection_handler"
        
        # Test routing to scheduler
        sample_workflow_context.workflow_state = WorkflowState.SCHEDULING
        route = orchestrator._route_from_conversation(sample_workflow_context)
        assert route == "scheduler"
        
        # Test routing to end
        sample_workflow_context.workflow_state = WorkflowState.COMPLETED
        route = orchestrator._route_from_conversation(sample_workflow_context)
        assert route == "end"

    def test_route_from_qualification(self, orchestrator, sample_workflow_context):
        """Test routing from qualification agent"""
        # Test routing to objection handler
        sample_workflow_context.workflow_state = WorkflowState.HANDLING_OBJECTION
        route = orchestrator._route_from_qualification(sample_workflow_context)
        assert route == "objection_handler"
        
        # Test routing to scheduler
        sample_workflow_context.workflow_state = WorkflowState.SCHEDULING
        route = orchestrator._route_from_qualification(sample_workflow_context)
        assert route == "scheduler"
        
        # Test routing back to conversation
        sample_workflow_context.workflow_state = WorkflowState.GREETING
        route = orchestrator._route_from_qualification(sample_workflow_context)
        assert route == "conversation"
        
        # Test routing to end
        sample_workflow_context.workflow_state = WorkflowState.COMPLETED
        route = orchestrator._route_from_qualification(sample_workflow_context)
        assert route == "end"

    def test_route_from_objection_handler(self, orchestrator, sample_workflow_context):
        """Test routing from objection handler agent"""
        # Test routing to qualification
        sample_workflow_context.workflow_state = WorkflowState.QUALIFYING
        route = orchestrator._route_from_objection_handler(sample_workflow_context)
        assert route == "qualification"
        
        # Test routing to scheduler
        sample_workflow_context.workflow_state = WorkflowState.SCHEDULING
        route = orchestrator._route_from_objection_handler(sample_workflow_context)
        assert route == "scheduler"
        
        # Test routing to conversation
        sample_workflow_context.workflow_state = WorkflowState.GREETING
        route = orchestrator._route_from_objection_handler(sample_workflow_context)
        assert route == "conversation"

    def test_route_from_scheduler(self, orchestrator, sample_workflow_context):
        """Test routing from scheduler agent"""
        # Test routing to qualification
        sample_workflow_context.workflow_state = WorkflowState.QUALIFYING
        route = orchestrator._route_from_scheduler(sample_workflow_context)
        assert route == "qualification"
        
        # Test routing to conversation
        sample_workflow_context.workflow_state = WorkflowState.GREETING
        route = orchestrator._route_from_scheduler(sample_workflow_context)
        assert route == "conversation"
        
        # Test routing to end
        sample_workflow_context.workflow_state = WorkflowState.COMPLETED
        route = orchestrator._route_from_scheduler(sample_workflow_context)
        assert route == "end"

    def test_get_context(self, orchestrator):
        """Test getting workflow context"""
        # Test non-existent context
        context = orchestrator.get_context("non-existent")
        assert context is None
        
        # Test existing context
        existing_context = WorkflowContext(call_id="existing")
        orchestrator.active_contexts["existing"] = existing_context
        
        context = orchestrator.get_context("existing")
        assert context is existing_context

    @pytest.mark.asyncio
    async def test_end_workflow_success(self, orchestrator):
        """Test successful workflow ending"""
        # Create context
        context = WorkflowContext(call_id="test-call")
        orchestrator.active_contexts["test-call"] = context
        
        # Mock analytics agent
        mock_analytics = AsyncMock()
        orchestrator.agents[AgentType.ANALYTICS] = mock_analytics
        
        # Set up completion callback
        completion_calls = []
        def mock_completion_callback(ctx):
            completion_calls.append(ctx)
        
        orchestrator.on_workflow_complete(mock_completion_callback)
        
        result = orchestrator.end_workflow("test-call")
        
        assert result is True
        assert context.workflow_state == WorkflowState.COMPLETED
        assert "test-call" not in orchestrator.active_contexts
        assert len(completion_calls) == 1

    def test_end_workflow_not_found(self, orchestrator):
        """Test ending non-existent workflow"""
        result = orchestrator.end_workflow("non-existent")
        assert result is False

    def test_get_active_workflows(self, orchestrator):
        """Test getting active workflows"""
        # Empty initially
        workflows = orchestrator.get_active_workflows()
        assert workflows == {}
        
        # Add some contexts
        context1 = WorkflowContext(call_id="call1")
        context2 = WorkflowContext(call_id="call2")
        orchestrator.active_contexts["call1"] = context1
        orchestrator.active_contexts["call2"] = context2
        
        workflows = orchestrator.get_active_workflows()
        assert len(workflows) == 2
        assert "call1" in workflows
        assert "call2" in workflows
        # Should be a copy, not the original
        assert workflows is not orchestrator.active_contexts

    def test_callback_registration(self, orchestrator):
        """Test callback registration"""
        # Test agent transition callback
        transition_callback = Mock()
        orchestrator.on_agent_transition(transition_callback)
        assert orchestrator._on_agent_transition is transition_callback
        
        # Test workflow complete callback
        complete_callback = Mock()
        orchestrator.on_workflow_complete(complete_callback)
        assert orchestrator._on_workflow_complete is complete_callback
        
        # Test tier escalation callback
        escalation_callback = Mock()
        orchestrator.on_tier_escalation(escalation_callback)
        assert orchestrator._on_tier_escalation is escalation_callback

    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self, orchestrator):
        """Test health check when not initialized"""
        health = await orchestrator.health_check()
        
        assert health["status"] == "unhealthy"
        assert "not initialized" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, orchestrator):
        """Test health check when healthy"""
        await orchestrator.initialize()
        
        # Add some active contexts
        orchestrator.active_contexts["call1"] = WorkflowContext(call_id="call1")
        orchestrator.active_contexts["call2"] = WorkflowContext(call_id="call2")
        
        health = await orchestrator.health_check()
        
        assert health["status"] == "healthy"
        assert health["active_workflows"] == 2
        assert len(health["agents"]) == 5
        assert health["workflow_graph_ready"] is True
        assert health["initialized"] is True

    @pytest.mark.asyncio
    async def test_health_check_exception(self, orchestrator):
        """Test health check with exception"""
        await orchestrator.initialize()
        
        # Mock len() to raise exception when accessing active_contexts
        with patch('builtins.len', side_effect=Exception("Health error")):
            health = await orchestrator.health_check()
            
            assert health["status"] == "unhealthy"
            assert "Health error" in health["error"]

    @pytest.mark.asyncio
    async def test_cleanup(self, orchestrator, mock_agents):
        """Test orchestrator cleanup"""
        # Set up orchestrator with mock agents
        orchestrator.agents = mock_agents
        await orchestrator.initialize()
        
        # Add some active contexts
        orchestrator.active_contexts["call1"] = WorkflowContext(call_id="call1")
        orchestrator.active_contexts["call2"] = WorkflowContext(call_id="call2")
        
        await orchestrator.cleanup()
        
        # Verify cleanup
        assert orchestrator.is_initialized is False
        assert len(orchestrator.active_contexts) == 0
        
        # Verify all agents were cleaned up
        for agent in mock_agents.values():
            agent.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_exception(self, orchestrator):
        """Test cleanup with exception"""
        await orchestrator.initialize()
        
        # Mock agent cleanup to raise exception
        mock_agent = AsyncMock()
        mock_agent.cleanup.side_effect = Exception("Cleanup error")
        orchestrator.agents[AgentType.CONVERSATION] = mock_agent
        
        # Should not raise exception but should still set initialized to False
        await orchestrator.cleanup()
        # The cleanup method catches exceptions but still sets is_initialized to False
        assert orchestrator.is_initialized is False


class TestAgentOrchestratorIntegration:
    """Integration tests for AgentOrchestrator with mock client interactions"""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for integration tests"""
        return AgentOrchestrator()

    @pytest.fixture
    def mock_client_session(self):
        """Mock client session for testing full conversation flow"""
        return {
            "call_id": "integration-test-call",
            "lead_data": {
                "name": "Jane Smith",
                "phone": "+1555123456",
                "email": "jane.smith@example.com",
                "company": "Smith Enterprises",
                "interest": "commercial_real_estate"
            },
            "conversation_log": []
        }

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, orchestrator, mock_client_session):
        """Test complete conversation flow from greeting to scheduling"""
        await orchestrator.initialize()
        
        # Mock all agents to simulate realistic responses
        conversation_responses = [
            AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="Hello Jane! I'm calling from ABC Realty about commercial real estate opportunities. How are you today?",
                state_updates={"workflow_state": WorkflowState.GREETING},
                processing_time_ms=120.0
            )
        ]
        
        qualification_responses = [
            AgentResponse(
                agent_type=AgentType.QUALIFICATION,
                response_text="That's great to hear! Can you tell me what type of commercial property you're looking for?",
                state_updates={
                    "workflow_state": WorkflowState.QUALIFYING,
                    "qualification_score": 0.4
                },
                processing_time_ms=180.0
            ),
            AgentResponse(
                agent_type=AgentType.QUALIFICATION,
                response_text="Excellent! What's your budget range for this investment?",
                state_updates={
                    "workflow_state": WorkflowState.QUALIFYING,
                    "qualification_score": 0.7
                },
                processing_time_ms=160.0
            )
        ]
        
        scheduler_responses = [
            AgentResponse(
                agent_type=AgentType.SCHEDULER,
                response_text="Perfect! I'd love to show you some properties that match your criteria. I have availability tomorrow at 2 PM or Thursday at 10 AM. Which works better for you?",
                state_updates={
                    "workflow_state": WorkflowState.SCHEDULING,
                    "scheduling_attempts": 1
                },
                processing_time_ms=200.0
            )
        ]
        
        # Mock workflow graph to simulate conversation flow
        conversation_step = 0
        qualification_step = 0
        scheduler_step = 0
        
        async def mock_workflow_invoke(context, config):
            nonlocal conversation_step, qualification_step, scheduler_step
            
            current_input = context.metadata.get("current_input", "")
            
            # Simulate conversation routing based on input
            if "hello" in current_input.lower() or context.workflow_state == WorkflowState.INITIALIZING:
                response = conversation_responses[min(conversation_step, len(conversation_responses) - 1)]
                conversation_step += 1
            elif "office building" in current_input.lower() or "warehouse" in current_input.lower():
                response = qualification_responses[min(qualification_step, len(qualification_responses) - 1)]
                qualification_step += 1
            elif "million" in current_input.lower() or "$" in current_input:
                response = qualification_responses[min(qualification_step, len(qualification_responses) - 1)]
                qualification_step += 1
            elif "schedule" in current_input.lower() or "meeting" in current_input.lower():
                response = scheduler_responses[min(scheduler_step, len(scheduler_responses) - 1)]
                scheduler_step += 1
            else:
                response = AgentResponse(
                    agent_type=AgentType.CONVERSATION,
                    response_text="I understand. Can you tell me more about that?",
                    processing_time_ms=100.0
                )
            
            # Update context with response
            if response.state_updates:
                for key, value in response.state_updates.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                    else:
                        context.metadata[key] = value
            
            context.metadata["agent_response"] = response
            return context
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke = mock_workflow_invoke
        
        # Simulate conversation flow
        conversation_inputs = [
            "Hello, this is Jane",
            "I'm looking for an office building",
            "Around 2-3 million dollars",
            "Yes, I'd like to schedule a meeting"
        ]
        
        responses = []
        for user_input in conversation_inputs:
            response = await orchestrator.process_input(
                call_id=mock_client_session["call_id"],
                user_input=user_input,
                lead_data=mock_client_session["lead_data"]
            )
            
            responses.append(response)
            mock_client_session["conversation_log"].append({
                "user": user_input,
                "agent": response.response_text if response else "No response",
                "agent_type": response.agent_type.value if response else "unknown"
            })
        
        # Verify conversation flow
        assert len(responses) == 4
        assert all(r is not None for r in responses)
        
        # Verify progression through different agents
        agent_types_used = [r.agent_type for r in responses]
        assert AgentType.CONVERSATION in agent_types_used
        assert AgentType.QUALIFICATION in agent_types_used
        assert AgentType.SCHEDULER in agent_types_used
        
        # Verify context was maintained
        final_context = orchestrator.get_context(mock_client_session["call_id"])
        assert final_context is not None
        assert final_context.qualification_score > 0.5
        assert len(final_context.conversation_history) > 0

    @pytest.mark.asyncio
    async def test_objection_handling_flow(self, orchestrator):
        """Test conversation flow with objection handling"""
        await orchestrator.initialize()
        
        # Mock objection scenario
        objection_responses = [
            AgentResponse(
                agent_type=AgentType.OBJECTION_HANDLER,
                response_text="I completely understand your concern about the price. Let me show you the value proposition and potential ROI of this investment.",
                state_updates={
                    "workflow_state": WorkflowState.HANDLING_OBJECTION,
                    "objection_count": 1
                },
                processing_time_ms=220.0
            ),
            AgentResponse(
                agent_type=AgentType.QUALIFICATION,
                response_text="Based on the location and market trends, properties in this area have appreciated 15% annually. Would you like to see the detailed analysis?",
                state_updates={
                    "workflow_state": WorkflowState.QUALIFYING,
                    "qualification_score": 0.6
                },
                processing_time_ms=180.0
            )
        ]
        
        objection_step = 0
        
        async def mock_objection_workflow(context, config):
            nonlocal objection_step
            
            current_input = context.metadata.get("current_input", "")
            
            if "too expensive" in current_input.lower() or "can't afford" in current_input.lower():
                response = objection_responses[0]
                objection_step = 1
            elif objection_step == 1:
                response = objection_responses[1]
                objection_step = 2
            else:
                response = AgentResponse(
                    agent_type=AgentType.CONVERSATION,
                    response_text="I understand. What other questions do you have?",
                    processing_time_ms=100.0
                )
            
            # Update context
            if response.state_updates:
                for key, value in response.state_updates.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                    else:
                        context.metadata[key] = value
            
            context.metadata["agent_response"] = response
            return context
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke = mock_objection_workflow
        
        # Simulate objection handling conversation
        objection_inputs = [
            "This is too expensive for our budget",
            "That sounds more reasonable"
        ]
        
        responses = []
        for user_input in objection_inputs:
            response = await orchestrator.process_input(
                call_id="objection-test-call",
                user_input=user_input
            )
            responses.append(response)
        
        # Verify objection handling
        assert len(responses) == 2
        assert responses[0].agent_type == AgentType.OBJECTION_HANDLER
        assert "understand your concern" in responses[0].response_text
        
        # Verify context tracking
        context = orchestrator.get_context("objection-test-call")
        assert context.objection_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_workflow_processing(self, orchestrator):
        """Test handling multiple concurrent workflows"""
        await orchestrator.initialize()
        
        # Mock workflow graph for concurrent processing
        async def mock_concurrent_workflow(context, config):
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            response = AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text=f"Response for {context.call_id}",
                processing_time_ms=100.0
            )
            
            context.metadata["agent_response"] = response
            return context
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke = mock_concurrent_workflow
        
        # Start multiple concurrent workflows
        call_ids = ["call1", "call2", "call3"]
        tasks = []
        
        for call_id in call_ids:
            task = orchestrator.process_input(call_id, f"Hello from {call_id}")
            tasks.append(task)
        
        # Wait for all to complete
        responses = await asyncio.gather(*tasks)
        
        # Verify all processed successfully
        assert len(responses) == 3
        assert all(r is not None for r in responses)
        
        # Verify separate contexts maintained
        for i, call_id in enumerate(call_ids):
            context = orchestrator.get_context(call_id)
            assert context is not None
            assert context.call_id == call_id
            assert f"Hello from {call_id}" in [msg.content for msg in context.conversation_history]

    @pytest.mark.asyncio
    async def test_workflow_state_transitions(self, orchestrator):
        """Test proper workflow state transitions"""
        await orchestrator.initialize()
        
        state_transitions = []
        
        async def mock_state_workflow(context, config):
            current_input = context.metadata.get("current_input", "")
            
            # Track state transitions
            if current_input == "start":
                context.workflow_state = WorkflowState.GREETING
                state_transitions.append(WorkflowState.GREETING)
            elif current_input == "qualify":
                context.workflow_state = WorkflowState.QUALIFYING
                state_transitions.append(WorkflowState.QUALIFYING)
            elif current_input == "object":
                context.workflow_state = WorkflowState.HANDLING_OBJECTION
                state_transitions.append(WorkflowState.HANDLING_OBJECTION)
            elif current_input == "schedule":
                context.workflow_state = WorkflowState.SCHEDULING
                state_transitions.append(WorkflowState.SCHEDULING)
            elif current_input == "end":
                context.workflow_state = WorkflowState.COMPLETED
                state_transitions.append(WorkflowState.COMPLETED)
            
            response = AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text=f"State: {context.workflow_state.value}",
                processing_time_ms=50.0
            )
            
            context.metadata["agent_response"] = response
            return context
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke = mock_state_workflow
        
        # Test state progression
        inputs = ["start", "qualify", "object", "schedule", "end"]
        
        for user_input in inputs:
            await orchestrator.process_input("state-test", user_input)
        
        # Verify state transitions
        expected_states = [
            WorkflowState.GREETING,
            WorkflowState.QUALIFYING,
            WorkflowState.HANDLING_OBJECTION,
            WorkflowState.SCHEDULING,
            WorkflowState.COMPLETED
        ]
        
        assert state_transitions == expected_states

    @pytest.mark.asyncio
    async def test_callback_invocation(self, orchestrator):
        """Test that callbacks are properly invoked"""
        await orchestrator.initialize()
        
        # Set up callback tracking
        transition_calls = []
        completion_calls = []
        escalation_calls = []
        
        def track_transition(call_id, from_agent, to_agent):
            transition_calls.append((call_id, from_agent, to_agent))
        
        def track_completion(context):
            completion_calls.append(context.call_id)
        
        def track_escalation(call_id, score):
            escalation_calls.append((call_id, score))
        
        # Register callbacks
        orchestrator.on_agent_transition(track_transition)
        orchestrator.on_workflow_complete(track_completion)
        orchestrator.on_tier_escalation(track_escalation)
        
        # Test escalation callback
        mock_result = WorkflowContext(
            call_id="callback-test",
            qualification_score=0.9
        )
        mock_result.metadata = {
            "agent_response": AgentResponse(
                agent_type=AgentType.QUALIFICATION,
                response_text="Escalating to premium service",
                should_escalate_tier=True,
                processing_time_ms=150.0
            )
        }
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke.return_value = mock_result
        
        await orchestrator.process_input("callback-test", "I'm very interested")
        
        # Test completion callback
        orchestrator.end_workflow("callback-test")
        
        # Verify callbacks were invoked
        assert len(escalation_calls) == 1
        assert escalation_calls[0] == ("callback-test", 0.9)
        assert len(completion_calls) == 1
        assert completion_calls[0] == "callback-test"

    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback(self, orchestrator):
        """Test error recovery and fallback mechanisms"""
        await orchestrator.initialize()
        
        # Test agent failure recovery
        failure_count = 0
        
        async def failing_workflow(context, config):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:
                raise Exception(f"Simulated failure {failure_count}")
            
            # Success after failures
            response = AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="Recovered successfully",
                processing_time_ms=100.0
            )
            
            context.metadata["agent_response"] = response
            return context
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke = failing_workflow
        
        # First two calls should get error responses
        response1 = await orchestrator.process_input("error-test", "Hello")
        response2 = await orchestrator.process_input("error-test", "Hello again")
        
        # Both should return error responses but not crash
        assert response1 is not None
        assert response2 is not None
        assert "technical issue" in response1.response_text
        assert "technical issue" in response2.response_text
        
        # Third call should succeed (if we had retry logic)
        # For now, just verify the orchestrator remains stable
        assert orchestrator.is_initialized is True

    @pytest.mark.asyncio
    async def test_memory_and_context_persistence(self, orchestrator):
        """Test that conversation context persists across interactions"""
        await orchestrator.initialize()
        
        conversation_history = []
        
        async def memory_workflow(context, config):
            current_input = context.metadata.get("current_input", "")
            
            # Store conversation in our tracking
            conversation_history.append(current_input)
            
            response = AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text=f"I remember you said: {', '.join(conversation_history[-3:])}",
                processing_time_ms=100.0
            )
            
            context.metadata["agent_response"] = response
            return context
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke = memory_workflow
        
        # Have a multi-turn conversation
        inputs = [
            "My name is John",
            "I'm looking for office space",
            "In downtown area",
            "Budget is 2 million"
        ]
        
        for user_input in inputs:
            response = await orchestrator.process_input("memory-test", user_input)
            assert response is not None
        
        # Verify context persistence
        context = orchestrator.get_context("memory-test")
        assert context is not None
        assert len(context.conversation_history) == len(inputs) * 2  # User + agent messages
        
        # Verify conversation history contains all inputs
        user_messages = [msg for msg in context.conversation_history if msg.content in inputs]
        assert len(user_messages) == len(inputs)


class TestAgentOrchestratorEdgeCases:
    """Test edge cases and error conditions for AgentOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        return AgentOrchestrator()

    @pytest.mark.asyncio
    async def test_empty_user_input(self, orchestrator):
        """Test handling of empty user input"""
        await orchestrator.initialize()
        
        mock_result = WorkflowContext(call_id="empty-test")
        mock_result.metadata = {
            "agent_response": AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="I didn't catch that. Could you repeat?",
                processing_time_ms=50.0
            )
        }
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke.return_value = mock_result
        
        response = await orchestrator.process_input("empty-test", "")
        
        assert response is not None
        assert response.agent_type == AgentType.CONVERSATION

    @pytest.mark.asyncio
    async def test_very_long_user_input(self, orchestrator):
        """Test handling of very long user input"""
        await orchestrator.initialize()
        
        # Create very long input (simulate edge case)
        long_input = "This is a very long message. " * 100
        
        mock_result = WorkflowContext(call_id="long-test")
        mock_result.metadata = {
            "agent_response": AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="I understand you have a lot to share. Let me help you with that.",
                processing_time_ms=200.0
            )
        }
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke.return_value = mock_result
        
        response = await orchestrator.process_input("long-test", long_input)
        
        assert response is not None
        # Verify long input was stored in context
        context = orchestrator.get_context("long-test")
        assert context is not None
        # Check that the user input was added to conversation history
        user_messages = [msg for msg in context.conversation_history if msg.content == long_input]
        assert len(user_messages) >= 1

    @pytest.mark.asyncio
    async def test_rapid_successive_inputs(self, orchestrator):
        """Test handling of rapid successive inputs from same caller"""
        await orchestrator.initialize()
        
        call_count = 0
        
        async def rapid_workflow(context, config):
            nonlocal call_count
            call_count += 1
            
            # Simulate processing delay
            await asyncio.sleep(0.05)
            
            response = AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text=f"Processing call {call_count}",
                processing_time_ms=50.0
            )
            
            context.metadata["agent_response"] = response
            return context
        
        orchestrator.workflow_graph = AsyncMock()
        orchestrator.workflow_graph.ainvoke = rapid_workflow
        
        # Send rapid inputs
        tasks = []
        for i in range(5):
            task = orchestrator.process_input("rapid-test", f"Message {i}")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(responses) == 5
        assert all(r is not None for r in responses)
        assert call_count == 5

    def test_invalid_call_id_formats(self, orchestrator):
        """Test handling of various call ID formats"""
        # Test with different call ID formats
        call_ids = [
            "normal-call-123",
            "call_with_underscores",
            "call-with-special-chars!@#",
            "very-long-call-id-" + "x" * 100,
            "123456789",  # Numeric
            "",  # Empty (edge case)
        ]
        
        for call_id in call_ids:
            context = orchestrator._get_or_create_context(call_id)
            assert context is not None
            assert context.call_id == call_id

    @pytest.mark.asyncio
    async def test_workflow_graph_none_handling(self, orchestrator):
        """Test behavior when workflow_graph is None"""
        # Initialize but then set workflow_graph to None
        await orchestrator.initialize()
        orchestrator.workflow_graph = None
        orchestrator.is_initialized = True  # Keep initialized flag
        
        response = await orchestrator.process_input("none-test", "Hello")
        
        # Should handle gracefully
        assert response is not None
        assert "technical issue" in response.response_text

    @pytest.mark.asyncio
    async def test_large_number_of_active_contexts(self, orchestrator):
        """Test performance with large number of active contexts"""
        await orchestrator.initialize()
        
        # Create many active contexts
        for i in range(100):
            context = WorkflowContext(call_id=f"load-test-{i}")
            orchestrator.active_contexts[f"load-test-{i}"] = context
        
        # Verify operations still work
        health = await orchestrator.health_check()
        assert health["status"] == "healthy"
        assert health["active_workflows"] == 100
        
        workflows = orchestrator.get_active_workflows()
        assert len(workflows) == 100


if __name__ == "__main__":
    pytest.main([__file__])
