"""
Unit tests for ConversationAgent
"""
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

# Handle optional imports gracefully for testing
try:
    from src.agents.conversation_agent import ConversationAgent
    from src.agents.models import (
        AgentType, AgentResponse, WorkflowContext, WorkflowState, 
        AgentMessage, AgentDecision
    )
    from src.llm.models import LLMResponse
except ImportError:
    # Skip tests if dependencies are not available
    pytest.skip("Agent dependencies not available", allow_module_level=True)


class TestConversationAgent:
    """Test suite for ConversationAgent"""

    @pytest.fixture
    def agent(self):
        """Create conversation agent instance"""
        return ConversationAgent()

    @pytest.fixture
    def greeting_context(self, sample_lead_data):
        """Context in greeting state"""
        return WorkflowContext(
            call_id="test-call-greeting",
            workflow_state=WorkflowState.INITIALIZING,
            conversation_history=[],
            lead_data=sample_lead_data,
            qualification_score=0.0,
            created_at=time.time(),
            updated_at=time.time()
        )

    @pytest.fixture
    def introduction_context(self, sample_lead_data):
        """Context in introduction state"""
        return WorkflowContext(
            call_id="test-call-intro",
            workflow_state=WorkflowState.GREETING,
            conversation_history=[
                AgentMessage(
                    agent_type=AgentType.CONVERSATION,
                    content="Hi John, this is calling from ABC Realty. How are you doing today?",
                    timestamp=time.time() - 10
                )
            ],
            lead_data=sample_lead_data,
            qualification_score=0.0,
            created_at=time.time() - 60,
            updated_at=time.time() - 10
        )

    @pytest.fixture
    def general_conversation_context(self, sample_lead_data):
        """Context in general conversation state"""
        return WorkflowContext(
            call_id="test-call-general",
            workflow_state=WorkflowState.QUALIFYING,
            conversation_history=[
                AgentMessage(agent_type=AgentType.CONVERSATION, content="Hello", timestamp=time.time() - 30),
                AgentMessage(agent_type=AgentType.CONVERSATION, content="Hi there", timestamp=time.time() - 25),
                AgentMessage(agent_type=AgentType.CONVERSATION, content="How can I help?", timestamp=time.time() - 20),
                AgentMessage(agent_type=AgentType.CONVERSATION, content="Tell me more", timestamp=time.time() - 15),
            ],
            lead_data=sample_lead_data,
            qualification_score=0.3,
            created_at=time.time() - 120,
            updated_at=time.time() - 15
        )

    def test_agent_initialization(self, agent):
        """Test conversation agent initialization"""
        assert agent.agent_type == AgentType.CONVERSATION
        assert len(agent.conversation_starters) > 0
        assert all("{name}" in starter for starter in agent.conversation_starters)

    def test_can_handle_appropriate_states(self, agent, greeting_context):
        """Test that agent can handle appropriate workflow states"""
        # Should handle these states
        appropriate_states = [
            WorkflowState.INITIALIZING,
            WorkflowState.GREETING,
            WorkflowState.QUALIFYING,
            WorkflowState.CLOSING
        ]
        
        for state in appropriate_states:
            greeting_context.workflow_state = state
            assert agent.can_handle(greeting_context) is True

    def test_cannot_handle_specialized_states(self, agent, greeting_context):
        """Test that agent cannot handle specialized states"""
        # Should not handle these states
        specialized_states = [
            WorkflowState.HANDLING_OBJECTION,
            WorkflowState.SCHEDULING
        ]
        
        for state in specialized_states:
            greeting_context.workflow_state = state
            assert agent.can_handle(greeting_context) is False

    def test_confidence_threshold(self, agent):
        """Test conversation agent confidence threshold"""
        assert agent.get_confidence_threshold() == 0.6

    def test_get_system_prompt(self, agent, greeting_context):
        """Test system prompt generation"""
        prompt = agent.get_system_prompt(greeting_context)
        
        assert "conversationalist" in prompt.lower()
        assert "outbound sales call" in prompt.lower()
        assert greeting_context.lead_data["name"] in prompt
        assert greeting_context.lead_data["company"] in prompt
        assert str(greeting_context.workflow_state) in prompt
        assert str(greeting_context.qualification_score) in prompt

    @pytest.mark.asyncio
    async def test_process_greeting_receptive_response(self, agent, greeting_context):
        """Test processing receptive greeting response"""
        user_input = "Good, thanks! What's this about?"
        
        # Mock greeting analysis to return receptive
        with patch.object(agent, '_analyze_greeting_response') as mock_analyze:
            mock_analyze.return_value = {
                "receptive": True,
                "engagement_level": 0.8,
                "sentiment": "positive"
            }
            
            with patch.object(agent, '_generate_introduction') as mock_intro:
                mock_intro.return_value = "Thanks John! I'm calling because..."
                
                response = await agent.process(greeting_context, user_input)
                
                assert response.agent_type == AgentType.CONVERSATION
                assert response.response_text == "Thanks John! I'm calling because..."
                assert response.state_updates["workflow_state"] == WorkflowState.QUALIFYING
                mock_analyze.assert_called_once_with(greeting_context, user_input)
                mock_intro.assert_called_once_with(greeting_context)

    @pytest.mark.asyncio
    async def test_process_greeting_non_receptive_response(self, agent, greeting_context):
        """Test processing non-receptive greeting response"""
        user_input = "I'm really busy right now"
        
        # Mock greeting analysis to return non-receptive
        with patch.object(agent, '_analyze_greeting_response') as mock_analyze:
            mock_analyze.return_value = {
                "receptive": False,
                "objection_type": "time",
                "engagement_level": 0.2,
                "sentiment": "negative"
            }
            
            with patch.object(agent, '_handle_initial_objection') as mock_objection:
                mock_objection.return_value = "I completely understand you're busy..."
                
                response = await agent.process(greeting_context, user_input)
                
                assert response.agent_type == AgentType.CONVERSATION
                assert response.response_text == "I completely understand you're busy..."
                assert response.state_updates["workflow_state"] == WorkflowState.HANDLING_OBJECTION
                mock_objection.assert_called_once_with(greeting_context, "time")

    @pytest.mark.asyncio
    async def test_process_introduction_phase(self, agent, introduction_context, mock_llm_service):
        """Test processing introduction phase"""
        user_input = "Tell me more about what you do"
        
        with patch.object(agent, '_generate_llm_response') as mock_generate:
            mock_generate.return_value = "We help businesses like yours save money..."
            
            with patch.object(agent, '_should_transition_to_qualification') as mock_transition:
                mock_transition.return_value = True
                
                response = await agent.process(introduction_context, user_input)
                
                assert response.agent_type == AgentType.CONVERSATION
                assert response.response_text == "We help businesses like yours save money..."
                assert response.state_updates["workflow_state"] == WorkflowState.QUALIFYING
                mock_generate.assert_called_once_with(context=introduction_context, user_input=user_input)
                mock_transition.assert_called_once_with(introduction_context, user_input)

    @pytest.mark.asyncio
    async def test_process_general_conversation(self, agent, general_conversation_context):
        """Test processing general conversation"""
        user_input = "That sounds interesting, how does it work?"
        
        with patch.object(agent, '_generate_llm_response') as mock_generate:
            mock_generate.return_value = "Great question! Here's how it works..."
            
            with patch.object(agent, '_make_conversation_decision') as mock_decision:
                mock_decision.return_value = AgentDecision(
                    agent_type=AgentType.CONVERSATION,
                    action="continue_conversation",
                    reasoning="User is engaged",
                    confidence=0.8
                )
                
                with patch.object(agent, '_should_escalate_tier') as mock_escalate:
                    mock_escalate.return_value = False
                    
                    response = await agent.process(general_conversation_context, user_input)
                    
                    assert response.agent_type == AgentType.CONVERSATION
                    assert response.response_text == "Great question! Here's how it works..."
                    assert response.decision.action == "continue_conversation"
                    assert response.should_escalate_tier is False

    @pytest.mark.asyncio
    async def test_process_with_tier_escalation(self, agent, general_conversation_context):
        """Test processing with tier escalation"""
        user_input = "This is exactly what we need!"
        
        with patch.object(agent, '_generate_llm_response') as mock_generate:
            mock_generate.return_value = "Excellent! Let me tell you more..."
            
            with patch.object(agent, '_make_conversation_decision') as mock_decision:
                mock_decision.return_value = None
                
                with patch.object(agent, '_should_escalate_tier') as mock_escalate:
                    mock_escalate.return_value = True
                    
                    response = await agent.process(general_conversation_context, user_input)
                    
                    assert response.should_escalate_tier is True

    @pytest.mark.asyncio
    async def test_process_exception_handling(self, agent, greeting_context):
        """Test exception handling in process method"""
        user_input = "Hello"
        
        with patch.object(agent, '_handle_greeting') as mock_handle:
            mock_handle.side_effect = Exception("Test error")
            
            response = await agent.process(greeting_context, user_input)
            
            assert response.agent_type == AgentType.CONVERSATION
            assert "could you repeat that" in response.response_text.lower()

    @pytest.mark.asyncio
    async def test_analyze_greeting_response_receptive(self, agent, greeting_context):
        """Test greeting response analysis for receptive user"""
        user_input = "Good morning! What can I do for you?"
        
        mock_structured_data = {
            "receptive": True,
            "objection_type": "other",
            "engagement_level": 0.9,
            "sentiment": "positive"
        }
        
        with patch.object(agent, '_generate_structured_response') as mock_generate:
            mock_generate.return_value = mock_structured_data
            
            result = await agent._analyze_greeting_response(greeting_context, user_input)
            
            assert result["receptive"] is True
            assert result["sentiment"] == "positive"
            assert result["engagement_level"] == 0.9

    @pytest.mark.asyncio
    async def test_analyze_greeting_response_non_receptive(self, agent, greeting_context):
        """Test greeting response analysis for non-receptive user"""
        user_input = "I'm not interested, please remove me from your list"
        
        mock_structured_data = {
            "receptive": False,
            "objection_type": "interest",
            "engagement_level": 0.1,
            "sentiment": "negative"
        }
        
        with patch.object(agent, '_generate_structured_response') as mock_generate:
            mock_generate.return_value = mock_structured_data
            
            result = await agent._analyze_greeting_response(greeting_context, user_input)
            
            assert result["receptive"] is False
            assert result["objection_type"] == "interest"
            assert result["sentiment"] == "negative"

    @pytest.mark.asyncio
    async def test_analyze_greeting_response_fallback(self, agent, greeting_context):
        """Test greeting response analysis fallback when LLM fails"""
        user_input = "Hello"
        
        with patch.object(agent, '_generate_structured_response') as mock_generate:
            mock_generate.return_value = None  # Simulate LLM failure
            
            result = await agent._analyze_greeting_response(greeting_context, user_input)
            
            # Should return default fallback
            assert result["receptive"] is False
            assert result["objection_type"] == "time"

    @pytest.mark.asyncio
    async def test_generate_introduction(self, agent, greeting_context):
        """Test introduction generation"""
        introduction = await agent._generate_introduction(greeting_context)
        
        assert greeting_context.lead_data["name"] in introduction
        assert greeting_context.lead_data["company"] in introduction
        assert "save time and money" in introduction.lower()
        assert "quick question" in introduction.lower()

    @pytest.mark.asyncio
    async def test_handle_initial_objection_time(self, agent, greeting_context):
        """Test handling time objection"""
        response = await agent._handle_initial_objection(greeting_context, "time")
        
        assert "busy" in response.lower()
        assert "30 seconds" in response or "minute" in response
        assert "fair enough" in response.lower()

    @pytest.mark.asyncio
    async def test_handle_initial_objection_interest(self, agent, greeting_context):
        """Test handling interest objection"""
        response = await agent._handle_initial_objection(greeting_context, "interest")
        
        assert "hear that a lot" in response.lower()
        assert "save money" in response.lower()
        assert "example" in response.lower()

    @pytest.mark.asyncio
    async def test_handle_initial_objection_authority(self, agent, greeting_context):
        """Test handling authority objection"""
        response = await agent._handle_initial_objection(greeting_context, "authority")
        
        assert "no problem" in response.lower()
        assert "decisions" in response.lower()
        assert "someone else" in response.lower()

    @pytest.mark.asyncio
    async def test_handle_initial_objection_other(self, agent, greeting_context):
        """Test handling other objections"""
        response = await agent._handle_initial_objection(greeting_context, "other")
        
        assert "understand" in response.lower()
        assert "save" in response.lower() and "%" in response
        assert "60 seconds" in response.lower()

    @pytest.mark.asyncio
    async def test_should_transition_to_qualification_positive(self, agent, introduction_context):
        """Test qualification transition detection - positive case"""
        positive_inputs = [
            "Tell me more about this",
            "How does that work?",
            "What do you offer?",
            "I'm interested in learning more",
            "That sounds good"
        ]
        
        for user_input in positive_inputs:
            result = await agent._should_transition_to_qualification(introduction_context, user_input)
            assert result is True, f"Should transition for input: {user_input}"

    @pytest.mark.asyncio
    async def test_should_transition_to_qualification_negative(self, agent, introduction_context):
        """Test qualification transition detection - negative case"""
        negative_inputs = [
            "I'm not interested",
            "Please remove me",
            "I'm busy right now",
            "Call me later",
            "No thanks"
        ]
        
        for user_input in negative_inputs:
            result = await agent._should_transition_to_qualification(introduction_context, user_input)
            assert result is False, f"Should not transition for input: {user_input}"

    @pytest.mark.asyncio
    async def test_make_conversation_decision(self, agent, general_conversation_context):
        """Test conversation decision making"""
        user_input = "I need to think about this"
        
        mock_decision = AgentDecision(
            agent_type=AgentType.CONVERSATION,
            action="schedule_callback",
            reasoning="User needs time to consider",
            confidence=0.7
        )
        
        with patch.object(agent, '_make_decision') as mock_make_decision:
            mock_make_decision.return_value = mock_decision
            
            result = await agent._make_conversation_decision(general_conversation_context, user_input)
            
            assert result == mock_decision
            # Verify decision options were provided
            call_args = mock_make_decision.call_args
            decision_options = call_args[0][2]  # Third argument
            expected_actions = ["continue_conversation", "transition_to_qualification", "handle_objection", "schedule_callback"]
            assert all(action in decision_options for action in expected_actions)

    def test_should_escalate_tier_positive(self, agent, general_conversation_context):
        """Test tier escalation decision - positive case"""
        # Set up context for escalation
        general_conversation_context.conversation_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(8)  # 8 messages (>= 6 threshold)
        ]
        general_conversation_context.qualification_score = 0.5  # >= 0.4 threshold
        general_conversation_context.tier_escalated = False
        
        result = agent._should_escalate_tier(general_conversation_context)
        assert result is True

    def test_should_escalate_tier_negative_cases(self, agent, general_conversation_context):
        """Test tier escalation decision - negative cases"""
        # Test case 1: Not enough conversation
        general_conversation_context.conversation_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(3)  # Only 3 messages (< 6 threshold)
        ]
        general_conversation_context.qualification_score = 0.5
        general_conversation_context.tier_escalated = False
        
        result = agent._should_escalate_tier(general_conversation_context)
        assert result is False
        
        # Test case 2: Low qualification score
        general_conversation_context.conversation_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(8)
        ]
        general_conversation_context.qualification_score = 0.2  # < 0.4 threshold
        general_conversation_context.tier_escalated = False
        
        result = agent._should_escalate_tier(general_conversation_context)
        assert result is False
        
        # Test case 3: Already escalated
        general_conversation_context.conversation_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(8)
        ]
        general_conversation_context.qualification_score = 0.5
        general_conversation_context.tier_escalated = True  # Already escalated
        
        result = agent._should_escalate_tier(general_conversation_context)
        assert result is False


class TestConversationAgentIntegration:
    """Integration tests for ConversationAgent with mocked LLM service"""

    @pytest.fixture
    def agent(self):
        return ConversationAgent()

    @pytest.mark.asyncio
    async def test_full_greeting_flow_receptive(self, agent, greeting_context, mock_llm_service):
        """Test complete greeting flow for receptive user"""
        user_input = "Hi there, I'm doing well. What's this about?"
        
        # Mock structured response for greeting analysis
        mock_llm_service.generate_structured_response.return_value = LLMResponse(
            content="analysis",
            structured_data={
                "receptive": True,
                "objection_type": "other",
                "engagement_level": 0.8,
                "sentiment": "positive"
            }
        )
        
        with patch('src.agents.conversation_agent.llm_service', mock_llm_service):
            response = await agent.process(greeting_context, user_input)
            
            assert response.agent_type == AgentType.CONVERSATION
            assert response.state_updates["workflow_state"] == WorkflowState.QUALIFYING
            assert "Thanks" in response.response_text
            assert greeting_context.lead_data["name"] in response.response_text

    @pytest.mark.asyncio
    async def test_full_greeting_flow_non_receptive(self, agent, greeting_context, mock_llm_service):
        """Test complete greeting flow for non-receptive user"""
        user_input = "I'm really busy and not interested"
        
        # Mock structured response for greeting analysis
        mock_llm_service.generate_structured_response.return_value = LLMResponse(
            content="analysis",
            structured_data={
                "receptive": False,
                "objection_type": "time",
                "engagement_level": 0.2,
                "sentiment": "negative"
            }
        )
        
        with patch('src.agents.conversation_agent.llm_service', mock_llm_service):
            response = await agent.process(greeting_context, user_input)
            
            assert response.agent_type == AgentType.CONVERSATION
            assert response.state_updates["workflow_state"] == WorkflowState.HANDLING_OBJECTION
            assert "busy" in response.response_text.lower()

    @pytest.mark.asyncio
    async def test_conversation_with_llm_integration(self, agent, general_conversation_context, mock_llm_service):
        """Test conversation with actual LLM service integration"""
        user_input = "How much does this cost?"
        
        # Mock LLM response
        mock_llm_service.generate_response.return_value = LLMResponse(
            content="That's a great question! Our pricing depends on your specific needs...",
            usage_tokens=30,
            response_time_ms=200.0
        )
        
        # Mock structured decision response
        mock_llm_service.generate_structured_response.return_value = LLMResponse(
            content="decision",
            structured_data={
                "action": "continue_conversation",
                "reasoning": "User is asking about pricing, showing interest",
                "confidence": 0.8,
                "parameters": {}
            }
        )
        
        with patch('src.agents.conversation_agent.llm_service', mock_llm_service):
            response = await agent.process(general_conversation_context, user_input)
            
            assert response.agent_type == AgentType.CONVERSATION
            assert "great question" in response.response_text.lower()
            assert response.decision.action == "continue_conversation"
            
            # Verify LLM service was called correctly
            mock_llm_service.generate_response.assert_called_once()
            mock_llm_service.generate_structured_response.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
