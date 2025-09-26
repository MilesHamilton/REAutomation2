"""
Unit tests for QualificationAgent
"""
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch

# Handle optional imports gracefully for testing
try:
    from src.agents.qualification_agent import QualificationAgent
    from src.agents.models import (
        AgentType, AgentResponse, WorkflowContext, WorkflowState, 
        AgentMessage, QualificationFactors
    )
    from src.llm.models import LLMResponse
except ImportError:
    # Skip tests if dependencies are not available
    pytest.skip("Agent dependencies not available", allow_module_level=True)


class TestQualificationAgent:
    """Test suite for QualificationAgent"""

    @pytest.fixture
    def agent(self):
        """Create qualification agent instance"""
        return QualificationAgent()

    @pytest.fixture
    def qualifying_context(self, sample_lead_data):
        """Context in qualifying state"""
        return WorkflowContext(
            call_id="test-call-qualifying",
            workflow_state=WorkflowState.QUALIFYING,
            conversation_history=[
                AgentMessage(
                    agent_type=AgentType.CONVERSATION,
                    content="I'm interested in learning more about your services",
                    timestamp=time.time() - 30
                ),
                AgentMessage(
                    agent_type=AgentType.QUALIFICATION,
                    content="Great! Let me ask you a few questions to see how we can best help you.",
                    timestamp=time.time() - 20
                )
            ],
            lead_data=sample_lead_data,
            qualification_score=0.2,
            created_at=time.time() - 120,
            updated_at=time.time() - 20,
            metadata={
                "qualification_factors": {
                    "intent": 0.3,
                    "budget": 0.1,
                    "timeline": 0.2,
                    "authority": 0.4,
                    "needs": 0.3
                }
            }
        )

    @pytest.fixture
    def mock_qualification_analysis(self):
        """Mock qualification analysis response"""
        return {
            "intent_indicators": {"score": 0.7, "evidence": "User expressed interest in solution"},
            "budget_indicators": {"score": 0.5, "evidence": "Mentioned budget considerations"},
            "timeline_indicators": {"score": 0.6, "evidence": "Wants solution within 3 months"},
            "authority_indicators": {"score": 0.8, "evidence": "Decision maker for this area"},
            "needs_indicators": {"score": 0.7, "evidence": "Clear pain points identified"},
            "overall_sentiment": "positive",
            "engagement_level": 0.8
        }

    def test_agent_initialization(self, agent):
        """Test qualification agent initialization"""
        assert agent.agent_type == AgentType.QUALIFICATION
        assert len(agent.qualification_questions) > 0
        
        # Verify all qualification factors are covered
        factors = {q["factor"] for q in agent.qualification_questions}
        expected_factors = {"intent", "budget", "timeline", "authority", "needs"}
        assert factors == expected_factors

    def test_can_handle_qualifying_state(self, agent, qualifying_context):
        """Test that agent can handle qualifying state"""
        assert agent.can_handle(qualifying_context) is True

    def test_cannot_handle_other_states(self, agent, qualifying_context):
        """Test that agent cannot handle non-qualifying states"""
        other_states = [
            WorkflowState.INITIALIZING,
            WorkflowState.GREETING,
            WorkflowState.HANDLING_OBJECTION,
            WorkflowState.SCHEDULING,
            WorkflowState.CLOSING
        ]
        
        for state in other_states:
            qualifying_context.workflow_state = state
            assert agent.can_handle(qualifying_context) is False

    def test_confidence_threshold(self, agent):
        """Test qualification agent confidence threshold"""
        assert agent.get_confidence_threshold() == 0.8

    def test_get_system_prompt(self, agent, qualifying_context):
        """Test system prompt generation"""
        prompt = agent.get_system_prompt(qualifying_context)
        
        assert "qualification agent" in prompt.lower()
        assert "b2b lead qualification" in prompt.lower()
        assert qualifying_context.lead_data["company"] in prompt
        assert qualifying_context.lead_data["industry"] in prompt
        assert str(qualifying_context.qualification_score) in prompt
        
        # Check that current qualification factors are included
        factors = qualifying_context.metadata["qualification_factors"]
        for factor_name in factors.keys():
            assert factor_name in prompt.lower()

    @pytest.mark.asyncio
    async def test_process_ask_question_action(self, agent, qualifying_context, mock_qualification_analysis):
        """Test processing when next action is to ask a question"""
        user_input = "We're looking for a solution to help with our operations"
        
        with patch.object(agent, '_analyze_qualification_response') as mock_analyze:
            mock_analyze.return_value = mock_qualification_analysis
            
            with patch.object(agent, '_determine_next_qualification_action') as mock_action:
                mock_action.return_value = {"action": "ask_question", "factor": "budget"}
                
                with patch.object(agent, '_generate_next_question') as mock_question:
                    mock_question.return_value = "What kind of budget do you typically allocate for solutions like this?"
                    
                    response = await agent.process(qualifying_context, user_input)
                    
                    assert response.agent_type == AgentType.QUALIFICATION
                    assert "budget" in response.response_text.lower()
                    assert "qualification_score" in response.state_updates
                    assert "metadata.qualification_factors" in response.state_updates
                    
                    mock_analyze.assert_called_once_with(qualifying_context, user_input)
                    mock_question.assert_called_once_with(qualifying_context, "budget")

    @pytest.mark.asyncio
    async def test_process_complete_qualification_action(self, agent, qualifying_context, mock_qualification_analysis):
        """Test processing when qualification is complete"""
        user_input = "Yes, I have the authority to make this decision and we need it ASAP"
        
        with patch.object(agent, '_analyze_qualification_response') as mock_analyze:
            mock_analyze.return_value = mock_qualification_analysis
            
            with patch.object(agent, '_determine_next_qualification_action') as mock_action:
                mock_action.return_value = {"action": "complete_qualification"}
                
                with patch.object(agent, '_generate_qualification_summary') as mock_summary:
                    mock_summary.return_value = "Based on what you've shared, there's a real opportunity..."
                    
                    with patch.object(agent, '_determine_next_workflow_state') as mock_state:
                        mock_state.return_value = WorkflowState.SCHEDULING
                        
                        with patch('src.agents.qualification_agent.settings') as mock_settings:
                            mock_settings.tier_escalation_threshold = 0.7
                            
                            response = await agent.process(qualifying_context, user_input)
                            
                            assert response.agent_type == AgentType.QUALIFICATION
                            assert "opportunity" in response.response_text.lower()
                            assert response.state_updates["workflow_state"] == WorkflowState.SCHEDULING
                            assert "qualification_score" in response.state_updates
                            # Should escalate tier if score is high enough
                            assert response.should_escalate_tier is True

    @pytest.mark.asyncio
    async def test_process_continue_conversation_action(self, agent, qualifying_context, mock_qualification_analysis):
        """Test processing when action is to continue conversation"""
        user_input = "That's an interesting question, let me think about it"
        
        with patch.object(agent, '_analyze_qualification_response') as mock_analyze:
            mock_analyze.return_value = mock_qualification_analysis
            
            with patch.object(agent, '_determine_next_qualification_action') as mock_action:
                mock_action.return_value = {"action": "continue_conversation"}
                
                with patch.object(agent, '_generate_llm_response') as mock_generate:
                    mock_generate.return_value = "Of course, take your time. Let me provide some context..."
                    
                    response = await agent.process(qualifying_context, user_input)
                    
                    assert response.agent_type == AgentType.QUALIFICATION
                    assert "context" in response.response_text.lower()
                    assert "qualification_score" in response.state_updates
                    assert "workflow_state" not in response.state_updates  # No state change

    @pytest.mark.asyncio
    async def test_process_exception_handling(self, agent, qualifying_context):
        """Test exception handling in process method"""
        user_input = "Test input"
        
        with patch.object(agent, '_analyze_qualification_response') as mock_analyze:
            mock_analyze.side_effect = Exception("Test error")
            
            response = await agent.process(qualifying_context, user_input)
            
            assert response.agent_type == AgentType.QUALIFICATION
            assert "helpful to know" in response.response_text.lower()

    @pytest.mark.asyncio
    async def test_analyze_qualification_response_success(self, agent, qualifying_context, mock_qualification_analysis):
        """Test successful qualification response analysis"""
        user_input = "We have a $50k budget and need this implemented within 3 months"
        
        with patch.object(agent, '_generate_structured_response') as mock_generate:
            mock_generate.return_value = mock_qualification_analysis
            
            result = await agent._analyze_qualification_response(qualifying_context, user_input)
            
            assert result == mock_qualification_analysis
            assert result["budget_indicators"]["score"] == 0.5
            assert result["timeline_indicators"]["score"] == 0.6
            assert result["overall_sentiment"] == "positive"

    @pytest.mark.asyncio
    async def test_analyze_qualification_response_fallback(self, agent, qualifying_context):
        """Test qualification response analysis fallback when LLM fails"""
        user_input = "Test input"
        
        with patch.object(agent, '_generate_structured_response') as mock_generate:
            mock_generate.return_value = None  # Simulate LLM failure
            
            result = await agent._analyze_qualification_response(qualifying_context, user_input)
            
            # Should return default analysis
            assert result["intent_indicators"]["score"] == 0.3
            assert result["budget_indicators"]["score"] == 0.3
            assert result["overall_sentiment"] == "neutral"

    def test_update_qualification_factors_new_information(self, agent, qualifying_context, mock_qualification_analysis):
        """Test updating qualification factors with new information"""
        factors = agent._update_qualification_factors(qualifying_context, mock_qualification_analysis)
        
        assert isinstance(factors, QualificationFactors)
        # Should incorporate both existing and new information
        assert factors.intent > qualifying_context.metadata["qualification_factors"]["intent"]
        assert factors.budget > qualifying_context.metadata["qualification_factors"]["budget"]
        assert factors.overall_score > 0

    def test_update_qualification_factors_empty_context(self, agent, mock_qualification_analysis):
        """Test updating qualification factors with empty context"""
        empty_context = WorkflowContext(
            call_id="test",
            workflow_state=WorkflowState.QUALIFYING,
            conversation_history=[],
            lead_data={},
            metadata={}
        )
        
        factors = agent._update_qualification_factors(empty_context, mock_qualification_analysis)
        
        assert isinstance(factors, QualificationFactors)
        assert factors.intent == 0.7  # Should use new scores directly
        assert factors.budget == 0.5
        assert factors.timeline == 0.6

    @pytest.mark.asyncio
    async def test_determine_next_qualification_action_ask_question(self, agent, qualifying_context):
        """Test determining next action - ask question"""
        # Create factors with low budget score
        factors = QualificationFactors(
            intent=0.7, budget=0.2, timeline=0.6, authority=0.8, needs=0.7
        )
        
        # Short conversation (< 6 messages)
        qualifying_context.conversation_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(3)
        ]
        
        action = await agent._determine_next_qualification_action(qualifying_context, factors)
        
        assert action["action"] == "ask_question"
        assert action["factor"] == "budget"  # Lowest scoring factor

    @pytest.mark.asyncio
    async def test_determine_next_qualification_action_complete_high_score(self, agent, qualifying_context):
        """Test determining next action - complete qualification (high score)"""
        # Create factors with high overall score
        factors = QualificationFactors(
            intent=0.8, budget=0.7, timeline=0.8, authority=0.9, needs=0.8
        )
        
        action = await agent._determine_next_qualification_action(qualifying_context, factors)
        
        assert action["action"] == "complete_qualification"

    @pytest.mark.asyncio
    async def test_determine_next_qualification_action_complete_long_conversation(self, agent, qualifying_context):
        """Test determining next action - complete qualification (long conversation)"""
        # Create long conversation (>= 8 messages)
        qualifying_context.conversation_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(10)
        ]
        
        factors = QualificationFactors(
            intent=0.5, budget=0.4, timeline=0.6, authority=0.5, needs=0.5
        )
        
        action = await agent._determine_next_qualification_action(qualifying_context, factors)
        
        assert action["action"] == "complete_qualification"

    @pytest.mark.asyncio
    async def test_determine_next_qualification_action_continue(self, agent, qualifying_context):
        """Test determining next action - continue conversation"""
        # Medium scores, medium conversation length
        factors = QualificationFactors(
            intent=0.6, budget=0.5, timeline=0.6, authority=0.5, needs=0.6
        )
        
        qualifying_context.conversation_history = [
            AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {i}", timestamp=time.time())
            for i in range(6)
        ]
        
        action = await agent._determine_next_qualification_action(qualifying_context, factors)
        
        assert action["action"] == "continue_conversation"

    @pytest.mark.asyncio
    async def test_generate_next_question_intent(self, agent, qualifying_context):
        """Test generating next question for intent factor"""
        question = await agent._generate_next_question(qualifying_context, "intent")
        
        assert "looking for solutions" in question.lower()
        assert qualifying_context.lead_data["industry"] in question.lower()
        # Should have conversational lead-in
        lead_ins = ["that makes sense", "appreciate", "thanks", "good to know"]
        assert any(lead_in in question.lower() for lead_in in lead_ins)

    @pytest.mark.asyncio
    async def test_generate_next_question_budget(self, agent, qualifying_context):
        """Test generating next question for budget factor"""
        question = await agent._generate_next_question(qualifying_context, "budget")
        
        assert "budget" in question.lower()
        assert "allocate" in question.lower()

    @pytest.mark.asyncio
    async def test_generate_next_question_unknown_factor(self, agent, qualifying_context):
        """Test generating next question for unknown factor"""
        question = await agent._generate_next_question(qualifying_context, "unknown_factor")
        
        # Should default to first question (intent)
        assert "looking for solutions" in question.lower()

    @pytest.mark.asyncio
    async def test_generate_qualification_summary_high_score(self, agent, qualifying_context):
        """Test generating qualification summary for high score"""
        factors = QualificationFactors(
            intent=0.8, budget=0.8, timeline=0.8, authority=0.9, needs=0.8
        )
        
        with patch('src.agents.qualification_agent.settings') as mock_settings:
            mock_settings.qualification_threshold = 0.7
            
            summary = await agent._generate_qualification_summary(qualifying_context, factors)
            
            assert "real opportunity" in summary.lower()
            assert "demo" in summary.lower()

    @pytest.mark.asyncio
    async def test_generate_qualification_summary_medium_score(self, agent, qualifying_context):
        """Test generating qualification summary for medium score"""
        factors = QualificationFactors(
            intent=0.6, budget=0.5, timeline=0.6, authority=0.6, needs=0.6
        )
        
        with patch('src.agents.qualification_agent.settings') as mock_settings:
            mock_settings.qualification_threshold = 0.7
            
            summary = await agent._generate_qualification_summary(qualifying_context, factors)
            
            assert "interesting possibilities" in summary.lower()
            assert "15 minutes" in summary.lower()

    @pytest.mark.asyncio
    async def test_generate_qualification_summary_low_score(self, agent, qualifying_context):
        """Test generating qualification summary for low score"""
        factors = QualificationFactors(
            intent=0.3, budget=0.2, timeline=0.4, authority=0.3, needs=0.3
        )
        
        summary = await agent._generate_qualification_summary(qualifying_context, factors)
        
        assert "appreciate you taking the time" in summary.lower()
        assert "few months" in summary.lower()

    def test_determine_next_workflow_state_high_score(self, agent):
        """Test determining next workflow state for high qualification score"""
        with patch('src.agents.qualification_agent.settings') as mock_settings:
            mock_settings.qualification_threshold = 0.7
            
            state = agent._determine_next_workflow_state(0.8)
            assert state == WorkflowState.SCHEDULING

    def test_determine_next_workflow_state_medium_score(self, agent):
        """Test determining next workflow state for medium qualification score"""
        with patch('src.agents.qualification_agent.settings') as mock_settings:
            mock_settings.qualification_threshold = 0.7
            
            state = agent._determine_next_workflow_state(0.6)
            assert state == WorkflowState.CLOSING

    def test_determine_next_workflow_state_low_score(self, agent):
        """Test determining next workflow state for low qualification score"""
        with patch('src.agents.qualification_agent.settings') as mock_settings:
            mock_settings.qualification_threshold = 0.7
            
            state = agent._determine_next_workflow_state(0.3)
            assert state == WorkflowState.CLOSING

    def test_get_default_qualification_analysis(self, agent):
        """Test getting default qualification analysis"""
        analysis = agent._get_default_qualification_analysis()
        
        assert "intent_indicators" in analysis
        assert "budget_indicators" in analysis
        assert "timeline_indicators" in analysis
        assert "authority_indicators" in analysis
        assert "needs_indicators" in analysis
        assert analysis["overall_sentiment"] == "neutral"
        assert analysis["engagement_level"] == 0.5

    def test_qualification_factors_overall_score(self):
        """Test QualificationFactors overall score calculation"""
        factors = QualificationFactors(
            intent=0.8, budget=0.6, timeline=0.7, authority=0.9, needs=0.5
        )
        
        expected_score = (0.8 + 0.6 + 0.7 + 0.9 + 0.5) / 5
        assert factors.overall_score == expected_score


class TestQualificationAgentIntegration:
    """Integration tests for QualificationAgent with mocked LLM service"""

    @pytest.fixture
    def agent(self):
        return QualificationAgent()

    @pytest.mark.asyncio
    async def test_full_qualification_flow_high_score(self, agent, qualifying_context, mock_llm_service):
        """Test complete qualification flow resulting in high score"""
        user_input = "Yes, we have a $100k budget and need this implemented within 2 months. I'm the decision maker."
        
        # Mock structured response for qualification analysis
        mock_llm_service.generate_structured_response.return_value = LLMResponse(
            content="analysis",
            structured_data={
                "intent_indicators": {"score": 0.9, "evidence": "Clear buying intent"},
                "budget_indicators": {"score": 0.9, "evidence": "$100k budget mentioned"},
                "timeline_indicators": {"score": 0.8, "evidence": "2 month timeline"},
                "authority_indicators": {"score": 1.0, "evidence": "Decision maker"},
                "needs_indicators": {"score": 0.8, "evidence": "Clear needs expressed"},
                "overall_sentiment": "positive",
                "engagement_level": 0.9
            }
        )
        
        with patch('src.agents.qualification_agent.llm_service', mock_llm_service):
            with patch('src.agents.qualification_agent.settings') as mock_settings:
                mock_settings.qualification_threshold = 0.7
                mock_settings.tier_escalation_threshold = 0.8
                
                response = await agent.process(qualifying_context, user_input)
                
                assert response.agent_type == AgentType.QUALIFICATION
                assert response.state_updates["workflow_state"] == WorkflowState.SCHEDULING
                assert response.should_escalate_tier is True
                assert response.state_updates["qualification_score"] > 0.8

    @pytest.mark.asyncio
    async def test_qualification_with_llm_integration(self, agent, qualifying_context, mock_llm_service):
        """Test qualification with actual LLM service integration"""
        user_input = "We're exploring options but haven't set a budget yet"
        
        # Mock structured analysis response
        mock_llm_service.generate_structured_response.return_value = LLMResponse(
            content="analysis",
            structured_data={
                "intent_indicators": {"score": 0.6, "evidence": "Exploring options"},
                "budget_indicators": {"score": 0.2, "evidence": "No budget set"},
                "timeline_indicators": {"score": 0.4, "evidence": "No urgency"},
                "authority_indicators": {"score": 0.5, "evidence": "Unclear authority"},
                "needs_indicators": {"score": 0.5, "evidence": "Some interest"},
                "overall_sentiment": "neutral",
                "engagement_level": 0.6
            }
        )
        
        with patch('src.agents.qualification_agent.llm_service', mock_llm_service):
            response = await agent.process(qualifying_context, user_input)
            
            assert response.agent_type == AgentType.QUALIFICATION
            assert "qualification_score" in response.state_updates
            assert "metadata.qualification_factors" in response.state_updates
            
            # Verify LLM service was called for structured analysis
            mock_llm_service.generate_structured_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_conversational_question_generation(self, agent, qualifying_context):
        """Test that generated questions are conversational"""
        # Test different conversation lengths to verify lead-in variety
        for i in range(4):
            qualifying_context.conversation_history = [
                AgentMessage(agent_type=AgentType.CONVERSATION, content=f"Message {j}", timestamp=time.time())
                for j in range(i + 1)
            ]
            
            question = await agent._generate_next_question(qualifying_context, "budget")
            
            # Should have conversational lead-in
            lead_ins = ["that makes sense", "appreciate", "thanks", "good to know"]
            assert any(lead_in in question.lower() for lead_in in lead_ins)
            assert "budget" in question.lower()


if __name__ == "__main__":
    pytest.main([__file__])
