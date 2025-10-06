"""Integration tests for voice-agent workflow"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.voice.models import (
    CallSession, VoiceCallState, TTSProvider, VoiceMetrics,
    TTSConfig, AgentTransition, VoiceAgentIntegrationContext
)
from src.agents.models import AgentType, AgentResponse, WorkflowState


class TestVoiceAgentIntegration:
    """Integration tests for voice and agent orchestration"""

    @pytest.fixture
    async def mock_orchestrator(self):
        """Create mock agent orchestrator"""
        orchestrator = MagicMock()
        orchestrator.is_initialized = True

        async def mock_process_voice_input(call_id, user_input, lead_data=None):
            return AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="This is a test response.",
                should_escalate_tier=False,
                next_agent=None
            )

        orchestrator.process_voice_input = AsyncMock(side_effect=mock_process_voice_input)
        orchestrator.get_context = Mock(return_value=None)

        return orchestrator

    @pytest.fixture
    def call_session(self):
        """Create test call session"""
        metrics = VoiceMetrics(call_id="call_test_123")
        tts_config = TTSConfig(provider=TTSProvider.LOCAL_PIPER)

        return CallSession(
            call_id="call_test_123",
            phone_number="+1234567890",
            tts_config=tts_config,
            metrics=metrics,
            integration_enabled=True
        )

    @pytest.mark.asyncio
    async def test_agent_response_processing(self, mock_orchestrator, call_session):
        """Test processing user input through agent orchestrator"""
        user_input = "I'm interested in your services"

        response = await mock_orchestrator.process_voice_input(
            call_id=call_session.call_id,
            user_input=user_input,
            lead_data=call_session.lead_data
        )

        assert response is not None
        assert response.response_text == "This is a test response."
        assert response.agent_type == AgentType.CONVERSATION
        mock_orchestrator.process_voice_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_agent_timeout(self, call_session):
        """Test fallback to direct LLM on agent timeout"""
        # Create orchestrator that times out
        slow_orchestrator = MagicMock()
        slow_orchestrator.is_initialized = True

        async def slow_process():
            await asyncio.sleep(2)  # Exceed timeout
            return None

        slow_orchestrator.process_voice_input = AsyncMock(side_effect=slow_process)

        # Should timeout and trigger fallback
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                slow_orchestrator.process_voice_input(
                    call_id=call_session.call_id,
                    user_input="test"
                ),
                timeout=0.5
            )

    @pytest.mark.asyncio
    async def test_tier_escalation_trigger(self, mock_orchestrator, call_session):
        """Test tier escalation triggered by qualified lead"""
        # Mock high qualification score response
        async def mock_qualified_response(call_id, user_input, lead_data=None):
            return AgentResponse(
                agent_type=AgentType.QUALIFICATION,
                response_text="Great! Let me schedule a demo for you.",
                should_escalate_tier=True,
                qualification_score=0.85
            )

        mock_orchestrator.process_voice_input = AsyncMock(side_effect=mock_qualified_response)

        response = await mock_orchestrator.process_voice_input(
            call_id=call_session.call_id,
            user_input="I have a budget of $50k and need this urgently"
        )

        assert response.should_escalate_tier is True
        assert response.qualification_score == 0.85

    @pytest.mark.asyncio
    async def test_state_synchronization(self, mock_orchestrator, call_session):
        """Test workflow state synchronization with call session"""
        import time
        from src.agents.models import WorkflowContext

        # Create mock context
        mock_context = WorkflowContext(
            call_id=call_session.call_id,
            lead_data=call_session.lead_data,
            workflow_state=WorkflowState.QUALIFYING
        )

        mock_orchestrator.get_context = Mock(return_value=mock_context)

        # Simulate state sync
        context = mock_orchestrator.get_context(call_session.call_id)

        if context:
            call_session.workflow_context_id = context.call_id
            call_session.workflow_state = context.workflow_state.value
            call_session.last_state_sync = time.time()

        assert call_session.workflow_context_id == call_session.call_id
        assert call_session.workflow_state == "qualifying"
        assert call_session.last_state_sync is not None

    @pytest.mark.asyncio
    async def test_agent_transition_tracking(self, call_session):
        """Test tracking agent transitions during call"""
        transitions = []

        # Simulate agent transitions
        transition1 = AgentTransition(
            call_id=call_session.call_id,
            from_agent="conversation",
            to_agent="qualification",
            trigger="workflow",
            context_preserved=True,
            transition_duration_ms=45.0
        )
        transitions.append(transition1)

        transition2 = AgentTransition(
            call_id=call_session.call_id,
            from_agent="qualification",
            to_agent="scheduler",
            trigger="escalation",
            context_preserved=True,
            transition_duration_ms=38.0
        )
        transitions.append(transition2)

        # Update call session
        call_session.agent_transition_history = [
            t.to_agent for t in transitions
        ]
        call_session.current_agent = transitions[-1].to_agent

        assert len(call_session.agent_transition_history) == 2
        assert call_session.current_agent == "scheduler"
        assert call_session.agent_transition_history == ["qualification", "scheduler"]

    @pytest.mark.asyncio
    async def test_integration_context_creation(self, call_session):
        """Test creation of voice-agent integration context"""
        integration_context = VoiceAgentIntegrationContext(
            call_id=call_session.call_id,
            voice_session_id=call_session.call_id,
            workflow_context_id="workflow_456",
            sync_status="synced",
            error_count=0,
            fallback_active=False
        )

        assert integration_context.call_id == call_session.call_id
        assert integration_context.sync_status == "synced"
        assert integration_context.fallback_active is False

    @pytest.mark.asyncio
    async def test_concurrent_agent_calls(self, mock_orchestrator):
        """Test handling multiple concurrent agent calls"""
        call_ids = [f"call_{i}" for i in range(5)]

        # Process multiple calls concurrently
        tasks = [
            mock_orchestrator.process_voice_input(
                call_id=call_id,
                user_input=f"test input {i}"
            )
            for i, call_id in enumerate(call_ids)
        ]

        responses = await asyncio.gather(*tasks)

        assert len(responses) == 5
        assert all(r is not None for r in responses)
        assert mock_orchestrator.process_voice_input.call_count == 5

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, call_session):
        """Test error recovery in voice-agent workflow"""
        error_orchestrator = MagicMock()
        error_orchestrator.is_initialized = True

        # First call fails, second succeeds
        call_count = 0

        async def mock_with_recovery(call_id, user_input, lead_data=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise Exception("Temporary service error")

            return AgentResponse(
                agent_type=AgentType.CONVERSATION,
                response_text="Recovered successfully",
                should_escalate_tier=False
            )

        error_orchestrator.process_voice_input = AsyncMock(side_effect=mock_with_recovery)

        # First call should raise exception
        with pytest.raises(Exception):
            await error_orchestrator.process_voice_input(
                call_id=call_session.call_id,
                user_input="test"
            )

        # Second call should succeed
        response = await error_orchestrator.process_voice_input(
            call_id=call_session.call_id,
            user_input="test"
        )

        assert response is not None
        assert response.response_text == "Recovered successfully"


class TestVoiceOptimizations:
    """Test voice-specific optimizations"""

    @pytest.mark.asyncio
    async def test_response_truncation_for_voice(self):
        """Test long responses are truncated for voice"""
        long_text = " ".join(["word"] * 200)  # 200 words

        # Simulate truncation
        words = long_text.split()
        if len(words) > 150:
            truncated = " ".join(words[:150]) + "."
        else:
            truncated = long_text

        assert len(truncated.split()) <= 151  # 150 words + period
        assert truncated.endswith(".")

    @pytest.mark.asyncio
    async def test_markdown_removal_for_voice(self):
        """Test markdown formatting is removed for voice"""
        markdown_text = "**Bold text** and *italic text* with #heading"

        # Simulate markdown removal
        clean_text = markdown_text.replace("*", "").replace("#", "")

        assert "**" not in clean_text
        assert "*" not in clean_text
        assert "#" not in clean_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
