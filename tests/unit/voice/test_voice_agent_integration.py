"""Unit tests for voice-agent integration"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.voice.models import CallSession, VoiceCallState, TTSProvider, VoiceMetrics, TTSConfig
from src.voice.circuit_breaker import CircuitBreaker, CircuitState, CircuitBreakerOpenError
from src.voice.response_cache import ResponseCache
from src.monitoring.voice_agent_metrics import VoiceAgentIntegrationMetrics


class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    @pytest.fixture
    def breaker(self):
        return CircuitBreaker(
            failure_threshold=3,
            timeout_seconds=1,
            half_open_max_calls=2
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self, breaker):
        """Test circuit breaker in closed state"""
        async def successful_operation():
            return "success"

        result = await breaker.call(successful_operation)
        assert result == "success"
        assert breaker.get_state() == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, breaker):
        """Test circuit breaker opens after threshold failures"""
        async def failing_operation():
            raise Exception("Service error")

        # Trigger failures
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_operation)

        assert breaker.get_state() == CircuitState.OPEN

        # Next call should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing_operation)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self, breaker):
        """Test circuit breaker transitions to half-open after timeout"""
        async def failing_operation():
            raise Exception("Service error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_operation)

        assert breaker.get_state() == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Should transition to half-open on next attempt
        async def successful_operation():
            return "success"

        result = await breaker.call(successful_operation)
        assert result == "success"
        assert breaker.get_state() == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, breaker):
        """Test circuit breaker closes after successful recovery"""
        async def failing_operation():
            raise Exception("Service error")

        async def successful_operation():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_operation)

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Successful calls should close the circuit
        for _ in range(2):
            await breaker.call(successful_operation)

        assert breaker.get_state() == CircuitState.CLOSED
        assert breaker.failure_count == 0


class TestResponseCache:
    """Test response caching functionality"""

    @pytest.fixture
    def cache(self):
        return ResponseCache(max_size=10, ttl_seconds=2)

    def test_cache_set_and_get(self, cache):
        """Test basic cache set and get"""
        call_id = "call_123"
        user_input = "Hello"
        response = "Hi there!"

        cache.set(call_id, user_input, response)
        cached_response = cache.get(call_id, user_input)

        assert cached_response == response
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss(self, cache):
        """Test cache miss"""
        cached_response = cache.get("call_123", "Hello")

        assert cached_response is None
        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_cache_expiration(self, cache):
        """Test cache entry expiration"""
        import time

        call_id = "call_123"
        user_input = "Hello"
        response = "Hi there!"

        cache.set(call_id, user_input, response)

        # Wait for TTL to expire
        time.sleep(2.1)

        cached_response = cache.get(call_id, user_input)
        assert cached_response is None

    def test_cache_max_size(self, cache):
        """Test cache respects max size"""
        # Fill cache beyond max size
        for i in range(15):
            cache.set(f"call_{i}", f"input_{i}", f"response_{i}")

        stats = cache.get_stats()
        assert stats["size"] <= cache.max_size

    def test_cache_invalidation(self, cache):
        """Test cache invalidation"""
        call_id = "call_123"
        cache.set(call_id, "input_1", "response_1")
        cache.set(call_id, "input_2", "response_2")
        cache.set("call_456", "input_3", "response_3")

        # Invalidate specific call
        cache.invalidate(call_id)

        assert cache.get(call_id, "input_1") is None
        assert cache.get(call_id, "input_2") is None
        assert cache.get("call_456", "input_3") == "response_3"

    def test_cache_hit_rate(self, cache):
        """Test cache hit rate calculation"""
        cache.set("call_1", "input_1", "response_1")

        # One hit
        cache.get("call_1", "input_1")

        # Two misses
        cache.get("call_2", "input_2")
        cache.get("call_3", "input_3")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert abs(stats["hit_rate_pct"] - 33.33) < 0.1


class TestVoiceAgentMetrics:
    """Test voice-agent integration metrics"""

    @pytest.fixture
    def metrics(self):
        return VoiceAgentIntegrationMetrics()

    def test_record_agent_transition(self, metrics):
        """Test recording agent transitions"""
        metrics.record_agent_transition(
            call_id="call_123",
            from_agent="conversation",
            to_agent="qualification",
            transition_duration_ms=50.0,
            context_preserved=True
        )

        stats = metrics.get_transition_stats()
        assert "conversation->qualification" in stats
        assert stats["conversation->qualification"] == 1

    def test_record_state_sync(self, metrics):
        """Test recording state synchronization"""
        metrics.record_state_sync(
            call_id="call_123",
            sync_duration_ms=10.0,
            sync_status="success",
            error_count=0
        )

        summary = metrics.get_summary()
        assert summary["state_synchronization"]["total_syncs"] == 1
        assert summary["state_synchronization"]["failures"] == 0

    def test_record_tier_escalation(self, metrics):
        """Test recording tier escalation"""
        metrics.record_tier_escalation(
            call_id="call_123",
            from_tier="local_piper",
            to_tier="elevenlabs",
            trigger="qualification",
            qualification_score=0.85,
            budget_available=True,
            escalation_approved=True,
            cost_impact=0.05
        )

        summary = metrics.get_summary()
        assert summary["tier_escalations"]["total"] == 1
        assert "local_piper->elevenlabs" in summary["tier_escalations"]["by_type"]

    def test_record_performance(self, metrics):
        """Test recording performance metrics"""
        metrics.record_performance(
            call_id="call_123",
            agent_processing_latency_ms=450.0,
            total_response_latency_ms=1800.0,
            fallback_triggered=False,
            circuit_breaker_tripped=False,
            cache_hit=True
        )

        summary = metrics.get_summary()
        assert summary["performance"]["total_requests"] == 1
        assert summary["performance"]["fallback_count"] == 0

    def test_fallback_rate_calculation(self, metrics):
        """Test fallback rate calculation"""
        # Record 10 requests with 3 fallbacks
        for i in range(10):
            fallback = i < 3
            metrics.record_performance(
                call_id=f"call_{i}",
                agent_processing_latency_ms=400.0,
                total_response_latency_ms=1500.0,
                fallback_triggered=fallback
            )

        fallback_rate = metrics.get_fallback_rate()
        assert abs(fallback_rate - 30.0) < 0.1

    def test_metrics_reset(self, metrics):
        """Test metrics reset"""
        metrics.record_agent_transition("call_1", "a", "b", 10.0)
        metrics.record_state_sync("call_1", 5.0, "success")

        metrics.reset_metrics()

        summary = metrics.get_summary()
        assert summary["agent_transitions"]["total"] == 0
        assert summary["state_synchronization"]["total_syncs"] == 0


class TestCallSessionIntegration:
    """Test CallSession with workflow integration fields"""

    def test_call_session_with_workflow_fields(self):
        """Test CallSession includes workflow integration fields"""
        metrics = VoiceMetrics(call_id="call_123")
        tts_config = TTSConfig(provider=TTSProvider.LOCAL_PIPER)

        session = CallSession(
            call_id="call_123",
            phone_number="+1234567890",
            tts_config=tts_config,
            metrics=metrics,
            workflow_context_id="workflow_456",
            current_agent="conversation",
            agent_transition_history=["conversation"],
            workflow_state="greeting",
            integration_enabled=True
        )

        assert session.workflow_context_id == "workflow_456"
        assert session.current_agent == "conversation"
        assert session.integration_enabled is True
        assert len(session.agent_transition_history) == 1

    def test_call_session_default_workflow_fields(self):
        """Test CallSession has default values for workflow fields"""
        metrics = VoiceMetrics(call_id="call_123")
        tts_config = TTSConfig(provider=TTSProvider.LOCAL_PIPER)

        session = CallSession(
            call_id="call_123",
            phone_number="+1234567890",
            tts_config=tts_config,
            metrics=metrics
        )

        assert session.workflow_context_id is None
        assert session.current_agent is None
        assert session.agent_transition_history == []
        assert session.integration_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
