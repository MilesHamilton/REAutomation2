"""Tests for cost calculator"""

import pytest
from datetime import datetime

from src.cost_control.cost_calculator import CostCalculator, ServiceCosts


class TestServiceCosts:
    """Test ServiceCosts configuration"""

    def test_default_costs(self):
        """Test default cost configuration"""
        costs = ServiceCosts()

        # Local services should be free or very low cost
        assert costs.ollama_local_cost_per_token == 0.0
        assert costs.local_piper_cost_per_char == 0.0
        assert costs.whisper_local_cost_per_minute == 0.0

        # Premium services should have reasonable costs
        assert costs.elevenlabs_cost_per_char > 0
        assert costs.twilio_voice_cost_per_minute > 0

    def test_custom_costs(self):
        """Test custom cost configuration"""
        custom_costs = ServiceCosts(
            ollama_local_cost_per_token=0.001,
            elevenlabs_cost_per_char=0.0003,
            twilio_voice_cost_per_minute=0.01
        )

        assert custom_costs.ollama_local_cost_per_token == 0.001
        assert custom_costs.elevenlabs_cost_per_char == 0.0003
        assert custom_costs.twilio_voice_cost_per_minute == 0.01


class TestCostCalculator:
    """Test CostCalculator functionality"""

    def test_initialize_call_costs(self, cost_calculator):
        """Test initializing cost tracking for a call"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        assert call_id in cost_calculator.call_costs
        costs = cost_calculator.call_costs[call_id]

        assert costs["llm"] == 0.0
        assert costs["tts"] == 0.0
        assert costs["stt"] == 0.0
        assert costs["twilio"] == 0.0
        assert costs["infrastructure"] > 0  # Should have base infrastructure cost
        assert costs["total"] == costs["infrastructure"]
        assert "started_at" in costs

    def test_calculate_llm_cost_local(self, cost_calculator):
        """Test LLM cost calculation for local provider"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        cost = cost_calculator.calculate_llm_cost(
            call_id=call_id,
            provider="ollama",
            input_tokens=100,
            output_tokens=50
        )

        # Local Ollama should be free
        assert cost == 0.0
        assert cost_calculator.call_costs[call_id]["llm"] == 0.0

    def test_calculate_llm_cost_premium(self, cost_calculator):
        """Test LLM cost calculation for premium provider"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        cost = cost_calculator.calculate_llm_cost(
            call_id=call_id,
            provider="openai-gpt-4",
            input_tokens=100,
            output_tokens=50
        )

        # Premium should have cost
        assert cost > 0
        assert cost_calculator.call_costs[call_id]["llm"] == cost
        expected_cost = 150 * cost_calculator.costs.openai_gpt4_cost_per_token
        assert abs(cost - expected_cost) < 0.000001

    def test_calculate_tts_cost_local(self, cost_calculator):
        """Test TTS cost calculation for local provider"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        cost = cost_calculator.calculate_tts_cost(
            call_id=call_id,
            provider="piper",
            character_count=100
        )

        # Local TTS should be free
        assert cost == 0.0
        assert cost_calculator.call_costs[call_id]["tts"] == 0.0

    def test_calculate_tts_cost_premium(self, cost_calculator):
        """Test TTS cost calculation for premium provider"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        cost = cost_calculator.calculate_tts_cost(
            call_id=call_id,
            provider="elevenlabs",
            character_count=100
        )

        # Premium should have cost
        assert cost > 0
        assert cost_calculator.call_costs[call_id]["tts"] == cost
        expected_cost = 100 * cost_calculator.costs.elevenlabs_cost_per_char
        assert abs(cost - expected_cost) < 0.000001

    def test_calculate_stt_cost(self, cost_calculator):
        """Test STT cost calculation"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        # Local STT should be free
        cost_local = cost_calculator.calculate_stt_cost(
            call_id=call_id,
            provider="whisper-local",
            duration_minutes=2.5
        )
        assert cost_local == 0.0

        # API STT should have cost
        cost_api = cost_calculator.calculate_stt_cost(
            call_id=call_id,
            provider="whisper-api",
            duration_minutes=2.5
        )
        assert cost_api > 0
        expected_cost = 2.5 * cost_calculator.costs.whisper_api_cost_per_minute
        assert abs(cost_api - expected_cost) < 0.000001

    def test_calculate_twilio_cost(self, cost_calculator):
        """Test Twilio cost calculation"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        # Voice only
        cost_voice = cost_calculator.calculate_twilio_cost(
            call_id=call_id,
            duration_minutes=3.0,
            sms_count=0
        )
        expected_voice = 3.0 * cost_calculator.costs.twilio_voice_cost_per_minute
        assert abs(cost_voice - expected_voice) < 0.000001

        # Voice + SMS
        cost_combined = cost_calculator.calculate_twilio_cost(
            call_id=call_id,
            duration_minutes=2.0,
            sms_count=2
        )
        expected_combined = (2.0 * cost_calculator.costs.twilio_voice_cost_per_minute +
                           2 * cost_calculator.costs.twilio_sms_cost_per_message)
        # Should be additive (previous cost + new cost)
        assert cost_calculator.call_costs[call_id]["twilio"] == cost_voice + cost_combined

    def test_get_call_cost_breakdown(self, cost_calculator):
        """Test getting call cost breakdown"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        # Add various costs
        cost_calculator.calculate_llm_cost(call_id, "ollama", 100, 50)
        cost_calculator.calculate_tts_cost(call_id, "elevenlabs", 200)
        cost_calculator.calculate_twilio_cost(call_id, 2.0, 0)

        breakdown = cost_calculator.get_call_cost_breakdown(call_id)

        assert "llm" in breakdown
        assert "tts" in breakdown
        assert "twilio" in breakdown
        assert "infrastructure" in breakdown
        assert "total" in breakdown
        assert breakdown["total"] > breakdown["infrastructure"]  # Should have additional costs

    def test_get_total_call_cost(self, cost_calculator):
        """Test getting total call cost"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        # Initially just infrastructure
        initial_cost = cost_calculator.get_total_call_cost(call_id)
        assert initial_cost == cost_calculator.costs.infrastructure_cost_per_call

        # Add some costs
        cost_calculator.calculate_tts_cost(call_id, "elevenlabs", 100)
        cost_calculator.calculate_twilio_cost(call_id, 1.0, 0)

        final_cost = cost_calculator.get_total_call_cost(call_id)
        assert final_cost > initial_cost

    def test_estimate_remaining_cost(self, cost_calculator):
        """Test estimating remaining cost"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        # Estimate for local tier
        local_estimate = cost_calculator.estimate_remaining_cost(
            call_id=call_id,
            estimated_duration_minutes=3.0,
            estimated_messages=5,
            estimated_tts_characters=300,
            tier="local"
        )

        # Should mostly be Twilio cost for local tier
        expected_twilio = 3.0 * cost_calculator.costs.twilio_voice_cost_per_minute
        assert local_estimate >= expected_twilio

        # Estimate for premium tier
        premium_estimate = cost_calculator.estimate_remaining_cost(
            call_id=call_id,
            estimated_duration_minutes=3.0,
            estimated_messages=5,
            estimated_tts_characters=300,
            tier="premium"
        )

        # Premium should be higher due to ElevenLabs cost
        assert premium_estimate > local_estimate

    def test_finalize_call_costs(self, cost_calculator):
        """Test finalizing call costs"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        # Add some costs
        cost_calculator.calculate_tts_cost(call_id, "elevenlabs", 150)
        cost_calculator.calculate_twilio_cost(call_id, 2.5, 1)

        # Finalize
        final_costs = cost_calculator.finalize_call_costs(call_id)

        assert "total" in final_costs
        assert "ended_at" in final_costs
        assert "duration_seconds" in final_costs
        assert final_costs["total"] > 0
        assert cost_calculator.call_costs[call_id]["finalized"] is True

    def test_cleanup_old_calls(self, cost_calculator):
        """Test cleaning up old finalized calls"""
        # Create and finalize multiple calls
        for i in range(5):
            call_id = f"old_call_{i}"
            cost_calculator.initialize_call_costs(call_id)
            cost_calculator.finalize_call_costs(call_id)

        # Manually set old timestamps for testing
        for call_id in cost_calculator.call_costs:
            if cost_calculator.call_costs[call_id].get("finalized"):
                cost_calculator.call_costs[call_id]["ended_at"] = datetime.now().timestamp() - 86400  # 24 hours ago

        initial_count = len(cost_calculator.call_costs)
        cost_calculator.cleanup_old_calls(max_age_hours=12)  # Clean calls older than 12 hours

        # Should have cleaned up old calls
        remaining_count = len(cost_calculator.call_costs)
        assert remaining_count < initial_count

    def test_get_cost_summary(self, cost_calculator):
        """Test getting cost summary"""
        # Create active and finalized calls
        active_call = "active_call"
        finalized_call = "finalized_call"

        cost_calculator.initialize_call_costs(active_call)
        cost_calculator.calculate_twilio_cost(active_call, 1.0, 0)

        cost_calculator.initialize_call_costs(finalized_call)
        cost_calculator.calculate_twilio_cost(finalized_call, 2.0, 0)
        cost_calculator.finalize_call_costs(finalized_call)

        summary = cost_calculator.get_cost_summary()

        assert "active_calls" in summary
        assert "finalized_calls" in summary
        assert "total_active_cost" in summary
        assert "total_finalized_cost" in summary
        assert "total_cost" in summary
        assert "average_cost_per_call" in summary

        assert summary["active_calls"] == 1
        assert summary["finalized_calls"] == 1
        assert summary["total_cost"] > 0

    def test_unknown_providers(self, cost_calculator):
        """Test handling of unknown providers"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        # Unknown providers should default to local/free costs
        llm_cost = cost_calculator.calculate_llm_cost(call_id, "unknown_llm", 100, 50)
        tts_cost = cost_calculator.calculate_tts_cost(call_id, "unknown_tts", 100)
        stt_cost = cost_calculator.calculate_stt_cost(call_id, "unknown_stt", 1.0)

        # Should default to local costs (which are typically 0)
        assert llm_cost == 0.0
        assert tts_cost == 0.0
        assert stt_cost == 0.0

    def test_cost_accumulation(self, cost_calculator):
        """Test that costs accumulate properly"""
        call_id = "test_call_123"
        cost_calculator.initialize_call_costs(call_id)

        initial_total = cost_calculator.get_total_call_cost(call_id)

        # Add costs multiple times
        cost1 = cost_calculator.calculate_tts_cost(call_id, "elevenlabs", 50)
        total_after_1 = cost_calculator.get_total_call_cost(call_id)

        cost2 = cost_calculator.calculate_tts_cost(call_id, "elevenlabs", 50)
        total_after_2 = cost_calculator.get_total_call_cost(call_id)

        # Costs should accumulate
        assert total_after_1 == initial_total + cost1
        assert total_after_2 == total_after_1 + cost2
        assert cost_calculator.call_costs[call_id]["tts"] == cost1 + cost2