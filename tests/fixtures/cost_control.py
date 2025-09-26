"""Cost control test fixtures and utilities"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.cost_control.cost_calculator import CostCalculator, ServiceCosts
from src.cost_control.budget_manager import BudgetManager, BudgetStatus, BudgetAlert, AlertLevel
from src.cost_control.tier_decision import TierDecisionEngine, TierSwitchDecision
from src.cost_control.cost_service import CostControlService
from src.voice.models import TTSProvider


@pytest.fixture
def test_service_costs():
    """Test service cost configuration"""
    return ServiceCosts(
        # LLM costs (per token)
        ollama_local_cost_per_token=0.0,
        openai_gpt4_cost_per_token=0.03 / 1000,
        openai_gpt35_cost_per_token=0.002 / 1000,

        # TTS costs
        local_piper_cost_per_char=0.0,
        elevenlabs_cost_per_char=0.0002,

        # STT costs
        whisper_local_cost_per_minute=0.0,
        whisper_api_cost_per_minute=0.006,

        # Twilio costs
        twilio_voice_cost_per_minute=0.0085,
        twilio_sms_cost_per_message=0.0075,

        # Infrastructure
        infrastructure_cost_per_call=0.001
    )


@pytest.fixture
def cost_calculator(test_service_costs):
    """Create a cost calculator for testing"""
    return CostCalculator(test_service_costs)


@pytest.fixture
def budget_manager():
    """Create a budget manager for testing"""
    return BudgetManager(
        daily_budget=10.0,  # Lower budget for testing
        weekly_budget=50.0,
        monthly_budget=200.0,
        cost_per_call_limit=0.50,
        emergency_stop_threshold=0.90
    )


@pytest.fixture
def tier_decision_engine():
    """Create a tier decision engine for testing"""
    return TierDecisionEngine(
        qualification_threshold=0.7,
        budget_tier_switch_threshold=0.8,
        cost_savings_threshold=0.02,
        min_conversation_length=3
    )


@pytest.fixture
async def cost_control_service():
    """Create a cost control service for testing"""
    service = CostControlService()
    await service.initialize()
    return service


@pytest.fixture
def sample_budget_alerts():
    """Sample budget alerts for testing"""
    now = datetime.now()
    return [
        BudgetAlert(
            level=AlertLevel.WARNING,
            message="Daily budget 70% utilized",
            budget_type="daily",
            current_spend=7.0,
            budget_limit=10.0,
            utilization_percentage=70.0,
            timestamp=now - timedelta(minutes=30)
        ),
        BudgetAlert(
            level=AlertLevel.CRITICAL,
            message="Daily budget 85% utilized",
            budget_type="daily",
            current_spend=8.5,
            budget_limit=10.0,
            utilization_percentage=85.0,
            timestamp=now - timedelta(minutes=15)
        ),
        BudgetAlert(
            level=AlertLevel.EMERGENCY,
            message="Call exceeded per-call limit",
            budget_type="per_call",
            current_spend=0.55,
            budget_limit=0.50,
            utilization_percentage=110.0,
            timestamp=now,
            call_id="test_call_123"
        )
    ]


@pytest.fixture
def cost_calculation_scenarios():
    """Different cost calculation scenarios for testing"""
    return [
        {
            "name": "local_only_call",
            "llm_provider": "ollama",
            "tts_provider": "piper",
            "stt_provider": "whisper-local",
            "duration_minutes": 3.0,
            "input_tokens": 150,
            "output_tokens": 100,
            "tts_characters": 200,
            "expected_cost_range": (0.025, 0.035)  # Mostly Twilio cost
        },
        {
            "name": "premium_tts_call",
            "llm_provider": "ollama",
            "tts_provider": "elevenlabs",
            "stt_provider": "whisper-local",
            "duration_minutes": 3.0,
            "input_tokens": 150,
            "output_tokens": 100,
            "tts_characters": 200,
            "expected_cost_range": (0.065, 0.075)  # Twilio + ElevenLabs
        },
        {
            "name": "premium_all_call",
            "llm_provider": "openai-gpt-4",
            "tts_provider": "elevenlabs",
            "stt_provider": "whisper-api",
            "duration_minutes": 5.0,
            "input_tokens": 300,
            "output_tokens": 200,
            "tts_characters": 400,
            "expected_cost_range": (0.120, 0.150)  # All premium services
        }
    ]


@pytest.fixture
def tier_switch_test_cases():
    """Test cases for tier switching decisions"""
    return [
        {
            "name": "high_qualification_escalation",
            "current_tier": TTSProvider.LOCAL_PIPER,
            "qualification_score": 0.9,
            "conversation_length": 5,
            "budget_utilization": 0.3,
            "call_cost_so_far": 0.02,
            "lead_data": {"company_size": "enterprise", "decision_maker": True},
            "expected_switch": True,
            "expected_target": TTSProvider.ELEVENLABS,
            "expected_confidence_min": 0.8
        },
        {
            "name": "low_qualification_no_switch",
            "current_tier": TTSProvider.LOCAL_PIPER,
            "qualification_score": 0.3,
            "conversation_length": 4,
            "budget_utilization": 0.5,
            "call_cost_so_far": 0.03,
            "lead_data": {},
            "expected_switch": False,
            "expected_target": TTSProvider.LOCAL_PIPER,
            "expected_confidence_min": 0.6
        },
        {
            "name": "budget_constraint_no_switch",
            "current_tier": TTSProvider.LOCAL_PIPER,
            "qualification_score": 0.85,
            "conversation_length": 6,
            "budget_utilization": 0.95,
            "call_cost_so_far": 0.04,
            "lead_data": {"urgent_timeline": True},
            "expected_switch": False,
            "expected_target": TTSProvider.LOCAL_PIPER,
            "expected_confidence_min": 0.5
        },
        {
            "name": "downgrade_from_premium",
            "current_tier": TTSProvider.ELEVENLABS,
            "qualification_score": 0.2,
            "conversation_length": 10,
            "budget_utilization": 0.9,
            "call_cost_so_far": 0.15,
            "lead_data": {},
            "expected_switch": True,
            "expected_target": TTSProvider.LOCAL_PIPER,
            "expected_confidence_min": 0.7
        }
    ]


@pytest.fixture
def budget_scenarios():
    """Different budget scenarios for testing"""
    return [
        {
            "name": "normal_usage",
            "daily_budget": 50.0,
            "costs_recorded": [2.5, 3.0, 1.5, 4.0],  # Total: 11.0
            "expected_utilization": 0.22,  # 11.0 / 50.0
            "expected_over_budget": False,
            "expected_alert_levels": []
        },
        {
            "name": "warning_threshold",
            "daily_budget": 20.0,
            "costs_recorded": [5.0, 4.0, 3.5, 2.0],  # Total: 14.5
            "expected_utilization": 0.725,  # 14.5 / 20.0
            "expected_over_budget": False,
            "expected_alert_levels": [AlertLevel.WARNING]
        },
        {
            "name": "critical_threshold",
            "daily_budget": 20.0,
            "costs_recorded": [6.0, 5.0, 4.0, 2.5],  # Total: 17.5
            "expected_utilization": 0.875,  # 17.5 / 20.0
            "expected_over_budget": False,
            "expected_alert_levels": [AlertLevel.WARNING, AlertLevel.CRITICAL]
        },
        {
            "name": "over_budget",
            "daily_budget": 15.0,
            "costs_recorded": [8.0, 5.0, 3.5, 4.0],  # Total: 20.5
            "expected_utilization": 1.367,  # 20.5 / 15.0
            "expected_over_budget": True,
            "expected_alert_levels": [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
        }
    ]


@pytest.fixture
def cost_tracking_data():
    """Sample cost tracking data for different time periods"""
    base_date = datetime.now()
    return {
        "daily_costs": {
            (base_date - timedelta(days=i)).strftime("%Y-%m-%d"): 5.0 + (i * 0.5)
            for i in range(7)
        },
        "weekly_costs": {
            f"2024-W{week}": 35.0 + (week * 2.0)
            for week in range(1, 5)
        },
        "monthly_costs": {
            f"2024-{month:02d}": 150.0 + (month * 10.0)
            for month in range(1, 4)
        },
        "call_costs": {
            f"test_call_{i}": 0.05 + (i * 0.01)
            for i in range(1, 11)
        }
    }


@pytest.fixture
def mock_database_service():
    """Mock database service for cost control testing"""
    service = AsyncMock()
    service.is_ready.return_value = True
    service.record_cost.return_value = True
    return service


@pytest.fixture
def performance_test_scenarios():
    """Performance testing scenarios for cost control"""
    return [
        {
            "name": "concurrent_cost_recording",
            "concurrent_calls": 10,
            "costs_per_call": 5,
            "expected_max_time_seconds": 1.0
        },
        {
            "name": "large_budget_calculation",
            "num_cost_entries": 1000,
            "expected_max_time_seconds": 0.5
        },
        {
            "name": "tier_decision_batch",
            "num_decisions": 50,
            "expected_max_time_seconds": 2.0
        }
    ]