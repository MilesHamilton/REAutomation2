"""Tests for tier decision engine"""

import pytest
from unittest.mock import MagicMock

from src.cost_control.tier_decision import TierDecisionEngine, TierSwitchDecision
from src.voice.models import TTSProvider


class TestTierDecisionEngine:
    """Test TierDecisionEngine functionality"""

    def test_initialize_tier_decision_engine(self, tier_decision_engine):
        """Test tier decision engine initialization"""
        assert tier_decision_engine.qualification_threshold == 0.7
        assert tier_decision_engine.budget_tier_switch_threshold == 0.8
        assert tier_decision_engine.cost_savings_threshold == 0.02
        assert tier_decision_engine.min_conversation_length == 3

    def test_high_qualification_escalation(self, tier_decision_engine, tier_switch_test_cases):
        """Test escalation for high qualification score"""
        test_case = next(tc for tc in tier_switch_test_cases if tc["name"] == "high_qualification_escalation")

        decision = tier_decision_engine.should_switch_tier(
            current_tier=test_case["current_tier"],
            qualification_score=test_case["qualification_score"],
            conversation_length=test_case["conversation_length"],
            budget_utilization=test_case["budget_utilization"],
            call_cost_so_far=test_case["call_cost_so_far"],
            lead_data=test_case["lead_data"]
        )

        assert decision.should_switch == test_case["expected_switch"]
        assert decision.target_tier == test_case["expected_target"]
        assert decision.confidence >= test_case["expected_confidence_min"]
        assert decision.reasoning is not None

    def test_low_qualification_no_switch(self, tier_decision_engine, tier_switch_test_cases):
        """Test no switch for low qualification score"""
        test_case = next(tc for tc in tier_switch_test_cases if tc["name"] == "low_qualification_no_switch")

        decision = tier_decision_engine.should_switch_tier(
            current_tier=test_case["current_tier"],
            qualification_score=test_case["qualification_score"],
            conversation_length=test_case["conversation_length"],
            budget_utilization=test_case["budget_utilization"],
            call_cost_so_far=test_case["call_cost_so_far"],
            lead_data=test_case["lead_data"]
        )

        assert decision.should_switch == test_case["expected_switch"]
        assert decision.target_tier == test_case["expected_target"]
        assert decision.confidence >= test_case["expected_confidence_min"]

    def test_budget_constraint_no_switch(self, tier_decision_engine, tier_switch_test_cases):
        """Test no switch due to budget constraints"""
        test_case = next(tc for tc in tier_switch_test_cases if tc["name"] == "budget_constraint_no_switch")

        decision = tier_decision_engine.should_switch_tier(
            current_tier=test_case["current_tier"],
            qualification_score=test_case["qualification_score"],
            conversation_length=test_case["conversation_length"],
            budget_utilization=test_case["budget_utilization"],
            call_cost_so_far=test_case["call_cost_so_far"],
            lead_data=test_case["lead_data"]
        )

        assert decision.should_switch == test_case["expected_switch"]
        assert decision.target_tier == test_case["expected_target"]
        assert "budget" in decision.reasoning.lower()

    def test_downgrade_from_premium(self, tier_decision_engine, tier_switch_test_cases):
        """Test downgrade from premium tier"""
        test_case = next(tc for tc in tier_switch_test_cases if tc["name"] == "downgrade_from_premium")

        decision = tier_decision_engine.should_switch_tier(
            current_tier=test_case["current_tier"],
            qualification_score=test_case["qualification_score"],
            conversation_length=test_case["conversation_length"],
            budget_utilization=test_case["budget_utilization"],
            call_cost_so_far=test_case["call_cost_so_far"],
            lead_data=test_case["lead_data"]
        )

        assert decision.should_switch == test_case["expected_switch"]
        assert decision.target_tier == test_case["expected_target"]
        assert decision.confidence >= test_case["expected_confidence_min"]

    def test_short_conversation_no_switch(self, tier_decision_engine):
        """Test no switch for short conversations"""
        decision = tier_decision_engine.should_switch_tier(
            current_tier=TTSProvider.LOCAL_PIPER,
            qualification_score=0.9,
            conversation_length=1,  # Below min_conversation_length
            budget_utilization=0.3,
            call_cost_so_far=0.01,
            lead_data={}
        )

        assert decision.should_switch is False
        assert decision.target_tier == TTSProvider.LOCAL_PIPER
        assert "conversation too short" in decision.reasoning.lower()

    def test_calculate_tier_switch_confidence(self, tier_decision_engine):
        """Test confidence calculation for tier switches"""
        # High confidence scenario
        confidence_high = tier_decision_engine.calculate_tier_switch_confidence(
            qualification_score=0.9,
            conversation_length=10,
            budget_utilization=0.3,
            lead_indicators={"company_size": "enterprise", "decision_maker": True}
        )
        assert confidence_high > 0.8

        # Low confidence scenario
        confidence_low = tier_decision_engine.calculate_tier_switch_confidence(
            qualification_score=0.4,
            conversation_length=2,
            budget_utilization=0.9,
            lead_indicators={}
        )
        assert confidence_low < 0.5

    def test_estimate_cost_impact(self, tier_decision_engine):
        """Test cost impact estimation"""
        # Escalation cost impact
        cost_impact_up = tier_decision_engine.estimate_cost_impact(
            from_tier=TTSProvider.LOCAL_PIPER,
            to_tier=TTSProvider.ELEVENLABS,
            estimated_remaining_characters=500
        )
        assert cost_impact_up > 0  # Should cost more

        # Downgrade cost impact
        cost_impact_down = tier_decision_engine.estimate_cost_impact(
            from_tier=TTSProvider.ELEVENLABS,
            to_tier=TTSProvider.LOCAL_PIPER,
            estimated_remaining_characters=500
        )
        assert cost_impact_down < 0  # Should save money

    def test_get_tier_recommendation_factors(self, tier_decision_engine):
        """Test tier recommendation factor analysis"""
        factors = tier_decision_engine.get_tier_recommendation_factors(
            qualification_score=0.8,
            conversation_length=6,
            budget_utilization=0.4,
            call_cost_so_far=0.03,
            lead_data={"company_size": "enterprise", "timeline": "urgent"}
        )

        assert "qualification" in factors
        assert "conversation_length" in factors
        assert "budget_status" in factors
        assert "lead_indicators" in factors
        assert factors["qualification"]["score"] == 0.8
        assert factors["conversation_length"]["length"] == 6

    def test_is_escalation_justified(self, tier_decision_engine):
        """Test escalation justification logic"""
        # Justified escalation
        justified = tier_decision_engine.is_escalation_justified(
            qualification_score=0.85,
            conversation_length=8,
            lead_data={"company_size": "enterprise", "decision_maker": True}
        )
        assert justified is True

        # Not justified escalation
        not_justified = tier_decision_engine.is_escalation_justified(
            qualification_score=0.4,
            conversation_length=2,
            lead_data={}
        )
        assert not_justified is False

    def test_is_downgrade_necessary(self, tier_decision_engine):
        """Test downgrade necessity logic"""
        # Necessary downgrade
        necessary = tier_decision_engine.is_downgrade_necessary(
            qualification_score=0.2,
            conversation_length=12,
            budget_utilization=0.95
        )
        assert necessary is True

        # Not necessary downgrade
        not_necessary = tier_decision_engine.is_downgrade_necessary(
            qualification_score=0.8,
            conversation_length=5,
            budget_utilization=0.3
        )
        assert not_necessary is False

    def test_apply_lead_data_modifiers(self, tier_decision_engine):
        """Test lead data modifiers on qualification score"""
        base_score = 0.6

        # High-value lead modifiers
        modified_score_high = tier_decision_engine.apply_lead_data_modifiers(
            base_score,
            lead_data={
                "company_size": "enterprise",
                "decision_maker": True,
                "timeline": "urgent",
                "budget_confirmed": True
            }
        )
        assert modified_score_high > base_score

        # Low-value lead modifiers
        modified_score_low = tier_decision_engine.apply_lead_data_modifiers(
            base_score,
            lead_data={
                "company_size": "startup",
                "decision_maker": False,
                "timeline": "exploring"
            }
        )
        assert modified_score_low <= base_score

    def test_get_switch_history_impact(self, tier_decision_engine):
        """Test impact of switch history on decisions"""
        # First switch - no impact
        impact_first = tier_decision_engine.get_switch_history_impact([])
        assert impact_first == 0.0

        # Multiple recent switches - negative impact
        recent_switches = [
            {"timestamp": 1640995200, "from_tier": "local", "to_tier": "premium"},
            {"timestamp": 1640995800, "from_tier": "premium", "to_tier": "local"}
        ]
        impact_multiple = tier_decision_engine.get_switch_history_impact(recent_switches)
        assert impact_multiple < 0

    def test_tier_switch_decision_model(self):
        """Test TierSwitchDecision model"""
        decision = TierSwitchDecision(
            should_switch=True,
            target_tier=TTSProvider.ELEVENLABS,
            confidence=0.85,
            reasoning="High qualification score and enterprise lead",
            estimated_cost_impact=0.05,
            factors={
                "qualification": {"score": 0.9, "weight": 0.4},
                "budget": {"utilization": 0.3, "weight": 0.3}
            }
        )

        assert decision.should_switch is True
        assert decision.target_tier == TTSProvider.ELEVENLABS
        assert decision.confidence == 0.85
        assert decision.estimated_cost_impact == 0.05
        assert "qualification" in decision.factors

    def test_edge_case_same_tier(self, tier_decision_engine):
        """Test decision when already on target tier"""
        decision = tier_decision_engine.should_switch_tier(
            current_tier=TTSProvider.ELEVENLABS,
            qualification_score=0.9,
            conversation_length=5,
            budget_utilization=0.3,
            call_cost_so_far=0.08,
            lead_data={"company_size": "enterprise"}
        )

        # Should not switch to same tier
        assert decision.should_switch is False
        assert decision.target_tier == TTSProvider.ELEVENLABS

    def test_budget_emergency_override(self, tier_decision_engine):
        """Test emergency budget override"""
        decision = tier_decision_engine.should_switch_tier(
            current_tier=TTSProvider.ELEVENLABS,
            qualification_score=0.9,
            conversation_length=8,
            budget_utilization=0.98,  # Emergency level
            call_cost_so_far=0.45,
            lead_data={"company_size": "enterprise"}
        )

        # Should downgrade due to budget emergency
        assert decision.should_switch is True
        assert decision.target_tier == TTSProvider.LOCAL_PIPER
        assert "emergency" in decision.reasoning.lower()