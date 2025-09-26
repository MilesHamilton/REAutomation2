import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..voice.models import TTSProvider

logger = logging.getLogger(__name__)


@dataclass
class TierSwitchDecision:
    """Decision result for tier switching"""
    should_switch: bool
    target_tier: TTSProvider
    current_tier: TTSProvider
    reason: str
    confidence: float  # 0.0 to 1.0
    cost_impact: float  # Estimated cost difference
    qualification_score: Optional[float] = None
    budget_utilization: Optional[float] = None


class TierDecisionEngine:
    """Intelligent tier switching decision engine"""

    def __init__(
        self,
        qualification_threshold: float = 0.7,
        budget_tier_switch_threshold: float = 0.8,
        cost_savings_threshold: float = 0.02,
        min_conversation_length: int = 3
    ):
        self.qualification_threshold = qualification_threshold
        self.budget_tier_switch_threshold = budget_tier_switch_threshold
        self.cost_savings_threshold = cost_savings_threshold
        self.min_conversation_length = min_conversation_length

        # Cost multipliers for different tiers
        self.tier_cost_multipliers = {
            TTSProvider.LOCAL_PIPER: 1.0,  # Baseline (essentially free)
            TTSProvider.LOCAL_COQUI: 1.0,  # Baseline (essentially free)
            TTSProvider.ELEVENLABS: 25.0,  # ~25x more expensive
        }

        logger.info(f"Tier Decision Engine initialized: qualification_threshold={qualification_threshold}")

    def should_escalate_tier(
        self,
        call_id: str,
        current_tier: TTSProvider,
        qualification_score: Optional[float] = None,
        conversation_length: int = 0,
        budget_utilization: float = 0.0,
        call_cost_so_far: float = 0.0,
        estimated_remaining_minutes: float = 5.0,
        lead_data: Optional[Dict[str, Any]] = None
    ) -> TierSwitchDecision:
        """Determine if call should be escalated to a higher tier"""

        try:
            # If already on highest tier, no escalation needed
            if current_tier == TTSProvider.ELEVENLABS:
                return TierSwitchDecision(
                    should_switch=False,
                    target_tier=current_tier,
                    current_tier=current_tier,
                    reason="already_on_highest_tier",
                    confidence=1.0,
                    cost_impact=0.0
                )

            # Check minimum conversation length
            if conversation_length < self.min_conversation_length:
                return TierSwitchDecision(
                    should_switch=False,
                    target_tier=current_tier,
                    current_tier=current_tier,
                    reason="insufficient_conversation_length",
                    confidence=0.8,
                    cost_impact=0.0,
                    qualification_score=qualification_score
                )

            # Calculate decision factors
            decision_factors = self._calculate_decision_factors(
                qualification_score=qualification_score,
                budget_utilization=budget_utilization,
                call_cost_so_far=call_cost_so_far,
                estimated_remaining_minutes=estimated_remaining_minutes,
                lead_data=lead_data
            )

            # Make tier decision
            if decision_factors["should_escalate"]:
                target_tier = TTSProvider.ELEVENLABS
                cost_impact = self._estimate_cost_impact(
                    current_tier, target_tier, estimated_remaining_minutes
                )

                return TierSwitchDecision(
                    should_switch=True,
                    target_tier=target_tier,
                    current_tier=current_tier,
                    reason=decision_factors["primary_reason"],
                    confidence=decision_factors["confidence"],
                    cost_impact=cost_impact,
                    qualification_score=qualification_score,
                    budget_utilization=budget_utilization
                )
            else:
                return TierSwitchDecision(
                    should_switch=False,
                    target_tier=current_tier,
                    current_tier=current_tier,
                    reason=decision_factors["primary_reason"],
                    confidence=decision_factors["confidence"],
                    cost_impact=0.0,
                    qualification_score=qualification_score,
                    budget_utilization=budget_utilization
                )

        except Exception as e:
            logger.error(f"Error in tier escalation decision for call {call_id}: {e}")
            return TierSwitchDecision(
                should_switch=False,
                target_tier=current_tier,
                current_tier=current_tier,
                reason="decision_error",
                confidence=0.0,
                cost_impact=0.0
            )

    def should_downgrade_tier(
        self,
        call_id: str,
        current_tier: TTSProvider,
        qualification_score: Optional[float] = None,
        budget_utilization: float = 0.0,
        call_cost_so_far: float = 0.0,
        conversation_quality_score: float = 0.5
    ) -> TierSwitchDecision:
        """Determine if call should be downgraded to a lower tier"""

        try:
            # If already on lowest tier, no downgrade needed
            if current_tier in [TTSProvider.LOCAL_PIPER, TTSProvider.LOCAL_COQUI]:
                return TierSwitchDecision(
                    should_switch=False,
                    target_tier=current_tier,
                    current_tier=current_tier,
                    reason="already_on_lowest_tier",
                    confidence=1.0,
                    cost_impact=0.0
                )

            # Check budget pressure
            budget_pressure = budget_utilization > self.budget_tier_switch_threshold

            # Check qualification score
            low_qualification = qualification_score is not None and qualification_score < (self.qualification_threshold * 0.7)

            # Check conversation quality
            poor_conversation = conversation_quality_score < 0.3

            # Decision logic for downgrade
            should_downgrade = False
            primary_reason = "no_downgrade_needed"
            confidence = 0.5

            if budget_pressure and low_qualification:
                should_downgrade = True
                primary_reason = "budget_pressure_low_qualification"
                confidence = 0.9
            elif budget_pressure:
                should_downgrade = True
                primary_reason = "budget_pressure"
                confidence = 0.7
            elif poor_conversation and low_qualification:
                should_downgrade = True
                primary_reason = "poor_performance_metrics"
                confidence = 0.6

            if should_downgrade:
                target_tier = TTSProvider.LOCAL_PIPER
                cost_impact = self._estimate_cost_impact(current_tier, target_tier, 3.0)  # Remaining time estimate

                return TierSwitchDecision(
                    should_switch=True,
                    target_tier=target_tier,
                    current_tier=current_tier,
                    reason=primary_reason,
                    confidence=confidence,
                    cost_impact=cost_impact,
                    qualification_score=qualification_score,
                    budget_utilization=budget_utilization
                )
            else:
                return TierSwitchDecision(
                    should_switch=False,
                    target_tier=current_tier,
                    current_tier=current_tier,
                    reason=primary_reason,
                    confidence=confidence,
                    cost_impact=0.0,
                    qualification_score=qualification_score,
                    budget_utilization=budget_utilization
                )

        except Exception as e:
            logger.error(f"Error in tier downgrade decision for call {call_id}: {e}")
            return TierSwitchDecision(
                should_switch=False,
                target_tier=current_tier,
                current_tier=current_tier,
                reason="decision_error",
                confidence=0.0,
                cost_impact=0.0
            )

    def _calculate_decision_factors(
        self,
        qualification_score: Optional[float] = None,
        budget_utilization: float = 0.0,
        call_cost_so_far: float = 0.0,
        estimated_remaining_minutes: float = 5.0,
        lead_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate various factors that influence tier decision"""

        factors = {
            "should_escalate": False,
            "primary_reason": "no_escalation_needed",
            "confidence": 0.5,
            "factors": {}
        }

        try:
            # Factor 1: Qualification Score
            qualification_factor = 0.0
            if qualification_score is not None:
                if qualification_score >= self.qualification_threshold:
                    qualification_factor = min(1.0, (qualification_score - self.qualification_threshold) / (1.0 - self.qualification_threshold))
                factors["factors"]["qualification"] = qualification_factor

            # Factor 2: Budget Availability
            budget_factor = max(0.0, 1.0 - budget_utilization)
            factors["factors"]["budget_availability"] = budget_factor

            # Factor 3: Lead Value (from lead_data)
            lead_value_factor = 0.0
            if lead_data:
                # Check for high-value indicators
                if lead_data.get("company_size") == "enterprise":
                    lead_value_factor += 0.3
                if lead_data.get("annual_revenue", 0) > 1000000:
                    lead_value_factor += 0.3
                if lead_data.get("decision_maker", False):
                    lead_value_factor += 0.2
                if lead_data.get("urgent_timeline", False):
                    lead_value_factor += 0.2

                lead_value_factor = min(1.0, lead_value_factor)
            factors["factors"]["lead_value"] = lead_value_factor

            # Factor 4: Cost Impact Assessment
            estimated_cost_increase = self._estimate_cost_impact(
                TTSProvider.LOCAL_PIPER, TTSProvider.ELEVENLABS, estimated_remaining_minutes
            )

            # Only escalate if cost increase is reasonable
            cost_acceptable = estimated_cost_increase < self.cost_savings_threshold or budget_factor > 0.5
            factors["factors"]["cost_acceptable"] = 1.0 if cost_acceptable else 0.0

            # Combined decision logic
            weighted_score = (
                qualification_factor * 0.4 +
                budget_factor * 0.3 +
                lead_value_factor * 0.2 +
                (1.0 if cost_acceptable else 0.0) * 0.1
            )

            factors["factors"]["weighted_score"] = weighted_score

            # Decision thresholds
            if weighted_score > 0.7 and qualification_factor > 0.5:
                factors["should_escalate"] = True
                factors["primary_reason"] = "high_qualification_score"
                factors["confidence"] = min(0.95, weighted_score)
            elif weighted_score > 0.6 and lead_value_factor > 0.5:
                factors["should_escalate"] = True
                factors["primary_reason"] = "high_value_lead"
                factors["confidence"] = min(0.85, weighted_score)
            elif qualification_factor > 0.8 and budget_factor > 0.3:
                factors["should_escalate"] = True
                factors["primary_reason"] = "strong_qualification_with_budget"
                factors["confidence"] = min(0.90, weighted_score)
            else:
                factors["primary_reason"] = "insufficient_escalation_factors"
                factors["confidence"] = 1.0 - weighted_score

            return factors

        except Exception as e:
            logger.error(f"Error calculating decision factors: {e}")
            factors["primary_reason"] = "calculation_error"
            factors["confidence"] = 0.0
            return factors

    def _estimate_cost_impact(
        self,
        from_tier: TTSProvider,
        to_tier: TTSProvider,
        estimated_minutes: float
    ) -> float:
        """Estimate cost impact of switching tiers"""

        try:
            # Get cost multipliers
            from_multiplier = self.tier_cost_multipliers.get(from_tier, 1.0)
            to_multiplier = self.tier_cost_multipliers.get(to_tier, 1.0)

            # Estimate characters that will be synthesized
            # Rough estimate: 150 characters per minute of speech
            estimated_characters = estimated_minutes * 150

            # Cost per character (rough estimates)
            base_cost_per_char = 0.0002  # ElevenLabs pricing

            from_cost = estimated_characters * base_cost_per_char * from_multiplier
            to_cost = estimated_characters * base_cost_per_char * to_multiplier

            return to_cost - from_cost

        except Exception as e:
            logger.error(f"Error estimating cost impact: {e}")
            return 0.0

    def get_tier_recommendation(
        self,
        call_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get tier recommendation based on call context"""

        try:
            call_id = call_context.get("call_id", "unknown")
            current_tier = call_context.get("current_tier", TTSProvider.LOCAL_PIPER)

            # Get escalation decision
            escalation_decision = self.should_escalate_tier(
                call_id=call_id,
                current_tier=current_tier,
                qualification_score=call_context.get("qualification_score"),
                conversation_length=call_context.get("conversation_length", 0),
                budget_utilization=call_context.get("budget_utilization", 0.0),
                call_cost_so_far=call_context.get("call_cost_so_far", 0.0),
                estimated_remaining_minutes=call_context.get("estimated_remaining_minutes", 5.0),
                lead_data=call_context.get("lead_data")
            )

            # If escalation not recommended, check downgrade
            if not escalation_decision.should_switch and current_tier == TTSProvider.ELEVENLABS:
                downgrade_decision = self.should_downgrade_tier(
                    call_id=call_id,
                    current_tier=current_tier,
                    qualification_score=call_context.get("qualification_score"),
                    budget_utilization=call_context.get("budget_utilization", 0.0),
                    call_cost_so_far=call_context.get("call_cost_so_far", 0.0),
                    conversation_quality_score=call_context.get("conversation_quality_score", 0.5)
                )

                if downgrade_decision.should_switch:
                    escalation_decision = downgrade_decision

            return {
                "should_switch": escalation_decision.should_switch,
                "target_tier": escalation_decision.target_tier.value,
                "current_tier": escalation_decision.current_tier.value,
                "reason": escalation_decision.reason,
                "confidence": escalation_decision.confidence,
                "cost_impact": escalation_decision.cost_impact,
                "recommendation_type": "escalation" if escalation_decision.target_tier != current_tier and escalation_decision.should_switch else "maintain"
            }

        except Exception as e:
            logger.error(f"Error getting tier recommendation: {e}")
            return {
                "should_switch": False,
                "target_tier": "local_piper",
                "current_tier": "local_piper",
                "reason": "error",
                "confidence": 0.0,
                "cost_impact": 0.0,
                "error": str(e)
            }

    def update_thresholds(
        self,
        qualification_threshold: Optional[float] = None,
        budget_tier_switch_threshold: Optional[float] = None,
        cost_savings_threshold: Optional[float] = None,
        min_conversation_length: Optional[int] = None
    ):
        """Update decision thresholds"""

        if qualification_threshold is not None:
            self.qualification_threshold = qualification_threshold
            logger.info(f"Updated qualification threshold to {qualification_threshold}")

        if budget_tier_switch_threshold is not None:
            self.budget_tier_switch_threshold = budget_tier_switch_threshold
            logger.info(f"Updated budget tier switch threshold to {budget_tier_switch_threshold}")

        if cost_savings_threshold is not None:
            self.cost_savings_threshold = cost_savings_threshold
            logger.info(f"Updated cost savings threshold to {cost_savings_threshold}")

        if min_conversation_length is not None:
            self.min_conversation_length = min_conversation_length
            logger.info(f"Updated minimum conversation length to {min_conversation_length}")

    def get_decision_metrics(self) -> Dict[str, Any]:
        """Get decision engine metrics and configuration"""

        return {
            "thresholds": {
                "qualification_threshold": self.qualification_threshold,
                "budget_tier_switch_threshold": self.budget_tier_switch_threshold,
                "cost_savings_threshold": self.cost_savings_threshold,
                "min_conversation_length": self.min_conversation_length
            },
            "tier_cost_multipliers": {
                tier.value: multiplier for tier, multiplier in self.tier_cost_multipliers.items()
            }
        }