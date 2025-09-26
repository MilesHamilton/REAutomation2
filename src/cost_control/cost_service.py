import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

from ..config import settings
from ..database.service import database_service
from .cost_calculator import CostCalculator, ServiceCosts
from .budget_manager import BudgetManager, BudgetStatus, BudgetAlert
from .tier_decision import TierDecisionEngine, TierSwitchDecision
from ..voice.models import TTSProvider, TierSwitchEvent

logger = logging.getLogger(__name__)


class CostControlService:
    """Main cost control service integrating all cost management components"""

    def __init__(self):
        # Initialize components
        self.cost_calculator = CostCalculator()
        self.budget_manager = BudgetManager(
            daily_budget=settings.daily_budget,
            cost_per_call_limit=settings.cost_per_call_limit
        )
        self.tier_decision_engine = TierDecisionEngine(
            qualification_threshold=settings.qualification_threshold
        )

        # Callback handlers
        self._on_budget_alert: Optional[Callable[[BudgetAlert], None]] = None
        self._on_tier_switch_recommended: Optional[Callable[[str, TierSwitchDecision], None]] = None

        # Service state
        self._initialized = False

        logger.info("Cost Control Service initialized")

    async def initialize(self) -> bool:
        """Initialize cost control service"""
        try:
            # Ensure database service is available for cost persistence
            if not await database_service.is_ready():
                logger.warning("Database service not ready for cost control")

            self._initialized = True
            logger.info("Cost Control Service initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Cost Control Service initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._initialized

    async def start_call_cost_tracking(
        self,
        call_id: str,
        estimated_duration_minutes: float = 5.0
    ) -> Dict[str, Any]:
        """Start cost tracking for a new call"""
        try:
            # Initialize cost tracking
            self.cost_calculator.initialize_call_costs(call_id)

            # Check if call should be blocked due to budget
            estimated_cost = self._estimate_call_cost(estimated_duration_minutes)
            block_decision = self.budget_manager.should_block_call(estimated_cost)

            if block_decision["should_block"]:
                logger.warning(f"Call {call_id} blocked: {block_decision['message']}")
                return {
                    "allowed": False,
                    "reason": block_decision["reason"],
                    "message": block_decision["message"],
                    "estimated_cost": estimated_cost,
                    "budget_status": self.budget_manager.get_budget_status()
                }

            logger.info(f"Call {call_id} approved for cost tracking, estimated: ${estimated_cost:.4f}")

            return {
                "allowed": True,
                "call_id": call_id,
                "estimated_cost": estimated_cost,
                "budget_status": self.budget_manager.get_budget_status()
            }

        except Exception as e:
            logger.error(f"Error starting call cost tracking for {call_id}: {e}")
            return {
                "allowed": False,
                "reason": "tracking_error",
                "message": f"Cost tracking initialization failed: {e}"
            }

    async def record_llm_cost(
        self,
        call_id: str,
        provider: str = "ollama",
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> float:
        """Record LLM usage cost"""
        try:
            cost = self.cost_calculator.calculate_llm_cost(call_id, provider, input_tokens, output_tokens)

            # Record in budget manager and check for alerts
            alerts = self.budget_manager.record_cost(call_id, cost)
            await self._handle_budget_alerts(alerts)

            # Persist to database
            if await database_service.is_ready():
                await database_service.record_cost(
                    cost_type="llm",
                    cost_amount=cost,
                    call_id=call_id,
                    units_consumed=input_tokens + output_tokens,
                    unit_type="tokens",
                    service_provider=provider
                )

            return cost

        except Exception as e:
            logger.error(f"Error recording LLM cost for call {call_id}: {e}")
            return 0.0

    async def record_tts_cost(
        self,
        call_id: str,
        provider: str,
        character_count: int,
        tier: Optional[str] = None
    ) -> float:
        """Record TTS synthesis cost"""
        try:
            cost = self.cost_calculator.calculate_tts_cost(call_id, provider, character_count)

            # Record in budget manager and check for alerts
            alerts = self.budget_manager.record_cost(call_id, cost)
            await self._handle_budget_alerts(alerts)

            # Persist to database
            if await database_service.is_ready():
                await database_service.record_cost(
                    cost_type="tts",
                    cost_amount=cost,
                    call_id=call_id,
                    units_consumed=character_count,
                    unit_type="characters",
                    service_provider=provider,
                    tier=tier
                )

            return cost

        except Exception as e:
            logger.error(f"Error recording TTS cost for call {call_id}: {e}")
            return 0.0

    async def record_stt_cost(
        self,
        call_id: str,
        provider: str = "whisper-local",
        duration_minutes: float = 0.0
    ) -> float:
        """Record STT transcription cost"""
        try:
            cost = self.cost_calculator.calculate_stt_cost(call_id, provider, duration_minutes)

            # Record in budget manager and check for alerts
            alerts = self.budget_manager.record_cost(call_id, cost)
            await self._handle_budget_alerts(alerts)

            # Persist to database
            if await database_service.is_ready():
                await database_service.record_cost(
                    cost_type="stt",
                    cost_amount=cost,
                    call_id=call_id,
                    units_consumed=duration_minutes,
                    unit_type="minutes",
                    service_provider=provider
                )

            return cost

        except Exception as e:
            logger.error(f"Error recording STT cost for call {call_id}: {e}")
            return 0.0

    async def record_twilio_cost(
        self,
        call_id: str,
        duration_minutes: float,
        sms_count: int = 0
    ) -> float:
        """Record Twilio communication cost"""
        try:
            cost = self.cost_calculator.calculate_twilio_cost(call_id, duration_minutes, sms_count)

            # Record in budget manager and check for alerts
            alerts = self.budget_manager.record_cost(call_id, cost)
            await self._handle_budget_alerts(alerts)

            # Persist to database
            if await database_service.is_ready():
                await database_service.record_cost(
                    cost_type="twilio",
                    cost_amount=cost,
                    call_id=call_id,
                    units_consumed=duration_minutes,
                    unit_type="minutes",
                    service_provider="twilio"
                )

            return cost

        except Exception as e:
            logger.error(f"Error recording Twilio cost for call {call_id}: {e}")
            return 0.0

    async def evaluate_tier_switch(
        self,
        call_id: str,
        current_tier: TTSProvider,
        qualification_score: Optional[float] = None,
        conversation_length: int = 0,
        lead_data: Optional[Dict[str, Any]] = None
    ) -> TierSwitchDecision:
        """Evaluate whether to switch TTS tier"""
        try:
            # Get current budget status
            budget_status = self.budget_manager.get_budget_status()
            call_cost = self.cost_calculator.get_total_call_cost(call_id)

            # Get tier decision
            decision = self.tier_decision_engine.should_escalate_tier(
                call_id=call_id,
                current_tier=current_tier,
                qualification_score=qualification_score,
                conversation_length=conversation_length,
                budget_utilization=budget_status.daily_utilization,
                call_cost_so_far=call_cost,
                estimated_remaining_minutes=3.0,  # Conservative estimate
                lead_data=lead_data
            )

            # If decision recommends switching, check budget constraints
            if decision.should_switch and decision.cost_impact > 0:
                # Ensure switch won't violate budget
                projected_cost = call_cost + decision.cost_impact
                if projected_cost > settings.cost_per_call_limit:
                    logger.warning(f"Call {call_id}: Tier switch blocked by per-call limit")
                    decision.should_switch = False
                    decision.reason = "blocked_by_call_limit"
                    decision.confidence *= 0.5

            # Trigger callback if switch is recommended
            if decision.should_switch and self._on_tier_switch_recommended:
                await self._on_tier_switch_recommended(call_id, decision)

            logger.debug(f"Call {call_id}: Tier decision - {decision.reason} (confidence: {decision.confidence:.2f})")

            return decision

        except Exception as e:
            logger.error(f"Error evaluating tier switch for call {call_id}: {e}")
            return TierSwitchDecision(
                should_switch=False,
                target_tier=current_tier,
                current_tier=current_tier,
                reason="evaluation_error",
                confidence=0.0,
                cost_impact=0.0
            )

    async def finalize_call_costs(self, call_id: str) -> Dict[str, float]:
        """Finalize costs for a completed call"""
        try:
            # Get final cost breakdown
            cost_breakdown = self.cost_calculator.finalize_call_costs(call_id)

            if not cost_breakdown:
                logger.warning(f"No cost data found for finalizing call {call_id}")
                return {}

            total_cost = cost_breakdown.get("total", 0.0)

            # Update final budget tracking
            alerts = self.budget_manager.record_cost(call_id, 0.0)  # Final reconciliation
            await self._handle_budget_alerts(alerts)

            logger.info(f"Call {call_id} costs finalized: ${total_cost:.4f}")

            return cost_breakdown

        except Exception as e:
            logger.error(f"Error finalizing costs for call {call_id}: {e}")
            return {}

    def get_budget_status(self) -> BudgetStatus:
        """Get current budget status"""
        return self.budget_manager.get_budget_status()

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary across all services"""
        try:
            # Get calculator summary
            calc_summary = self.cost_calculator.get_cost_summary()

            # Get budget status
            budget_status = self.budget_manager.get_budget_status()

            # Get trends
            cost_trends = self.budget_manager.get_cost_trends()

            return {
                "calculator_summary": calc_summary,
                "budget_status": budget_status.__dict__,
                "cost_trends": cost_trends,
                "service_status": "healthy" if self._initialized else "not_ready"
            }

        except Exception as e:
            logger.error(f"Error getting cost summary: {e}")
            return {"error": str(e)}

    async def check_call_approval(
        self,
        estimated_duration_minutes: float = 5.0,
        estimated_messages: int = 10
    ) -> Dict[str, Any]:
        """Check if a new call should be approved based on budget"""
        try:
            estimated_cost = self._estimate_call_cost(estimated_duration_minutes, estimated_messages)
            block_decision = self.budget_manager.should_block_call(estimated_cost)

            return {
                "approved": not block_decision["should_block"],
                "estimated_cost": estimated_cost,
                "budget_status": self.get_budget_status(),
                "block_reason": block_decision.get("reason") if block_decision["should_block"] else None,
                "message": block_decision.get("message", "Call approved")
            }

        except Exception as e:
            logger.error(f"Error checking call approval: {e}")
            return {
                "approved": False,
                "estimated_cost": 0.0,
                "error": str(e)
            }

    def _estimate_call_cost(
        self,
        duration_minutes: float = 5.0,
        estimated_messages: int = 10,
        tier: str = "local"
    ) -> float:
        """Estimate total cost for a call"""
        try:
            # Twilio cost
            twilio_cost = duration_minutes * self.cost_calculator.costs.twilio_voice_cost_per_minute

            # LLM cost (using Ollama local)
            estimated_tokens = estimated_messages * 100  # Rough estimate
            llm_cost = estimated_tokens * self.cost_calculator.costs.ollama_local_cost_per_token

            # STT cost (using local Whisper)
            stt_cost = duration_minutes * self.cost_calculator.costs.whisper_local_cost_per_minute

            # TTS cost
            estimated_chars = estimated_messages * 100  # Rough estimate
            if tier == "premium":
                tts_cost = estimated_chars * self.cost_calculator.costs.elevenlabs_cost_per_char
            else:
                tts_cost = estimated_chars * self.cost_calculator.costs.local_piper_cost_per_char

            # Infrastructure cost
            infra_cost = self.cost_calculator.costs.infrastructure_cost_per_call

            return twilio_cost + llm_cost + stt_cost + tts_cost + infra_cost

        except Exception as e:
            logger.error(f"Error estimating call cost: {e}")
            return 0.05  # Conservative fallback estimate

    async def _handle_budget_alerts(self, alerts: List[BudgetAlert]):
        """Handle budget alerts"""
        for alert in alerts:
            logger.warning(f"Budget Alert: {alert.message}")

            # Trigger callback if configured
            if self._on_budget_alert:
                try:
                    await self._on_budget_alert(alert)
                except Exception as e:
                    logger.error(f"Error in budget alert callback: {e}")

    # Callback setters
    def on_budget_alert(self, callback: Callable[[BudgetAlert], None]):
        """Set budget alert callback"""
        self._on_budget_alert = callback

    def on_tier_switch_recommended(self, callback: Callable[[str, TierSwitchDecision], None]):
        """Set tier switch recommendation callback"""
        self._on_tier_switch_recommended = callback

    async def health_check(self) -> Dict[str, Any]:
        """Check cost control service health"""
        try:
            budget_status = self.get_budget_status()

            return {
                "service_status": "healthy" if self._initialized else "not_ready",
                "cost_calculator": {
                    "active_calls": len(self.cost_calculator.call_costs),
                    "total_tracked_cost": sum(
                        costs.get("total", 0) for costs in self.cost_calculator.call_costs.values()
                    )
                },
                "budget_manager": {
                    "daily_utilization": budget_status.daily_utilization,
                    "over_budget": budget_status.over_budget,
                    "recent_alerts": len(budget_status.alerts) if budget_status.alerts else 0
                },
                "tier_decision_engine": self.tier_decision_engine.get_decision_metrics()
            }

        except Exception as e:
            return {
                "service_status": "unhealthy",
                "error": str(e)
            }

    async def cleanup(self):
        """Clean up cost control service"""
        try:
            # Clean up old cost data
            self.cost_calculator.cleanup_old_calls()
            self._initialized = False
            logger.info("Cost Control Service cleanup complete")

        except Exception as e:
            logger.error(f"Error during cost control service cleanup: {e}")


# Global cost control service instance
cost_control_service = CostControlService()