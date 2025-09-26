import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .connection import db_manager
from .repositories import (
    CallRepository, ConversationRepository, ContactRepository,
    CostTrackingRepository, MetricsRepository
)
from .models import create_call_id, CallRecord
from ..voice.models import CallSession, VoiceMetrics, TierSwitchEvent

logger = logging.getLogger(__name__)


class DatabaseService:
    """High-level database service for voice pipeline integration"""

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize database service"""
        try:
            success = await db_manager.initialize()
            if success:
                # Create tables if they don't exist
                await db_manager.create_tables()
                self._initialized = True
                logger.info("Database service initialized successfully")
            return success
        except Exception as e:
            logger.error(f"Database service initialization failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if database service is ready"""
        return self._initialized and db_manager.is_initialized()

    async def create_call_record(
        self,
        phone_number: str,
        lead_data: Optional[Dict[str, Any]] = None,
        initial_tier: str = "local_piper"
    ) -> str:
        """Create a new call record and return call_id"""
        try:
            call_id = create_call_id()

            async with db_manager.get_session() as session:
                call_repo = CallRepository(session)

                call_data = {
                    "call_id": call_id,
                    "phone_number": phone_number,
                    "status": "initiated",
                    "initial_tier": initial_tier,
                    "final_tier": initial_tier,
                    "lead_data": lead_data or {},
                    "total_cost": 0.0,
                    "error_occurred": False,
                }

                await call_repo.create_call(call_data)

                # Also create/update contact record
                contact_repo = ContactRepository(session)
                contact_data = lead_data or {}
                await contact_repo.create_or_update_contact(phone_number, contact_data)

                logger.info(f"Created call record: {call_id} for {phone_number}")
                return call_id

        except Exception as e:
            logger.error(f"Failed to create call record: {e}")
            raise

    async def update_call_status(
        self,
        call_id: str,
        status: str,
        metrics: Optional[VoiceMetrics] = None,
        **kwargs
    ) -> bool:
        """Update call status and metrics"""
        try:
            async with db_manager.get_session() as session:
                call_repo = CallRepository(session)

                update_data = kwargs.copy()

                # Add metrics if provided
                if metrics:
                    update_data.update({
                        "duration_seconds": metrics.total_audio_duration_ms / 1000.0 if metrics.total_audio_duration_ms else None,
                        "qualification_score": metrics.qualification_score,
                        "qualified": metrics.qualification_score >= 0.8 if metrics.qualification_score else False,
                        "tier_switches": metrics.tier_switches,
                        "total_cost": metrics.cost,
                        "llm_latency_ms": metrics.llm_latency_ms,
                        "tts_latency_ms": metrics.tts_latency_ms,
                        "stt_latency_ms": metrics.stt_latency_ms,
                    })

                success = await call_repo.update_call_status(call_id, status, **update_data)

                if success:
                    logger.debug(f"Updated call {call_id} status to {status}")
                else:
                    logger.warning(f"Failed to update call {call_id} status")

                return success

        except Exception as e:
            logger.error(f"Failed to update call status for {call_id}: {e}")
            return False

    async def end_call(
        self,
        call_id: str,
        end_status: str,
        session: CallSession
    ) -> bool:
        """End a call and store final metrics"""
        try:
            async with db_manager.get_session() as db_session:
                call_repo = CallRepository(db_session)
                contact_repo = ContactRepository(db_session)

                # Prepare final metrics
                metrics = {}
                if session.metrics:
                    metrics = {
                        "duration_seconds": session.metrics.total_audio_duration_ms / 1000.0 if session.metrics.total_audio_duration_ms else None,
                        "qualification_score": session.metrics.qualification_score,
                        "qualified": session.metrics.qualification_score >= 0.8 if session.metrics.qualification_score else False,
                        "tier_switches": session.metrics.tier_switches,
                        "total_cost": session.metrics.cost,
                        "final_tier": session.current_tier.value if session.current_tier else session.tts_config.provider.value,
                        "llm_latency_ms": session.metrics.llm_latency_ms,
                        "tts_latency_ms": session.metrics.tts_latency_ms,
                        "stt_latency_ms": session.metrics.stt_latency_ms,
                    }

                # Add error info if applicable
                if session.error_message:
                    metrics.update({
                        "error_occurred": True,
                        "error_message": session.error_message
                    })

                success = await call_repo.end_call(call_id, end_status, metrics)

                # Update contact statistics
                if success and session.phone_number:
                    qualified = metrics.get("qualified", False)
                    await contact_repo.update_call_stats(session.phone_number, qualified)

                logger.info(f"Ended call {call_id} with status {end_status}")
                return success

        except Exception as e:
            logger.error(f"Failed to end call {call_id}: {e}")
            return False

    async def record_conversation_message(
        self,
        call_id: str,
        role: str,
        content: str,
        message_order: int,
        processing_time_ms: Optional[float] = None,
        processing_cost: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Record a conversation message"""
        try:
            async with db_manager.get_session() as session:
                conv_repo = ConversationRepository(session)

                await conv_repo.add_message(
                    call_id=call_id,
                    role=role,
                    content=content,
                    message_order=message_order,
                    processing_time_ms=processing_time_ms,
                    processing_cost=processing_cost,
                    **kwargs
                )

                logger.debug(f"Recorded conversation message for call {call_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to record conversation message for {call_id}: {e}")
            return False

    async def record_tier_switch(self, switch_event: TierSwitchEvent) -> bool:
        """Record a tier switch event"""
        try:
            async with db_manager.get_session() as session:
                # Use raw SQL to insert tier switch record
                await session.execute(
                    """
                    INSERT INTO tier_switches (call_id, from_tier, to_tier, trigger, qualification_score_at_switch)
                    VALUES (:call_id, :from_tier, :to_tier, :trigger, :qualification_score)
                    """,
                    {
                        "call_id": switch_event.call_id,
                        "from_tier": switch_event.from_tier.value,
                        "to_tier": switch_event.to_tier.value,
                        "trigger": switch_event.trigger,
                        "qualification_score": switch_event.qualification_score
                    }
                )

                logger.info(f"Recorded tier switch for call {switch_event.call_id}: {switch_event.from_tier} -> {switch_event.to_tier}")
                return True

        except Exception as e:
            logger.error(f"Failed to record tier switch: {e}")
            return False

    async def record_cost(
        self,
        cost_type: str,
        cost_amount: float,
        call_id: Optional[str] = None,
        units_consumed: Optional[float] = None,
        unit_type: Optional[str] = None,
        service_provider: Optional[str] = None,
        tier: Optional[str] = None
    ) -> bool:
        """Record a cost entry"""
        try:
            async with db_manager.get_session() as session:
                cost_repo = CostTrackingRepository(session)

                await cost_repo.record_cost(
                    cost_type=cost_type,
                    cost_amount=cost_amount,
                    call_id=call_id,
                    units_consumed=units_consumed,
                    unit_type=unit_type,
                    service_provider=service_provider,
                    tier=tier
                )

                logger.debug(f"Recorded cost: {cost_type} = ${cost_amount:.4f} for call {call_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to record cost: {e}")
            return False

    async def get_call_record(self, call_id: str) -> Optional[CallRecord]:
        """Get call record by ID"""
        try:
            async with db_manager.get_session() as session:
                call_repo = CallRepository(session)
                return await call_repo.get_call_by_id(call_id)

        except Exception as e:
            logger.error(f"Failed to get call record {call_id}: {e}")
            return None

    async def get_daily_cost_summary(self, date: Optional[datetime] = None) -> Dict[str, float]:
        """Get daily cost summary"""
        try:
            async with db_manager.get_session() as session:
                cost_repo = CostTrackingRepository(session)
                return await cost_repo.get_daily_costs(date)

        except Exception as e:
            logger.error(f"Failed to get daily cost summary: {e}")
            return {}

    async def check_budget_status(self, daily_budget: float) -> Dict[str, Any]:
        """Check current budget status"""
        try:
            async with db_manager.get_session() as session:
                cost_repo = CostTrackingRepository(session)
                return await cost_repo.check_budget_status(daily_budget)

        except Exception as e:
            logger.error(f"Failed to check budget status: {e}")
            return {"error": str(e)}

    async def get_call_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get call metrics summary"""
        try:
            async with db_manager.get_session() as session:
                call_repo = CallRepository(session)
                return await call_repo.get_call_metrics_summary(days)

        except Exception as e:
            logger.error(f"Failed to get call metrics summary: {e}")
            return {}

    async def record_system_metric(
        self,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        **kwargs
    ) -> bool:
        """Record a system metric"""
        try:
            async with db_manager.get_session() as session:
                metrics_repo = MetricsRepository(session)

                await metrics_repo.record_metric(
                    metric_type=metric_type,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    **kwargs
                )

                return True

        except Exception as e:
            logger.error(f"Failed to record system metric: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Check database service health"""
        try:
            db_health = await db_manager.health_check()
            return {
                "database_service": {
                    "status": "healthy" if self._initialized else "not_initialized",
                    "initialized": self._initialized
                },
                "database_connection": db_health
            }

        except Exception as e:
            return {
                "database_service": {
                    "status": "unhealthy",
                    "error": str(e),
                    "initialized": self._initialized
                }
            }

    async def cleanup(self):
        """Clean up database service"""
        try:
            await db_manager.cleanup()
            self._initialized = False
            logger.info("Database service cleanup complete")

        except Exception as e:
            logger.error(f"Database service cleanup error: {e}")


# Global database service instance
database_service = DatabaseService()
