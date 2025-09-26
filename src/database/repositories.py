import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_, desc, asc
from sqlalchemy.orm import selectinload

from .models import (
    CallRecord, ConversationHistory, TierSwitchHistory, ContactRecord,
    SystemMetrics, CostTracking, ScheduledCalls, get_daily_key, get_monthly_key
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def commit(self):
        """Commit current transaction"""
        await self.session.commit()

    async def rollback(self):
        """Rollback current transaction"""
        await self.session.rollback()


class CallRepository(BaseRepository):
    """Repository for call-related operations"""

    async def create_call(self, call_data: Dict[str, Any]) -> CallRecord:
        """Create a new call record"""
        call = CallRecord(**call_data)
        self.session.add(call)
        await self.session.flush()  # Get the ID without committing
        return call

    async def get_call_by_id(self, call_id: str) -> Optional[CallRecord]:
        """Get call by call_id"""
        result = await self.session.execute(
            select(CallRecord)
            .where(CallRecord.call_id == call_id)
            .options(
                selectinload(CallRecord.conversations),
                selectinload(CallRecord.tier_switches_history)
            )
        )
        return result.scalar_one_or_none()

    async def update_call_status(self, call_id: str, status: str, **kwargs) -> bool:
        """Update call status and related fields"""
        update_data = {"status": status, "updated_at": datetime.utcnow(), **kwargs}

        result = await self.session.execute(
            update(CallRecord)
            .where(CallRecord.call_id == call_id)
            .values(**update_data)
        )
        return result.rowcount > 0

    async def end_call(self, call_id: str, end_status: str, metrics: Dict[str, Any]) -> bool:
        """End a call and update all metrics"""
        end_time = datetime.utcnow()
        update_data = {
            "status": end_status,
            "ended_at": end_time,
            "updated_at": end_time,
            **metrics
        }

        result = await self.session.execute(
            update(CallRecord)
            .where(CallRecord.call_id == call_id)
            .values(**update_data)
        )
        return result.rowcount > 0

    async def get_active_calls(self) -> List[CallRecord]:
        """Get all currently active calls"""
        result = await self.session.execute(
            select(CallRecord)
            .where(CallRecord.status.in_(["ringing", "connected", "in_progress"]))
            .order_by(desc(CallRecord.created_at))
        )
        return result.scalars().all()

    async def get_calls_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        status_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[CallRecord]:
        """Get calls within date range"""
        query = select(CallRecord).where(
            and_(
                CallRecord.created_at >= start_date,
                CallRecord.created_at <= end_date
            )
        )

        if status_filter:
            query = query.where(CallRecord.status == status_filter)

        query = query.order_by(desc(CallRecord.created_at)).limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_call_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get call metrics summary for the last N days"""
        start_date = datetime.utcnow() - timedelta(days=days)

        # Get basic call counts and metrics
        result = await self.session.execute(
            select(
                func.count(CallRecord.id).label("total_calls"),
                func.count(CallRecord.id).filter(CallRecord.status == "completed").label("completed_calls"),
                func.count(CallRecord.id).filter(CallRecord.qualified == True).label("qualified_calls"),
                func.avg(CallRecord.duration_seconds).label("avg_duration"),
                func.avg(CallRecord.qualification_score).label("avg_qualification_score"),
                func.sum(CallRecord.total_cost).label("total_cost"),
                func.avg(CallRecord.total_cost).label("avg_cost_per_call"),
            )
            .where(CallRecord.created_at >= start_date)
        )

        metrics = result.first()

        # Get tier distribution
        tier_result = await self.session.execute(
            select(
                CallRecord.final_tier.label("tier"),
                func.count(CallRecord.id).label("count")
            )
            .where(CallRecord.created_at >= start_date)
            .group_by(CallRecord.final_tier)
        )

        tier_distribution = {row.tier: row.count for row in tier_result.fetchall()}

        return {
            "total_calls": metrics.total_calls or 0,
            "completed_calls": metrics.completed_calls or 0,
            "qualified_calls": metrics.qualified_calls or 0,
            "qualification_rate": (metrics.qualified_calls or 0) / max(metrics.completed_calls or 1, 1),
            "avg_duration_seconds": float(metrics.avg_duration or 0),
            "avg_qualification_score": float(metrics.avg_qualification_score or 0),
            "total_cost": float(metrics.total_cost or 0),
            "avg_cost_per_call": float(metrics.avg_cost_per_call or 0),
            "tier_distribution": tier_distribution,
        }


class ConversationRepository(BaseRepository):
    """Repository for conversation history operations"""

    async def add_message(
        self,
        call_id: str,
        role: str,
        content: str,
        message_order: int,
        **kwargs
    ) -> ConversationHistory:
        """Add a message to conversation history"""
        message = ConversationHistory(
            call_id=call_id,
            role=role,
            content=content,
            message_order=message_order,
            **kwargs
        )
        self.session.add(message)
        await self.session.flush()
        return message

    async def get_conversation_history(
        self,
        call_id: str,
        limit: Optional[int] = None
    ) -> List[ConversationHistory]:
        """Get conversation history for a call"""
        query = select(ConversationHistory).where(
            ConversationHistory.call_id == call_id
        ).order_by(ConversationHistory.message_order)

        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_conversation_summary(self, call_id: str) -> Dict[str, Any]:
        """Get conversation summary statistics"""
        result = await self.session.execute(
            select(
                func.count(ConversationHistory.id).label("total_messages"),
                func.count(ConversationHistory.id).filter(
                    ConversationHistory.role == "user"
                ).label("user_messages"),
                func.count(ConversationHistory.id).filter(
                    ConversationHistory.role == "assistant"
                ).label("assistant_messages"),
                func.avg(ConversationHistory.processing_time_ms).label("avg_processing_time"),
                func.sum(ConversationHistory.processing_cost).label("total_processing_cost"),
            )
            .where(ConversationHistory.call_id == call_id)
        )

        summary = result.first()
        return {
            "total_messages": summary.total_messages or 0,
            "user_messages": summary.user_messages or 0,
            "assistant_messages": summary.assistant_messages or 0,
            "avg_processing_time_ms": float(summary.avg_processing_time or 0),
            "total_processing_cost": float(summary.total_processing_cost or 0),
        }


class ContactRepository(BaseRepository):
    """Repository for contact management"""

    async def create_or_update_contact(self, phone_number: str, contact_data: Dict[str, Any]) -> ContactRecord:
        """Create new contact or update existing one"""
        # Try to get existing contact
        result = await self.session.execute(
            select(ContactRecord).where(ContactRecord.phone_number == phone_number)
        )
        contact = result.scalar_one_or_none()

        if contact:
            # Update existing contact
            for key, value in contact_data.items():
                if hasattr(contact, key):
                    setattr(contact, key, value)
            contact.updated_at = datetime.utcnow()
        else:
            # Create new contact
            contact_data["phone_number"] = phone_number
            contact = ContactRecord(**contact_data)
            self.session.add(contact)

        await self.session.flush()
        return contact

    async def get_contact_by_phone(self, phone_number: str) -> Optional[ContactRecord]:
        """Get contact by phone number"""
        result = await self.session.execute(
            select(ContactRecord).where(ContactRecord.phone_number == phone_number)
        )
        return result.scalar_one_or_none()

    async def update_call_stats(self, phone_number: str, qualified: bool = False):
        """Update call statistics for a contact"""
        update_data = {
            "total_calls": ContactRecord.total_calls + 1,
            "last_contacted": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        if qualified:
            update_data["qualified_calls"] = ContactRecord.qualified_calls + 1

        await self.session.execute(
            update(ContactRecord)
            .where(ContactRecord.phone_number == phone_number)
            .values(**update_data)
        )

    async def get_contacts_by_campaign(self, campaign_id: str, limit: int = 100) -> List[ContactRecord]:
        """Get contacts by campaign ID"""
        result = await self.session.execute(
            select(ContactRecord)
            .where(ContactRecord.campaign_id == campaign_id)
            .order_by(desc(ContactRecord.created_at))
            .limit(limit)
        )
        return result.scalars().all()


class CostTrackingRepository(BaseRepository):
    """Repository for cost tracking and budget management"""

    async def record_cost(
        self,
        cost_type: str,
        cost_amount: float,
        call_id: Optional[str] = None,
        **kwargs
    ) -> CostTracking:
        """Record a cost entry"""
        cost_entry = CostTracking(
            cost_type=cost_type,
            cost_amount=cost_amount,
            call_id=call_id,
            daily_date=get_daily_key(),
            monthly_period=get_monthly_key(),
            **kwargs
        )
        self.session.add(cost_entry)
        await self.session.flush()
        return cost_entry

    async def get_daily_costs(self, date: Optional[datetime] = None) -> Dict[str, float]:
        """Get costs for a specific day"""
        date_key = get_daily_key(date)

        result = await self.session.execute(
            select(
                CostTracking.cost_type,
                func.sum(CostTracking.cost_amount).label("total_cost")
            )
            .where(CostTracking.daily_date == date_key)
            .group_by(CostTracking.cost_type)
        )

        return {row.cost_type: float(row.total_cost) for row in result.fetchall()}

    async def get_monthly_costs(self, date: Optional[datetime] = None) -> Dict[str, float]:
        """Get costs for a specific month"""
        month_key = get_monthly_key(date)

        result = await self.session.execute(
            select(
                CostTracking.cost_type,
                func.sum(CostTracking.cost_amount).label("total_cost")
            )
            .where(CostTracking.monthly_period == month_key)
            .group_by(CostTracking.cost_type)
        )

        return {row.cost_type: float(row.total_cost) for row in result.fetchall()}

    async def get_call_cost_breakdown(self, call_id: str) -> Dict[str, float]:
        """Get cost breakdown for a specific call"""
        result = await self.session.execute(
            select(
                CostTracking.cost_type,
                func.sum(CostTracking.cost_amount).label("total_cost")
            )
            .where(CostTracking.call_id == call_id)
            .group_by(CostTracking.cost_type)
        )

        return {row.cost_type: float(row.total_cost) for row in result.fetchall()}

    async def check_budget_status(self, daily_budget: float) -> Dict[str, Any]:
        """Check current budget status"""
        today_costs = await self.get_daily_costs()
        total_today = sum(today_costs.values())

        return {
            "daily_budget": daily_budget,
            "spent_today": total_today,
            "remaining_budget": daily_budget - total_today,
            "budget_utilization": total_today / daily_budget if daily_budget > 0 else 0,
            "over_budget": total_today > daily_budget,
            "cost_breakdown": today_costs
        }


class ScheduledCallRepository(BaseRepository):
    """Repository for scheduled call management"""

    async def create_scheduled_call(self, call_data: Dict[str, Any]) -> ScheduledCalls:
        """Create a scheduled call entry"""
        scheduled_call = ScheduledCalls(**call_data)
        self.session.add(scheduled_call)
        await self.session.flush()
        return scheduled_call

    async def get_pending_calls(self, limit: int = 50) -> List[ScheduledCalls]:
        """Get calls scheduled for now or earlier"""
        current_time = datetime.utcnow()

        result = await self.session.execute(
            select(ScheduledCalls)
            .where(
                and_(
                    ScheduledCalls.status == "scheduled",
                    ScheduledCalls.scheduled_for <= current_time
                )
            )
            .order_by(ScheduledCalls.priority.desc(), ScheduledCalls.scheduled_for.asc())
            .limit(limit)
        )
        return result.scalars().all()

    async def update_call_status(
        self,
        call_id: int,
        status: str,
        completion_status: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Update scheduled call status"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow(),
            **kwargs
        }

        if completion_status:
            update_data["completion_status"] = completion_status

        if status == "completed":
            update_data["completed_at"] = datetime.utcnow()

        result = await self.session.execute(
            update(ScheduledCalls)
            .where(ScheduledCalls.id == call_id)
            .values(**update_data)
        )
        return result.rowcount > 0

    async def schedule_retry(self, call_id: int, retry_time: datetime, reason: str) -> bool:
        """Schedule a call for retry"""
        result = await self.session.execute(
            update(ScheduledCalls)
            .where(ScheduledCalls.id == call_id)
            .values(
                status="scheduled",
                next_retry_at=retry_time,
                retry_reason=reason,
                attempts=ScheduledCalls.attempts + 1,
                updated_at=datetime.utcnow()
            )
        )
        return result.rowcount > 0


class MetricsRepository(BaseRepository):
    """Repository for system metrics"""

    async def record_metric(
        self,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        **kwargs
    ) -> SystemMetrics:
        """Record a system metric"""
        metric = SystemMetrics(
            metric_type=metric_type,
            metric_name=metric_name,
            metric_value=metric_value,
            **kwargs
        )
        self.session.add(metric)
        await self.session.flush()
        return metric

    async def get_metrics_by_type(
        self,
        metric_type: str,
        hours: int = 24,
        limit: int = 1000
    ) -> List[SystemMetrics]:
        """Get metrics by type for the last N hours"""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        result = await self.session.execute(
            select(SystemMetrics)
            .where(
                and_(
                    SystemMetrics.metric_type == metric_type,
                    SystemMetrics.recorded_at >= start_time
                )
            )
            .order_by(desc(SystemMetrics.recorded_at))
            .limit(limit)
        )
        return result.scalars().all()

    async def get_aggregated_metrics(
        self,
        metric_name: str,
        hours: int = 24
    ) -> Dict[str, float]:
        """Get aggregated metrics (avg, min, max) for a metric"""
        start_time = datetime.utcnow() - timedelta(hours=hours)

        result = await self.session.execute(
            select(
                func.avg(SystemMetrics.metric_value).label("avg_value"),
                func.min(SystemMetrics.metric_value).label("min_value"),
                func.max(SystemMetrics.metric_value).label("max_value"),
                func.count(SystemMetrics.id).label("count")
            )
            .where(
                and_(
                    SystemMetrics.metric_name == metric_name,
                    SystemMetrics.recorded_at >= start_time
                )
            )
        )

        metrics = result.first()
        return {
            "avg_value": float(metrics.avg_value or 0),
            "min_value": float(metrics.min_value or 0),
            "max_value": float(metrics.max_value or 0),
            "count": metrics.count or 0
        }