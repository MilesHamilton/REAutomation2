"""Tests for database models"""

import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import (
    CallRecord, ConversationHistory, TierSwitchHistory,
    ContactRecord, SystemMetrics, CostTracking, ScheduledCalls,
    create_call_id, get_daily_key, get_monthly_key
)


class TestCallRecord:
    """Test CallRecord model"""

    @pytest.mark.asyncio
    async def test_create_call_record(self, db_session: AsyncSession, sample_call_data):
        """Test creating a call record"""
        call = CallRecord(**sample_call_data)
        db_session.add(call)
        await db_session.commit()

        assert call.id is not None
        assert call.call_id == "test_call_123"
        assert call.phone_number == "+1234567890"
        assert call.status == "initiated"
        assert call.created_at is not None

    @pytest.mark.asyncio
    async def test_call_record_relationships(self, db_session: AsyncSession, sample_call_data):
        """Test call record relationships"""
        # Create call
        call = CallRecord(**sample_call_data)
        db_session.add(call)
        await db_session.flush()

        # Add conversation history
        conversation = ConversationHistory(
            call_id=call.call_id,
            message_order=1,
            role="assistant",
            content="Hello, test message"
        )
        db_session.add(conversation)

        # Add tier switch
        tier_switch = TierSwitchHistory(
            call_id=call.call_id,
            from_tier="local_piper",
            to_tier="elevenlabs",
            trigger="qualification"
        )
        db_session.add(tier_switch)

        await db_session.commit()

        # Test relationships
        assert len(call.conversations) == 1
        assert len(call.tier_switches_history) == 1
        assert call.conversations[0].content == "Hello, test message"
        assert call.tier_switches_history[0].to_tier == "elevenlabs"

    def test_call_id_generation(self):
        """Test call ID generation"""
        call_id = create_call_id()

        assert call_id.startswith("call_")
        assert len(call_id) > 20  # Should be long enough

        # Generate multiple IDs to ensure uniqueness
        call_ids = [create_call_id() for _ in range(10)]
        assert len(set(call_ids)) == 10  # All unique


class TestConversationHistory:
    """Test ConversationHistory model"""

    @pytest.mark.asyncio
    async def test_create_conversation_history(self, db_session: AsyncSession, sample_call_data):
        """Test creating conversation history"""
        # First create a call
        call = CallRecord(**sample_call_data)
        db_session.add(call)
        await db_session.flush()

        # Create conversation history
        conversation = ConversationHistory(
            call_id=call.call_id,
            message_order=1,
            role="user",
            content="Hello, how are you?",
            processing_time_ms=150.5,
            confidence_score=0.95,
            agent_type="conversation",
            llm_tokens_used=25,
            processing_cost=0.001
        )
        db_session.add(conversation)
        await db_session.commit()

        assert conversation.id is not None
        assert conversation.call_id == call.call_id
        assert conversation.message_order == 1
        assert conversation.processing_time_ms == 150.5
        assert conversation.created_at is not None

    @pytest.mark.asyncio
    async def test_conversation_ordering(self, db_session: AsyncSession, sample_call_data, sample_conversation_data):
        """Test conversation message ordering"""
        # Create call
        call = CallRecord(**sample_call_data)
        db_session.add(call)
        await db_session.flush()

        # Add multiple conversation messages
        for msg_data in sample_conversation_data:
            conversation = ConversationHistory(**msg_data)
            db_session.add(conversation)

        await db_session.commit()

        # Verify ordering
        messages = sorted(call.conversations, key=lambda x: x.message_order)
        assert len(messages) == 3
        assert messages[0].message_order == 1
        assert messages[1].message_order == 2
        assert messages[2].message_order == 3


class TestContactRecord:
    """Test ContactRecord model"""

    @pytest.mark.asyncio
    async def test_create_contact(self, db_session: AsyncSession, sample_contact_data):
        """Test creating a contact record"""
        contact = ContactRecord(**sample_contact_data)
        db_session.add(contact)
        await db_session.commit()

        assert contact.id is not None
        assert contact.phone_number == "+1234567890"
        assert contact.name == "Test Contact"
        assert contact.lead_score == 0.8
        assert contact.created_at is not None

    @pytest.mark.asyncio
    async def test_contact_unique_phone(self, db_session: AsyncSession, sample_contact_data):
        """Test contact phone number uniqueness"""
        # Create first contact
        contact1 = ContactRecord(**sample_contact_data)
        db_session.add(contact1)
        await db_session.commit()

        # Try to create second contact with same phone
        contact2 = ContactRecord(**sample_contact_data)
        contact2.name = "Different Name"
        db_session.add(contact2)

        with pytest.raises(Exception):  # Should raise integrity error
            await db_session.commit()


class TestCostTracking:
    """Test CostTracking model"""

    @pytest.mark.asyncio
    async def test_create_cost_entry(self, db_session: AsyncSession):
        """Test creating a cost tracking entry"""
        cost_entry = CostTracking(
            cost_type="llm",
            cost_amount=0.005,
            units_consumed=250,
            unit_type="tokens",
            call_id="test_call_123",
            service_provider="ollama",
            tier="local",
            daily_date="2024-01-15",
            monthly_period="2024-01"
        )
        db_session.add(cost_entry)
        await db_session.commit()

        assert cost_entry.id is not None
        assert cost_entry.cost_type == "llm"
        assert cost_entry.cost_amount == 0.005
        assert cost_entry.incurred_at is not None

    @pytest.mark.asyncio
    async def test_multiple_cost_entries(self, db_session: AsyncSession, sample_cost_data):
        """Test creating multiple cost entries"""
        for cost_data in sample_cost_data:
            cost_data.update({
                "daily_date": "2024-01-15",
                "monthly_period": "2024-01"
            })
            cost_entry = CostTracking(**cost_data)
            db_session.add(cost_entry)

        await db_session.commit()

        # Query back
        result = await db_session.execute(
            "SELECT COUNT(*) FROM cost_tracking WHERE call_id = :call_id",
            {"call_id": "test_call_123"}
        )
        count = result.scalar()
        assert count == 3


class TestSystemMetrics:
    """Test SystemMetrics model"""

    @pytest.mark.asyncio
    async def test_create_system_metric(self, db_session: AsyncSession):
        """Test creating a system metric"""
        metric = SystemMetrics(
            metric_type="performance",
            metric_name="response_time_ms",
            metric_value=150.5,
            metric_unit="ms",
            call_id="test_call_123",
            metadata={"endpoint": "/api/llm", "status": "success"}
        )
        db_session.add(metric)
        await db_session.commit()

        assert metric.id is not None
        assert metric.metric_value == 150.5
        assert metric.recorded_at is not None
        assert metric.metadata["endpoint"] == "/api/llm"


class TestScheduledCalls:
    """Test ScheduledCalls model"""

    @pytest.mark.asyncio
    async def test_create_scheduled_call(self, db_session: AsyncSession):
        """Test creating a scheduled call"""
        scheduled_time = datetime.now()

        scheduled_call = ScheduledCalls(
            phone_number="+1234567890",
            lead_data={"name": "Test Lead", "priority": "high"},
            priority=1,
            scheduled_for=scheduled_time,
            campaign_id="campaign_123",
            campaign_name="Test Campaign"
        )
        db_session.add(scheduled_call)
        await db_session.commit()

        assert scheduled_call.id is not None
        assert scheduled_call.status == "scheduled"
        assert scheduled_call.attempts == 0
        assert scheduled_call.scheduled_for == scheduled_time

    @pytest.mark.asyncio
    async def test_scheduled_call_updates(self, db_session: AsyncSession):
        """Test updating scheduled call status"""
        scheduled_call = ScheduledCalls(
            phone_number="+1234567890",
            scheduled_for=datetime.now(),
            priority=1
        )
        db_session.add(scheduled_call)
        await db_session.commit()

        # Update status
        scheduled_call.status = "in_progress"
        scheduled_call.started_at = datetime.now()
        scheduled_call.attempts = 1
        await db_session.commit()

        assert scheduled_call.status == "in_progress"
        assert scheduled_call.attempts == 1
        assert scheduled_call.started_at is not None


class TestUtilityFunctions:
    """Test utility functions"""

    def test_get_daily_key(self):
        """Test daily key generation"""
        test_date = datetime(2024, 1, 15, 14, 30, 0)
        daily_key = get_daily_key(test_date)

        assert daily_key == "2024-01-15"

    def test_get_monthly_key(self):
        """Test monthly key generation"""
        test_date = datetime(2024, 1, 15, 14, 30, 0)
        monthly_key = get_monthly_key(test_date)

        assert monthly_key == "2024-01"

    def test_keys_with_none_date(self):
        """Test key generation with None date (should use current date)"""
        daily_key = get_daily_key(None)
        monthly_key = get_monthly_key(None)

        assert len(daily_key) == 10  # YYYY-MM-DD format
        assert len(monthly_key) == 7  # YYYY-MM format
        assert daily_key.count('-') == 2
        assert monthly_key.count('-') == 1