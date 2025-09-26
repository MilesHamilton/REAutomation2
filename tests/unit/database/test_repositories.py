"""Tests for database repositories"""

import pytest
from datetime import datetime, timedelta

from src.database.repositories import (
    CallRepository, ConversationRepository, ContactRepository,
    CostTrackingRepository, MetricsRepository, ScheduledCallRepository
)
from src.database.models import CallRecord, ConversationHistory, ContactRecord


class TestCallRepository:
    """Test CallRepository"""

    @pytest.mark.asyncio
    async def test_create_call(self, call_repository, sample_call_data):
        """Test creating a call"""
        call = await call_repository.create_call(sample_call_data)

        assert call.call_id == "test_call_123"
        assert call.phone_number == "+1234567890"
        assert call.status == "initiated"

    @pytest.mark.asyncio
    async def test_get_call_by_id(self, call_repository, sample_call_data):
        """Test getting call by ID"""
        # Create call
        await call_repository.create_call(sample_call_data)
        await call_repository.commit()

        # Get call
        call = await call_repository.get_call_by_id("test_call_123")

        assert call is not None
        assert call.call_id == "test_call_123"
        assert call.phone_number == "+1234567890"

    @pytest.mark.asyncio
    async def test_update_call_status(self, call_repository, sample_call_data):
        """Test updating call status"""
        # Create call
        await call_repository.create_call(sample_call_data)
        await call_repository.commit()

        # Update status
        success = await call_repository.update_call_status(
            "test_call_123",
            "connected",
            qualification_score=0.85,
            duration_seconds=120.5
        )

        assert success is True

        # Verify update
        call = await call_repository.get_call_by_id("test_call_123")
        assert call.status == "connected"
        assert call.qualification_score == 0.85
        assert call.duration_seconds == 120.5

    @pytest.mark.asyncio
    async def test_end_call(self, call_repository, sample_call_data):
        """Test ending a call"""
        # Create call
        await call_repository.create_call(sample_call_data)
        await call_repository.commit()

        # End call
        metrics = {
            "duration_seconds": 180.0,
            "total_cost": 0.045,
            "qualification_score": 0.75,
            "qualified": True
        }
        success = await call_repository.end_call("test_call_123", "completed", metrics)

        assert success is True

        # Verify update
        call = await call_repository.get_call_by_id("test_call_123")
        assert call.status == "completed"
        assert call.ended_at is not None
        assert call.duration_seconds == 180.0
        assert call.qualified is True

    @pytest.mark.asyncio
    async def test_get_active_calls(self, call_repository):
        """Test getting active calls"""
        # Create multiple calls with different statuses
        call_data_1 = {
            "call_id": "active_call_1",
            "phone_number": "+1111111111",
            "status": "connected"
        }
        call_data_2 = {
            "call_id": "active_call_2",
            "phone_number": "+2222222222",
            "status": "ringing"
        }
        call_data_3 = {
            "call_id": "completed_call",
            "phone_number": "+3333333333",
            "status": "completed"
        }

        await call_repository.create_call(call_data_1)
        await call_repository.create_call(call_data_2)
        await call_repository.create_call(call_data_3)
        await call_repository.commit()

        # Get active calls
        active_calls = await call_repository.get_active_calls()

        assert len(active_calls) == 2
        active_call_ids = {call.call_id for call in active_calls}
        assert "active_call_1" in active_call_ids
        assert "active_call_2" in active_call_ids
        assert "completed_call" not in active_call_ids

    @pytest.mark.asyncio
    async def test_get_calls_by_date_range(self, call_repository):
        """Test getting calls by date range"""
        now = datetime.now()
        yesterday = now - timedelta(days=1)

        # Create calls with different timestamps
        call_data_1 = {
            "call_id": "recent_call",
            "phone_number": "+1111111111",
            "status": "completed"
        }
        call_data_2 = {
            "call_id": "old_call",
            "phone_number": "+2222222222",
            "status": "completed"
        }

        call_1 = await call_repository.create_call(call_data_1)
        call_2 = await call_repository.create_call(call_data_2)

        # Manually set created_at for testing
        call_1.created_at = now
        call_2.created_at = yesterday - timedelta(days=1)  # 2 days ago

        await call_repository.commit()

        # Get calls from yesterday to now
        calls = await call_repository.get_calls_by_date_range(
            start_date=yesterday,
            end_date=now + timedelta(minutes=1)
        )

        assert len(calls) == 1
        assert calls[0].call_id == "recent_call"

    @pytest.mark.asyncio
    async def test_get_call_metrics_summary(self, call_repository):
        """Test getting call metrics summary"""
        # Create multiple calls with various metrics
        for i in range(5):
            call_data = {
                "call_id": f"metrics_call_{i}",
                "phone_number": f"+{i:>10}",
                "status": "completed",
                "qualification_score": 0.5 + (i * 0.1),
                "qualified": i >= 3,  # Last 2 calls qualified
                "duration_seconds": 60.0 + (i * 30),
                "total_cost": 0.02 + (i * 0.01)
            }
            await call_repository.create_call(call_data)

        await call_repository.commit()

        # Get metrics summary
        summary = await call_repository.get_call_metrics_summary(days=7)

        assert summary["total_calls"] == 5
        assert summary["completed_calls"] == 5
        assert summary["qualified_calls"] == 2
        assert summary["qualification_rate"] == 0.4  # 2/5
        assert summary["avg_duration_seconds"] > 0
        assert summary["total_cost"] > 0


class TestConversationRepository:
    """Test ConversationRepository"""

    @pytest.mark.asyncio
    async def test_add_message(self, conversation_repository, call_repository, sample_call_data):
        """Test adding a conversation message"""
        # Create call first
        await call_repository.create_call(sample_call_data)
        await call_repository.commit()

        # Add message
        message = await conversation_repository.add_message(
            call_id="test_call_123",
            role="user",
            content="Hello, how are you?",
            message_order=1,
            processing_time_ms=120.5
        )

        assert message.call_id == "test_call_123"
        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert message.processing_time_ms == 120.5

    @pytest.mark.asyncio
    async def test_get_conversation_history(self, conversation_repository, call_repository, sample_call_data):
        """Test getting conversation history"""
        # Create call
        await call_repository.create_call(sample_call_data)
        await call_repository.commit()

        # Add multiple messages
        messages_data = [
            {"role": "assistant", "content": "Hello!", "message_order": 1},
            {"role": "user", "content": "Hi there", "message_order": 2},
            {"role": "assistant", "content": "How can I help?", "message_order": 3}
        ]

        for msg_data in messages_data:
            await conversation_repository.add_message(
                call_id="test_call_123",
                **msg_data
            )

        await conversation_repository.commit()

        # Get history
        history = await conversation_repository.get_conversation_history("test_call_123")

        assert len(history) == 3
        assert history[0].message_order == 1
        assert history[1].message_order == 2
        assert history[2].message_order == 3
        assert history[0].role == "assistant"
        assert history[1].role == "user"

    @pytest.mark.asyncio
    async def test_get_conversation_summary(self, conversation_repository, call_repository, sample_call_data):
        """Test getting conversation summary"""
        # Create call
        await call_repository.create_call(sample_call_data)
        await call_repository.commit()

        # Add messages with processing costs
        messages_data = [
            {"role": "assistant", "content": "Hello!", "message_order": 1, "processing_cost": 0.001},
            {"role": "user", "content": "Hi there", "message_order": 2, "processing_cost": 0.0},
            {"role": "assistant", "content": "How can I help?", "message_order": 3, "processing_cost": 0.002}
        ]

        for msg_data in messages_data:
            await conversation_repository.add_message(
                call_id="test_call_123",
                **msg_data
            )

        await conversation_repository.commit()

        # Get summary
        summary = await conversation_repository.get_conversation_summary("test_call_123")

        assert summary["total_messages"] == 3
        assert summary["user_messages"] == 1
        assert summary["assistant_messages"] == 2
        assert summary["total_processing_cost"] == 0.003


class TestContactRepository:
    """Test ContactRepository"""

    @pytest.mark.asyncio
    async def test_create_contact(self, contact_repository, sample_contact_data):
        """Test creating a new contact"""
        contact = await contact_repository.create_or_update_contact(
            "+1234567890", sample_contact_data
        )

        assert contact.phone_number == "+1234567890"
        assert contact.name == "Test Contact"
        assert contact.lead_score == 0.8

    @pytest.mark.asyncio
    async def test_update_existing_contact(self, contact_repository, sample_contact_data):
        """Test updating an existing contact"""
        # Create contact
        contact = await contact_repository.create_or_update_contact(
            "+1234567890", sample_contact_data
        )
        await contact_repository.commit()

        # Update contact
        updated_data = {"name": "Updated Name", "lead_score": 0.9}
        updated_contact = await contact_repository.create_or_update_contact(
            "+1234567890", updated_data
        )

        assert updated_contact.id == contact.id  # Same contact
        assert updated_contact.name == "Updated Name"
        assert updated_contact.lead_score == 0.9

    @pytest.mark.asyncio
    async def test_get_contact_by_phone(self, contact_repository, sample_contact_data):
        """Test getting contact by phone number"""
        # Create contact
        await contact_repository.create_or_update_contact(
            "+1234567890", sample_contact_data
        )
        await contact_repository.commit()

        # Get contact
        contact = await contact_repository.get_contact_by_phone("+1234567890")

        assert contact is not None
        assert contact.phone_number == "+1234567890"
        assert contact.name == "Test Contact"

    @pytest.mark.asyncio
    async def test_update_call_stats(self, contact_repository, sample_contact_data):
        """Test updating call statistics"""
        # Create contact
        contact = await contact_repository.create_or_update_contact(
            "+1234567890", sample_contact_data
        )
        await contact_repository.commit()

        # Update call stats
        await contact_repository.update_call_stats("+1234567890", qualified=True)
        await contact_repository.commit()

        # Check updated stats
        updated_contact = await contact_repository.get_contact_by_phone("+1234567890")
        assert updated_contact.total_calls == 1
        assert updated_contact.qualified_calls == 1


class TestCostTrackingRepository:
    """Test CostTrackingRepository"""

    @pytest.mark.asyncio
    async def test_record_cost(self, cost_repository):
        """Test recording a cost entry"""
        cost_entry = await cost_repository.record_cost(
            cost_type="llm",
            cost_amount=0.005,
            call_id="test_call_123",
            units_consumed=250,
            unit_type="tokens"
        )

        assert cost_entry.cost_type == "llm"
        assert cost_entry.cost_amount == 0.005
        assert cost_entry.call_id == "test_call_123"

    @pytest.mark.asyncio
    async def test_get_daily_costs(self, cost_repository):
        """Test getting daily costs"""
        # Record multiple costs
        costs_data = [
            ("llm", 0.005),
            ("tts", 0.010),
            ("twilio", 0.025)
        ]

        for cost_type, amount in costs_data:
            await cost_repository.record_cost(cost_type, amount)

        await cost_repository.commit()

        # Get daily costs
        daily_costs = await cost_repository.get_daily_costs()

        assert "llm" in daily_costs
        assert "tts" in daily_costs
        assert "twilio" in daily_costs
        assert daily_costs["llm"] == 0.005
        assert daily_costs["tts"] == 0.010
        assert daily_costs["twilio"] == 0.025

    @pytest.mark.asyncio
    async def test_get_call_cost_breakdown(self, cost_repository):
        """Test getting cost breakdown for a call"""
        call_id = "test_call_breakdown"

        # Record costs for specific call
        costs = [
            ("llm", 0.003, call_id),
            ("tts", 0.008, call_id),
            ("twilio", 0.020, call_id)
        ]

        for cost_type, amount, cid in costs:
            await cost_repository.record_cost(cost_type, amount, call_id=cid)

        await cost_repository.commit()

        # Get breakdown
        breakdown = await cost_repository.get_call_cost_breakdown(call_id)

        assert len(breakdown) == 3
        assert breakdown["llm"] == 0.003
        assert breakdown["tts"] == 0.008
        assert breakdown["twilio"] == 0.020

    @pytest.mark.asyncio
    async def test_check_budget_status(self, cost_repository):
        """Test checking budget status"""
        daily_budget = 5.0

        # Record some costs
        await cost_repository.record_cost("twilio", 2.0)
        await cost_repository.record_cost("llm", 0.5)
        await cost_repository.commit()

        # Check budget status
        status = await cost_repository.check_budget_status(daily_budget)

        assert status["daily_budget"] == 5.0
        assert status["spent_today"] == 2.5
        assert status["remaining_budget"] == 2.5
        assert status["budget_utilization"] == 0.5
        assert status["over_budget"] is False


class TestScheduledCallRepository:
    """Test ScheduledCallRepository"""

    @pytest.mark.asyncio
    async def test_create_scheduled_call(self, scheduled_call_repository):
        """Test creating a scheduled call"""
        call_data = {
            "phone_number": "+1234567890",
            "scheduled_for": datetime.now() + timedelta(hours=1),
            "priority": 1,
            "campaign_id": "test_campaign"
        }

        scheduled_call = await scheduled_call_repository.create_scheduled_call(call_data)

        assert scheduled_call.phone_number == "+1234567890"
        assert scheduled_call.status == "scheduled"
        assert scheduled_call.priority == 1

    @pytest.mark.asyncio
    async def test_get_pending_calls(self, scheduled_call_repository):
        """Test getting pending calls"""
        now = datetime.now()

        # Create calls with different scheduled times
        call_data_1 = {
            "phone_number": "+1111111111",
            "scheduled_for": now - timedelta(minutes=10),  # Past due
            "priority": 2
        }
        call_data_2 = {
            "phone_number": "+2222222222",
            "scheduled_for": now - timedelta(minutes=5),   # Past due
            "priority": 1
        }
        call_data_3 = {
            "phone_number": "+3333333333",
            "scheduled_for": now + timedelta(hours=1),     # Future
            "priority": 3
        }

        await scheduled_call_repository.create_scheduled_call(call_data_1)
        await scheduled_call_repository.create_scheduled_call(call_data_2)
        await scheduled_call_repository.create_scheduled_call(call_data_3)
        await scheduled_call_repository.commit()

        # Get pending calls
        pending = await scheduled_call_repository.get_pending_calls()

        assert len(pending) == 2  # Only past due calls
        # Should be ordered by priority (higher first), then by time
        assert pending[0].priority >= pending[1].priority

    @pytest.mark.asyncio
    async def test_update_call_status(self, scheduled_call_repository):
        """Test updating scheduled call status"""
        # Create scheduled call
        call_data = {
            "phone_number": "+1234567890",
            "scheduled_for": datetime.now(),
            "priority": 1
        }

        scheduled_call = await scheduled_call_repository.create_scheduled_call(call_data)
        await scheduled_call_repository.commit()

        # Update status
        success = await scheduled_call_repository.update_call_status(
            scheduled_call.id,
            "completed",
            completion_status="successful",
            call_id="actual_call_123"
        )

        assert success is True

        # Verify update (would need to query back to fully verify)
        await scheduled_call_repository.commit()


class TestMetricsRepository:
    """Test MetricsRepository"""

    @pytest.mark.asyncio
    async def test_record_metric(self, metrics_repository):
        """Test recording a system metric"""
        metric = await metrics_repository.record_metric(
            metric_type="performance",
            metric_name="response_time",
            metric_value=150.5,
            call_id="test_call_123",
            metadata={"endpoint": "/api/test"}
        )

        assert metric.metric_type == "performance"
        assert metric.metric_name == "response_time"
        assert metric.metric_value == 150.5

    @pytest.mark.asyncio
    async def test_get_aggregated_metrics(self, metrics_repository):
        """Test getting aggregated metrics"""
        # Record multiple metrics
        metric_name = "test_metric"
        values = [100.0, 150.0, 200.0, 125.0, 175.0]

        for value in values:
            await metrics_repository.record_metric(
                metric_type="test",
                metric_name=metric_name,
                metric_value=value
            )

        await metrics_repository.commit()

        # Get aggregated metrics
        aggregated = await metrics_repository.get_aggregated_metrics(metric_name, hours=24)

        assert aggregated["count"] == 5
        assert aggregated["avg_value"] == 150.0  # Average of values
        assert aggregated["min_value"] == 100.0
        assert aggregated["max_value"] == 200.0