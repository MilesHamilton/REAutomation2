"""Integration tests for database layer"""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import CallRecord, ConversationHistory, CostTracking, ContactRecord
from src.database.repositories import CallRepository, ConversationRepository, CostTrackingRepository
from src.database.service import DatabaseService


class TestDatabaseIntegration:
    """Integration tests for database layer"""

    @pytest.mark.asyncio
    async def test_full_call_lifecycle(self, db_session: AsyncSession):
        """Test complete call lifecycle through database"""
        # Create repositories
        call_repo = CallRepository(db_session)
        conv_repo = ConversationRepository(db_session)
        cost_repo = CostTrackingRepository(db_session)

        # 1. Create a call record
        call_data = {
            "call_id": "integration_test_call_456",
            "phone_number": "+1555123456",
            "status": "initiated",
            "start_time": datetime.now(),
            "twilio_call_sid": "CA123456789",
            "direction": "outbound"
        }
        call_id = await call_repo.create_call(call_data)
        assert call_id is not None

        # 2. Add conversation messages
        messages = [
            {
                "call_id": "integration_test_call_456",
                "message_order": 1,
                "role": "assistant",
                "content": "Hello! I'm calling from ABC Realty.",
                "processing_time_ms": 150.0,
                "agent_type": "conversation"
            },
            {
                "call_id": "integration_test_call_456",
                "message_order": 2,
                "role": "user",
                "content": "Hi, what is this about?",
                "processing_time_ms": 0.0,
                "agent_type": "conversation"
            },
            {
                "call_id": "integration_test_call_456",
                "message_order": 3,
                "role": "assistant",
                "content": "I'm calling about your interest in commercial real estate.",
                "processing_time_ms": 200.0,
                "agent_type": "conversation",
                "llm_tokens_used": 35
            }
        ]

        for msg in messages:
            await conv_repo.add_message(msg)

        # 3. Record costs throughout the call
        cost_entries = [
            {
                "cost_type": "llm",
                "cost_amount": 0.002,
                "units_consumed": 25,
                "unit_type": "tokens",
                "call_id": "integration_test_call_456",
                "service_provider": "ollama",
                "tier": "local"
            },
            {
                "cost_type": "tts",
                "cost_amount": 0.015,
                "units_consumed": 75,
                "unit_type": "characters",
                "call_id": "integration_test_call_456",
                "service_provider": "elevenlabs",
                "tier": "premium"
            },
            {
                "cost_type": "twilio",
                "cost_amount": 0.025,
                "units_consumed": 3.0,
                "unit_type": "minutes",
                "call_id": "integration_test_call_456",
                "service_provider": "twilio",
                "tier": "standard"
            }
        ]

        for cost in cost_entries:
            await cost_repo.record_cost(cost)

        # 4. Update call status and end
        await call_repo.update_call_status("integration_test_call_456", "completed")
        end_time = datetime.now()
        await call_repo.update_call_end_time("integration_test_call_456", end_time)

        # 5. Verify all data was stored correctly
        call = await call_repo.get_call_by_id("integration_test_call_456")
        assert call is not None
        assert call.status == "completed"
        assert call.end_time is not None

        # Verify conversation history
        conversation = await conv_repo.get_conversation_history("integration_test_call_456")
        assert len(conversation) == 3
        assert conversation[0].message_order == 1
        assert conversation[2].llm_tokens_used == 35

        # Verify cost tracking
        total_cost = await cost_repo.get_call_total_cost("integration_test_call_456")
        expected_total = 0.002 + 0.015 + 0.025
        assert abs(total_cost - expected_total) < 0.001

        # Verify cost breakdown
        breakdown = await cost_repo.get_call_cost_breakdown("integration_test_call_456")
        assert "llm" in breakdown
        assert "tts" in breakdown
        assert "twilio" in breakdown

    @pytest.mark.asyncio
    async def test_concurrent_call_handling(self, db_session: AsyncSession):
        """Test handling multiple concurrent calls"""
        call_repo = CallRepository(db_session)
        conv_repo = ConversationRepository(db_session)

        # Create multiple concurrent calls
        call_tasks = []
        for i in range(5):
            call_data = {
                "call_id": f"concurrent_call_{i}",
                "phone_number": f"+155512340{i}",
                "status": "initiated",
                "start_time": datetime.now(),
                "twilio_call_sid": f"CA12345678{i}",
                "direction": "outbound"
            }
            task = call_repo.create_call(call_data)
            call_tasks.append(task)

        # Execute all calls concurrently
        call_ids = await asyncio.gather(*call_tasks)
        assert len(call_ids) == 5
        assert all(cid is not None for cid in call_ids)

        # Add messages concurrently
        message_tasks = []
        for i in range(5):
            message_data = {
                "call_id": f"concurrent_call_{i}",
                "message_order": 1,
                "role": "assistant",
                "content": f"Hello from call {i}",
                "processing_time_ms": 100.0,
                "agent_type": "conversation"
            }
            task = conv_repo.add_message(message_data)
            message_tasks.append(task)

        await asyncio.gather(*message_tasks)

        # Verify all calls were created
        for i in range(5):
            call = await call_repo.get_call_by_id(f"concurrent_call_{i}")
            assert call is not None
            assert call.phone_number == f"+155512340{i}"

    @pytest.mark.asyncio
    async def test_cost_analytics_queries(self, db_session: AsyncSession):
        """Test complex cost analytics queries"""
        cost_repo = CostTrackingRepository(db_session)

        # Create cost data for different time periods
        base_date = datetime.now() - timedelta(days=30)
        cost_data = []

        for day in range(30):
            date = base_date + timedelta(days=day)
            daily_key = date.strftime("%Y-%m-%d")
            monthly_key = date.strftime("%Y-%m")

            # Create multiple cost entries per day
            for call_num in range(3):
                cost_entry = {
                    "cost_type": "llm",
                    "cost_amount": 0.003 + (day * 0.001),
                    "units_consumed": 50 + day,
                    "unit_type": "tokens",
                    "call_id": f"analytics_call_{day}_{call_num}",
                    "service_provider": "ollama",
                    "tier": "local",
                    "daily_date": daily_key,
                    "monthly_period": monthly_key
                }
                cost_data.append(cost_entry)

        # Record all costs
        for cost in cost_data:
            await cost_repo.record_cost(cost)

        # Test daily aggregation
        today_key = datetime.now().strftime("%Y-%m-%d")
        yesterday_key = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        daily_costs = await cost_repo.get_daily_costs([yesterday_key, today_key])
        assert len(daily_costs) <= 2

        # Test monthly aggregation
        current_month = datetime.now().strftime("%Y-%m")
        monthly_cost = await cost_repo.get_monthly_cost(current_month)
        assert monthly_cost >= 0

        # Test cost trends
        week_ago = datetime.now() - timedelta(days=7)
        trends = await cost_repo.get_cost_trends(week_ago, datetime.now())
        assert "daily_trend" in trends
        assert "total_cost" in trends

    @pytest.mark.asyncio
    async def test_contact_management_integration(self, db_session: AsyncSession):
        """Test contact management integration"""
        # Create contact record
        contact_data = {
            "phone_number": "+1555987654",
            "name": "Jane Customer",
            "email": "jane@example.com",
            "company": "Customer Corp",
            "lead_score": 0.8,
            "status": "qualified",
            "source": "website",
            "tags": ["enterprise", "urgent"],
            "custom_fields": {"industry": "technology", "employees": "500+"}
        }

        contact = ContactRecord(**contact_data)
        db_session.add(contact)
        await db_session.commit()

        # Create call associated with contact
        call_data = {
            "call_id": "contact_integration_call",
            "phone_number": "+1555987654",
            "status": "completed",
            "start_time": datetime.now() - timedelta(minutes=10),
            "end_time": datetime.now(),
            "twilio_call_sid": "CA_contact_test",
            "direction": "outbound",
            "qualification_score": 0.9,
            "outcome": "scheduled_meeting"
        }

        call_repo = CallRepository(db_session)
        await call_repo.create_call(call_data)

        # Verify relationship
        call = await call_repo.get_call_by_id("contact_integration_call")
        assert call.phone_number == contact.phone_number

        # Test contact lookup by phone
        result = await db_session.execute(
            "SELECT * FROM contact_records WHERE phone_number = :phone",
            {"phone": "+1555987654"}
        )
        found_contact = result.fetchone()
        assert found_contact is not None

    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, db_session: AsyncSession):
        """Test database performance under load"""
        import time

        call_repo = CallRepository(db_session)
        conv_repo = ConversationRepository(db_session)
        cost_repo = CostTrackingRepository(db_session)

        # Measure time for bulk operations
        start_time = time.time()

        # Create 50 calls rapidly
        call_tasks = []
        for i in range(50):
            call_data = {
                "call_id": f"perf_test_call_{i}",
                "phone_number": f"+1555{i:06d}",
                "status": "initiated",
                "start_time": datetime.now(),
                "twilio_call_sid": f"CA_perf_{i}",
                "direction": "outbound"
            }
            call_tasks.append(call_repo.create_call(call_data))

        await asyncio.gather(*call_tasks)

        call_creation_time = time.time() - start_time
        assert call_creation_time < 5.0  # Should complete within 5 seconds

        # Add conversation messages
        start_time = time.time()
        message_tasks = []
        for i in range(50):
            for j in range(5):  # 5 messages per call
                message_data = {
                    "call_id": f"perf_test_call_{i}",
                    "message_order": j + 1,
                    "role": "assistant" if j % 2 == 0 else "user",
                    "content": f"Message {j} in call {i}",
                    "processing_time_ms": 100.0,
                    "agent_type": "conversation"
                }
                message_tasks.append(conv_repo.add_message(message_data))

        await asyncio.gather(*message_tasks)

        message_creation_time = time.time() - start_time
        assert message_creation_time < 10.0  # Should complete within 10 seconds

        # Test query performance
        start_time = time.time()

        # Get all calls from today
        today = datetime.now().strftime("%Y-%m-%d")
        calls = await call_repo.get_calls_by_date_range(
            datetime.now().replace(hour=0, minute=0, second=0),
            datetime.now()
        )

        query_time = time.time() - start_time
        assert query_time < 1.0  # Queries should be fast
        assert len(calls) >= 50

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_session: AsyncSession):
        """Test transaction rollback on errors"""
        call_repo = CallRepository(db_session)

        try:
            # Start a transaction
            call_data = {
                "call_id": "rollback_test_call",
                "phone_number": "+1555rollback",
                "status": "initiated",
                "start_time": datetime.now(),
                "twilio_call_sid": "CA_rollback_test",
                "direction": "outbound"
            }

            await call_repo.create_call(call_data)

            # Intentionally cause an error (duplicate key)
            await call_repo.create_call(call_data)  # Same call_id

        except Exception:
            # Rollback should happen automatically
            await db_session.rollback()

        # Verify the call was not created
        call = await call_repo.get_call_by_id("rollback_test_call")
        assert call is None

    @pytest.mark.asyncio
    async def test_database_service_integration(self):
        """Test database service integration"""
        db_service = DatabaseService()

        # Test initialization
        await db_service.initialize()
        assert db_service.is_ready() is True

        # Test getting session
        async with db_service.get_session() as session:
            assert session is not None
            assert isinstance(session, AsyncSession)

            # Test basic query
            result = await session.execute("SELECT 1")
            value = result.scalar()
            assert value == 1

        # Test cleanup
        await db_service.cleanup()

    @pytest.mark.asyncio
    async def test_migration_compatibility(self, db_session: AsyncSession):
        """Test that current schema matches expectations"""
        # Test that all expected tables exist
        expected_tables = [
            'call_records',
            'conversation_history',
            'cost_tracking',
            'contact_records',
            'system_metrics',
            'scheduled_calls',
            'tier_switch_history'
        ]

        for table_name in expected_tables:
            result = await db_session.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)",
                {"table_name": table_name}
            )
            exists = result.scalar()
            assert exists is True, f"Table {table_name} should exist"

        # Test that indexes exist for performance-critical queries
        # This would be database-specific, simplified for testing
        result = await db_session.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'call_records'"
        )
        indexes = [row[0] for row in result.fetchall()]
        assert any("call_id" in idx for idx in indexes), "call_id index should exist"