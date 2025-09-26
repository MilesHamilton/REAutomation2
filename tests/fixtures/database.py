"""Database test fixtures and utilities"""

import pytest
import asyncio
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database, drop_database
import os
import tempfile

from src.database.models import Base
from src.database.connection import DatabaseManager
from src.database.repositories import (
    CallRepository, ConversationRepository, ContactRepository,
    CostTrackingRepository, MetricsRepository, ScheduledCallRepository
)


@pytest.fixture(scope="session")
def test_database_url():
    """Create a test database URL"""
    # Use SQLite for testing by default
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()
    return f"sqlite+aiosqlite:///{temp_db.name}"


@pytest.fixture(scope="session")
async def test_engine(test_database_url):
    """Create test database engine"""
    engine = create_async_engine(
        test_database_url,
        echo=False,
        future=True
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    await engine.dispose()
    # Remove temp file
    db_path = test_database_url.replace("sqlite+aiosqlite:///", "")
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
    SessionLocal = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with SessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def db_manager(test_database_url) -> AsyncGenerator[DatabaseManager, None]:
    """Create a test database manager"""
    manager = DatabaseManager(test_database_url)
    await manager.initialize()
    await manager.create_tables()

    yield manager

    await manager.cleanup()


@pytest.fixture
def call_repository(db_session) -> CallRepository:
    """Create a call repository for testing"""
    return CallRepository(db_session)


@pytest.fixture
def conversation_repository(db_session) -> ConversationRepository:
    """Create a conversation repository for testing"""
    return ConversationRepository(db_session)


@pytest.fixture
def contact_repository(db_session) -> ContactRepository:
    """Create a contact repository for testing"""
    return ContactRepository(db_session)


@pytest.fixture
def cost_repository(db_session) -> CostTrackingRepository:
    """Create a cost tracking repository for testing"""
    return CostTrackingRepository(db_session)


@pytest.fixture
def metrics_repository(db_session) -> MetricsRepository:
    """Create a metrics repository for testing"""
    return MetricsRepository(db_session)


@pytest.fixture
def scheduled_call_repository(db_session) -> ScheduledCallRepository:
    """Create a scheduled call repository for testing"""
    return ScheduledCallRepository(db_session)


# Test data factories
@pytest.fixture
def sample_call_data():
    """Sample call data for testing"""
    return {
        "call_id": "test_call_123",
        "phone_number": "+1234567890",
        "status": "initiated",
        "initial_tier": "local_piper",
        "final_tier": "local_piper",
        "lead_data": {"name": "Test Lead", "company": "Test Corp"},
        "total_cost": 0.05,
        "qualified": False
    }


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing"""
    return [
        {
            "call_id": "test_call_123",
            "message_order": 1,
            "role": "assistant",
            "content": "Hello! This is a test call."
        },
        {
            "call_id": "test_call_123",
            "message_order": 2,
            "role": "user",
            "content": "Hello, who is this?"
        },
        {
            "call_id": "test_call_123",
            "message_order": 3,
            "role": "assistant",
            "content": "I'm calling about your recent inquiry."
        }
    ]


@pytest.fixture
def sample_contact_data():
    """Sample contact data for testing"""
    return {
        "phone_number": "+1234567890",
        "name": "Test Contact",
        "email": "test@example.com",
        "company": "Test Corp",
        "lead_score": 0.8,
        "lead_status": "qualified"
    }


@pytest.fixture
def sample_cost_data():
    """Sample cost data for testing"""
    return [
        {
            "cost_type": "llm",
            "cost_amount": 0.001,
            "call_id": "test_call_123",
            "units_consumed": 100,
            "unit_type": "tokens"
        },
        {
            "cost_type": "tts",
            "cost_amount": 0.002,
            "call_id": "test_call_123",
            "units_consumed": 50,
            "unit_type": "characters"
        },
        {
            "cost_type": "twilio",
            "cost_amount": 0.025,
            "call_id": "test_call_123",
            "units_consumed": 3.0,
            "unit_type": "minutes"
        }
    ]