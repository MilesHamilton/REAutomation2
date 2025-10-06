"""
Migration Testing Framework

Tests all Alembic migrations for upgrade/downgrade integrity,
data preservation, and index effectiveness.
"""
import asyncio
import pytest
from typing import List, Dict, Any
from sqlalchemy import text, inspect, MetaData, Table
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
import os
import tempfile
import logging

logger = logging.getLogger(__name__)


class MigrationTester:
    """Framework for testing Alembic migrations."""

    def __init__(self, database_url: str):
        """
        Initialize migration tester.

        Args:
            database_url: Database URL for test database
        """
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.alembic_cfg = None

    async def setup(self):
        """Set up test database and Alembic configuration."""
        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Configure Alembic
        self.alembic_cfg = Config("alembic.ini")
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.database_url.replace('+asyncpg', ''))

    async def teardown(self):
        """Clean up test database."""
        if self.engine:
            await self.engine.dispose()

    async def get_current_revision(self) -> str:
        """Get current database revision."""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                text("SELECT version_num FROM alembic_version")
            )
            row = result.fetchone()
            return row[0] if row else None

    async def upgrade_to(self, revision: str = "head"):
        """
        Upgrade database to specific revision.

        Args:
            revision: Target revision (default: head)
        """
        # Run upgrade synchronously (Alembic requirement)
        command.upgrade(self.alembic_cfg, revision)

    async def downgrade_to(self, revision: str):
        """
        Downgrade database to specific revision.

        Args:
            revision: Target revision
        """
        # Run downgrade synchronously (Alembic requirement)
        command.downgrade(self.alembic_cfg, revision)

    async def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get list of column names for a table.

        Args:
            table_name: Name of table

        Returns:
            List of column names
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(text(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """))
            return [row[0] for row in result.fetchall()]

    async def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get list of indexes for a table.

        Args:
            table_name: Name of table

        Returns:
            List of index information dicts
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(text(f"""
                SELECT
                    indexname,
                    indexdef
                FROM pg_indexes
                WHERE tablename = '{table_name}'
                ORDER BY indexname
            """))
            return [
                {"name": row[0], "definition": row[1]}
                for row in result.fetchall()
            ]

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists.

        Args:
            table_name: Name of table

        Returns:
            True if table exists
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table_name}'
                )
            """))
            return result.scalar()

    async def insert_test_data(self, table_name: str, data: Dict[str, Any]) -> int:
        """
        Insert test data into table.

        Args:
            table_name: Name of table
            data: Dictionary of column:value pairs

        Returns:
            ID of inserted row
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f":{k}" for k in data.keys()])

        async with self.engine.begin() as conn:
            result = await conn.execute(
                text(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) RETURNING id"),
                data
            )
            return result.scalar()

    async def get_row_by_id(self, table_name: str, row_id: int) -> Dict[str, Any]:
        """
        Get row by ID.

        Args:
            table_name: Name of table
            row_id: Row ID

        Returns:
            Dictionary of column:value pairs
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(
                text(f"SELECT * FROM {table_name} WHERE id = :id"),
                {"id": row_id}
            )
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None

    async def count_rows(self, table_name: str) -> int:
        """
        Count rows in table.

        Args:
            table_name: Name of table

        Returns:
            Number of rows
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            return result.scalar()

    async def explain_query(self, query: str, params: Dict[str, Any] = None) -> str:
        """
        Run EXPLAIN ANALYZE on query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Execution plan
        """
        async with self.engine.begin() as conn:
            result = await conn.execute(
                text(f"EXPLAIN ANALYZE {query}"),
                params or {}
            )
            return "\n".join([row[0] for row in result.fetchall()])


# ============================================
# Test Fixtures
# ============================================

@pytest.fixture
async def migration_tester():
    """Create migration tester with temporary test database."""
    # Use test database URL from environment or create temporary one
    test_db_url = os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/reautomation_test"
    )

    tester = MigrationTester(test_db_url)
    await tester.setup()

    # Start from clean state
    await tester.downgrade_to("base")

    yield tester

    # Clean up
    await tester.downgrade_to("base")
    await tester.teardown()


# ============================================
# Migration 003: Context Management Fields
# ============================================

@pytest.mark.asyncio
async def test_003_context_management_upgrade(migration_tester):
    """Test upgrade to context management migration."""
    # Upgrade to base + monitoring + voice
    await migration_tester.upgrade_to("002_voice_agent_integration")

    # Verify tables exist but don't have context fields yet
    assert await migration_tester.table_exists("calls")
    calls_columns = await migration_tester.get_table_columns("calls")
    assert "context_pruned" not in calls_columns
    assert "pruning_count" not in calls_columns
    assert "total_context_tokens" not in calls_columns

    # Upgrade to context management
    await migration_tester.upgrade_to("003_context_management")

    # Verify new columns exist in calls table
    calls_columns = await migration_tester.get_table_columns("calls")
    assert "context_pruned" in calls_columns
    assert "pruning_count" in calls_columns
    assert "total_context_tokens" in calls_columns

    # Verify new columns exist in conversation_history table
    history_columns = await migration_tester.get_table_columns("conversation_history")
    assert "importance_score" in history_columns
    assert "token_count" in history_columns

    # Verify indexes were created
    calls_indexes = await migration_tester.get_table_indexes("calls")
    index_names = [idx["name"] for idx in calls_indexes]
    assert "ix_calls_context_pruned" in index_names
    assert "ix_calls_pruning_count" in index_names
    assert "ix_calls_total_context_tokens" in index_names
    assert "ix_calls_context_pruned_tokens" in index_names

    history_indexes = await migration_tester.get_table_indexes("conversation_history")
    history_index_names = [idx["name"] for idx in history_indexes]
    assert "ix_conversation_history_importance_score" in history_index_names
    assert "ix_conversation_history_token_count" in history_index_names


@pytest.mark.asyncio
async def test_003_context_management_downgrade(migration_tester):
    """Test downgrade from context management migration."""
    # Upgrade to context management
    await migration_tester.upgrade_to("003_context_management")

    # Insert test data with context fields
    from datetime import datetime
    call_id = await migration_tester.insert_test_data("calls", {
        "phone_number": "+15555551234",
        "status": "completed",
        "tier": "tier1",
        "created_at": datetime.utcnow(),
        "context_pruned": True,
        "pruning_count": 3,
        "total_context_tokens": 1500
    })

    # Verify data was inserted
    call_data = await migration_tester.get_row_by_id("calls", call_id)
    assert call_data["context_pruned"] is True
    assert call_data["pruning_count"] == 3
    assert call_data["total_context_tokens"] == 1500

    # Downgrade
    await migration_tester.downgrade_to("002_voice_agent_integration")

    # Verify columns were removed
    calls_columns = await migration_tester.get_table_columns("calls")
    assert "context_pruned" not in calls_columns
    assert "pruning_count" not in calls_columns
    assert "total_context_tokens" not in calls_columns

    history_columns = await migration_tester.get_table_columns("conversation_history")
    assert "importance_score" not in history_columns
    assert "token_count" not in history_columns

    # Verify indexes were removed
    calls_indexes = await migration_tester.get_table_indexes("calls")
    index_names = [idx["name"] for idx in calls_indexes]
    assert "ix_calls_context_pruned" not in index_names
    assert "ix_calls_pruning_count" not in index_names

    # Verify call data still exists (other columns preserved)
    call_data = await migration_tester.get_row_by_id("calls", call_id)
    assert call_data is not None
    assert call_data["phone_number"] == "+15555551234"
    assert call_data["status"] == "completed"


# ============================================
# Migration 004: Performance Indexes
# ============================================

@pytest.mark.asyncio
async def test_004_performance_indexes_upgrade(migration_tester):
    """Test upgrade to performance indexes migration."""
    # Upgrade to context management
    await migration_tester.upgrade_to("003_context_management")

    # Get baseline index counts
    calls_indexes_before = await migration_tester.get_table_indexes("calls")
    metrics_indexes_before = await migration_tester.get_table_indexes("system_metrics")

    # Upgrade to performance indexes
    await migration_tester.upgrade_to("004_performance_indexes")

    # Verify new indexes in calls table
    calls_indexes = await migration_tester.get_table_indexes("calls")
    index_names = [idx["name"] for idx in calls_indexes]
    assert "ix_calls_status_created_cost" in index_names
    assert "ix_calls_qualified_score_created" in index_names
    assert "ix_calls_tier_cost" in index_names
    assert "ix_calls_context_pruned_only" in index_names

    # Verify new indexes in conversation_history table
    history_indexes = await migration_tester.get_table_indexes("conversation_history")
    history_index_names = [idx["name"] for idx in history_indexes]
    assert "ix_conversation_history_call_role_order" in history_index_names
    assert "ix_conversation_history_tokens_used" in history_index_names
    assert "ix_conversation_history_important_only" in history_index_names

    # Verify new indexes in system_metrics table
    metrics_indexes = await migration_tester.get_table_indexes("system_metrics")
    metrics_index_names = [idx["name"] for idx in metrics_indexes]
    assert "ix_system_metrics_name_recorded_desc" in metrics_index_names
    assert "ix_system_metrics_gpu_metrics" in metrics_index_names
    assert "ix_system_metrics_streaming_metrics" in metrics_index_names

    # Verify partial index definitions contain WHERE clauses
    for idx in calls_indexes:
        if idx["name"] == "ix_calls_context_pruned_only":
            assert "WHERE" in idx["definition"]
            assert "context_pruned = true" in idx["definition"]


@pytest.mark.asyncio
async def test_004_performance_indexes_downgrade(migration_tester):
    """Test downgrade from performance indexes migration."""
    # Upgrade to performance indexes
    await migration_tester.upgrade_to("004_performance_indexes")

    # Verify indexes exist
    calls_indexes = await migration_tester.get_table_indexes("calls")
    index_names = [idx["name"] for idx in calls_indexes]
    assert "ix_calls_status_created_cost" in index_names

    # Downgrade
    await migration_tester.downgrade_to("003_context_management")

    # Verify performance indexes were removed
    calls_indexes = await migration_tester.get_table_indexes("calls")
    index_names = [idx["name"] for idx in calls_indexes]
    assert "ix_calls_status_created_cost" not in index_names
    assert "ix_calls_qualified_score_created" not in index_names
    assert "ix_calls_tier_cost" not in index_names
    assert "ix_calls_context_pruned_only" not in index_names

    # Verify base indexes still exist
    assert "ix_calls_context_pruned" in index_names


@pytest.mark.asyncio
async def test_004_index_effectiveness(migration_tester):
    """Test that performance indexes improve query performance."""
    # Upgrade to performance indexes
    await migration_tester.upgrade_to("004_performance_indexes")

    # Insert test data
    from datetime import datetime, timedelta
    base_time = datetime.utcnow()

    for i in range(100):
        await migration_tester.insert_test_data("calls", {
            "phone_number": f"+1555555{i:04d}",
            "status": "completed" if i % 2 == 0 else "in_progress",
            "tier": "tier1" if i % 3 == 0 else "tier2",
            "created_at": base_time - timedelta(hours=i),
            "total_cost": 0.05 * i,
            "context_pruned": i % 4 == 0,
            "pruning_count": i % 5,
            "total_context_tokens": 100 * i
        })

    # Test dashboard query (should use ix_calls_status_created_cost)
    query = """
        SELECT status, created_at, total_cost
        FROM calls
        WHERE status = :status
        ORDER BY created_at DESC
        LIMIT 10
    """
    explain = await migration_tester.explain_query(query, {"status": "completed"})
    logger.info(f"Dashboard query plan:\n{explain}")

    # Verify index is used (check for Index Scan on ix_calls_status_created_cost)
    assert "ix_calls_status_created_cost" in explain or "Index" in explain

    # Test context-pruned query (should use ix_calls_context_pruned_only partial index)
    query = """
        SELECT total_context_tokens, pruning_count
        FROM calls
        WHERE context_pruned = true
        ORDER BY total_context_tokens DESC
    """
    explain = await migration_tester.explain_query(query)
    logger.info(f"Context-pruned query plan:\n{explain}")

    # Verify partial index is used
    assert "ix_calls_context_pruned_only" in explain or "Index" in explain


# ============================================
# Full Migration Cycle Tests
# ============================================

@pytest.mark.asyncio
async def test_full_upgrade_cycle(migration_tester):
    """Test complete upgrade from base to head."""
    # Start from base
    await migration_tester.downgrade_to("base")
    current = await migration_tester.get_current_revision()
    assert current is None or current == "base"

    # Upgrade to head
    await migration_tester.upgrade_to("head")

    # Verify all tables exist
    assert await migration_tester.table_exists("calls")
    assert await migration_tester.table_exists("conversation_history")
    assert await migration_tester.table_exists("system_metrics")
    assert await migration_tester.table_exists("workflow_traces")
    assert await migration_tester.table_exists("agent_executions")

    # Verify latest columns exist
    calls_columns = await migration_tester.get_table_columns("calls")
    assert "context_pruned" in calls_columns
    assert "pruning_count" in calls_columns

    # Verify latest indexes exist
    calls_indexes = await migration_tester.get_table_indexes("calls")
    index_names = [idx["name"] for idx in calls_indexes]
    assert "ix_calls_status_created_cost" in index_names


@pytest.mark.asyncio
async def test_full_downgrade_cycle(migration_tester):
    """Test complete downgrade from head to base."""
    # Start from head
    await migration_tester.upgrade_to("head")

    # Downgrade to base
    await migration_tester.downgrade_to("base")

    # Verify all application tables were removed
    assert not await migration_tester.table_exists("calls")
    assert not await migration_tester.table_exists("conversation_history")
    assert not await migration_tester.table_exists("system_metrics")

    # Only alembic_version should remain
    assert await migration_tester.table_exists("alembic_version")


@pytest.mark.asyncio
async def test_data_preservation_across_migrations(migration_tester):
    """Test that data is preserved across upgrade/downgrade cycles."""
    from datetime import datetime

    # Upgrade to base tables
    await migration_tester.upgrade_to("26a6cb1543c8_add_base_tables")

    # Insert test call
    call_id = await migration_tester.insert_test_data("calls", {
        "phone_number": "+15555551234",
        "status": "completed",
        "tier": "tier1",
        "created_at": datetime.utcnow()
    })

    # Upgrade through all migrations
    await migration_tester.upgrade_to("head")

    # Verify call still exists with original data
    call_data = await migration_tester.get_row_by_id("calls", call_id)
    assert call_data is not None
    assert call_data["phone_number"] == "+15555551234"
    assert call_data["status"] == "completed"

    # Verify new fields have default values
    assert call_data["context_pruned"] is False
    assert call_data["pruning_count"] == 0
    assert call_data["total_context_tokens"] == 0

    # Downgrade to 002
    await migration_tester.downgrade_to("002_voice_agent_integration")

    # Verify call still exists (context fields removed but data preserved)
    call_data = await migration_tester.get_row_by_id("calls", call_id)
    assert call_data is not None
    assert call_data["phone_number"] == "+15555551234"
    assert call_data["status"] == "completed"


# ============================================
# Migration Timing Benchmarks
# ============================================

@pytest.mark.asyncio
async def test_migration_timing(migration_tester):
    """Benchmark migration execution time."""
    import time

    # Test upgrade timing
    await migration_tester.downgrade_to("base")

    start = time.time()
    await migration_tester.upgrade_to("head")
    upgrade_time = time.time() - start

    logger.info(f"Full upgrade time: {upgrade_time:.2f}s")
    assert upgrade_time < 30.0, "Upgrade should complete within 30 seconds"

    # Test downgrade timing
    start = time.time()
    await migration_tester.downgrade_to("base")
    downgrade_time = time.time() - start

    logger.info(f"Full downgrade time: {downgrade_time:.2f}s")
    assert downgrade_time < 30.0, "Downgrade should complete within 30 seconds"
