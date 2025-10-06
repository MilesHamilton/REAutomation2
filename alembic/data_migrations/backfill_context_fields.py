"""
Backfill Context Management Fields

This script backfills the context_pruned, pruning_count, and total_context_tokens
fields for existing calls based on their conversation history.

Usage:
    python -m alembic.data_migrations.backfill_context_fields
"""
import asyncio
import logging
from typing import Optional
from sqlalchemy import text, select, update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import settings
from src.database.models import Call, ConversationHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextFieldBackfiller:
    """Backfill context management fields for existing calls."""

    def __init__(self, database_url: str, batch_size: int = 100):
        """
        Initialize backfiller.

        Args:
            database_url: Database connection URL
            batch_size: Number of calls to process per batch
        """
        self.database_url = database_url
        self.batch_size = batch_size
        self.engine = None
        self.SessionLocal = None

    async def setup(self):
        """Set up database connection."""
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        logger.info("Database connection established")

    async def teardown(self):
        """Clean up database connection."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")

    async def count_calls_to_backfill(self) -> int:
        """
        Count calls that need backfilling.

        Returns:
            Number of calls with default context field values
        """
        async with self.SessionLocal() as session:
            result = await session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM calls
                    WHERE context_pruned = false
                      AND pruning_count = 0
                      AND total_context_tokens = 0
                      AND created_at < NOW() - INTERVAL '1 hour'
                """)
            )
            count = result.scalar()
            return count

    async def estimate_context_metrics(self, call_id: int, session: AsyncSession) -> dict:
        """
        Estimate context metrics for a call based on conversation history.

        Args:
            call_id: Call ID
            session: Database session

        Returns:
            Dictionary with estimated metrics
        """
        # Get conversation history
        result = await session.execute(
            select(ConversationHistory)
            .where(ConversationHistory.call_id == call_id)
            .order_by(ConversationHistory.message_order)
        )
        messages = result.scalars().all()

        if not messages:
            return {
                "context_pruned": False,
                "pruning_count": 0,
                "total_context_tokens": 0
            }

        # Estimate tokens (rough: 4 chars per token)
        total_tokens = 0
        for msg in messages:
            content_length = len(msg.content) if msg.content else 0
            estimated_tokens = content_length // 4
            total_tokens += estimated_tokens

        # Estimate if context was pruned
        # Heuristic: If there are many messages (>20) and high token count (>8000),
        # likely context was pruned
        context_pruned = len(messages) > 20 and total_tokens > 8000

        # Estimate pruning count
        # Heuristic: One prune per 8000 tokens over the 8000 threshold
        pruning_count = 0
        if context_pruned:
            pruning_count = max(1, (total_tokens - 8000) // 8000)

        return {
            "context_pruned": context_pruned,
            "pruning_count": pruning_count,
            "total_context_tokens": total_tokens
        }

    async def backfill_call_batch(self, offset: int) -> int:
        """
        Backfill a batch of calls.

        Args:
            offset: Offset for batch query

        Returns:
            Number of calls updated
        """
        async with self.SessionLocal() as session:
            # Get batch of calls to backfill
            result = await session.execute(
                text(f"""
                    SELECT id
                    FROM calls
                    WHERE context_pruned = false
                      AND pruning_count = 0
                      AND total_context_tokens = 0
                      AND created_at < NOW() - INTERVAL '1 hour'
                    ORDER BY id
                    LIMIT {self.batch_size}
                    OFFSET {offset}
                """)
            )
            call_ids = [row[0] for row in result.fetchall()]

            if not call_ids:
                return 0

            # Process each call
            updates = []
            for call_id in call_ids:
                metrics = await self.estimate_context_metrics(call_id, session)
                updates.append({
                    "id": call_id,
                    **metrics
                })

            # Bulk update
            if updates:
                for update_data in updates:
                    await session.execute(
                        update(Call)
                        .where(Call.id == update_data["id"])
                        .values(
                            context_pruned=update_data["context_pruned"],
                            pruning_count=update_data["pruning_count"],
                            total_context_tokens=update_data["total_context_tokens"]
                        )
                    )

                await session.commit()
                logger.info(f"Updated {len(updates)} calls (batch offset {offset})")

            return len(updates)

    async def run(self):
        """Run backfill process."""
        try:
            await self.setup()

            # Count calls to backfill
            total_calls = await self.count_calls_to_backfill()
            logger.info(f"Found {total_calls} calls to backfill")

            if total_calls == 0:
                logger.info("No calls need backfilling")
                return

            # Process in batches
            offset = 0
            updated_count = 0

            while offset < total_calls:
                batch_count = await self.backfill_call_batch(offset)
                if batch_count == 0:
                    break

                updated_count += batch_count
                offset += self.batch_size

                # Progress report
                progress = min(100, (updated_count / total_calls) * 100)
                logger.info(f"Progress: {updated_count}/{total_calls} ({progress:.1f}%)")

            logger.info(f"Backfill complete! Updated {updated_count} calls")

        except Exception as e:
            logger.error(f"Backfill failed: {e}", exc_info=True)
            raise
        finally:
            await self.teardown()


async def main():
    """Main entry point."""
    # Get database URL from environment or settings
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        database_url = f"postgresql+asyncpg://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"

    # Run backfiller
    backfiller = ContextFieldBackfiller(database_url, batch_size=100)
    await backfiller.run()


if __name__ == "__main__":
    asyncio.run(main())
