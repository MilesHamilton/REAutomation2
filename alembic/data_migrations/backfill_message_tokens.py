"""
Backfill Message Token Counts

This script backfills the token_count field in conversation_history
based on message content length.

Usage:
    python -m alembic.data_migrations.backfill_message_tokens
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
from src.database.models import ConversationHistory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageTokenBackfiller:
    """Backfill token counts for conversation history messages."""

    def __init__(self, database_url: str, batch_size: int = 500):
        """
        Initialize backfiller.

        Args:
            database_url: Database connection URL
            batch_size: Number of messages to process per batch
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

    def estimate_tokens(self, content: str) -> int:
        """
        Estimate token count for message content.

        Uses a simple heuristic: ~4 characters per token for English text.
        This matches OpenAI's rough estimation and is adequate for backfilling.

        Args:
            content: Message content

        Returns:
            Estimated token count
        """
        if not content:
            return 0

        # Basic estimation: 4 chars per token
        char_count = len(content)
        estimated_tokens = max(1, char_count // 4)

        return estimated_tokens

    async def count_messages_to_backfill(self) -> int:
        """
        Count messages that need backfilling.

        Returns:
            Number of messages with zero token count
        """
        async with self.SessionLocal() as session:
            result = await session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM conversation_history
                    WHERE token_count = 0
                      AND content IS NOT NULL
                      AND content != ''
                """)
            )
            count = result.scalar()
            return count

    async def backfill_message_batch(self, offset: int) -> int:
        """
        Backfill a batch of messages.

        Args:
            offset: Offset for batch query

        Returns:
            Number of messages updated
        """
        async with self.SessionLocal() as session:
            # Get batch of messages to backfill
            result = await session.execute(
                text(f"""
                    SELECT id, content
                    FROM conversation_history
                    WHERE token_count = 0
                      AND content IS NOT NULL
                      AND content != ''
                    ORDER BY id
                    LIMIT {self.batch_size}
                    OFFSET {offset}
                """)
            )
            messages = [(row[0], row[1]) for row in result.fetchall()]

            if not messages:
                return 0

            # Calculate token counts
            updates = []
            for msg_id, content in messages:
                token_count = self.estimate_tokens(content)
                updates.append({
                    "id": msg_id,
                    "token_count": token_count
                })

            # Bulk update
            if updates:
                for update_data in updates:
                    await session.execute(
                        update(ConversationHistory)
                        .where(ConversationHistory.id == update_data["id"])
                        .values(token_count=update_data["token_count"])
                    )

                await session.commit()
                logger.info(f"Updated {len(updates)} messages (batch offset {offset})")

            return len(updates)

    async def run(self):
        """Run backfill process."""
        try:
            await self.setup()

            # Count messages to backfill
            total_messages = await self.count_messages_to_backfill()
            logger.info(f"Found {total_messages} messages to backfill")

            if total_messages == 0:
                logger.info("No messages need backfilling")
                return

            # Process in batches
            offset = 0
            updated_count = 0

            while offset < total_messages:
                batch_count = await self.backfill_message_batch(offset)
                if batch_count == 0:
                    break

                updated_count += batch_count
                offset += self.batch_size

                # Progress report
                progress = min(100, (updated_count / total_messages) * 100)
                logger.info(f"Progress: {updated_count}/{total_messages} ({progress:.1f}%)")

            logger.info(f"Backfill complete! Updated {updated_count} messages")

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
    backfiller = MessageTokenBackfiller(database_url, batch_size=500)
    await backfiller.run()


if __name__ == "__main__":
    asyncio.run(main())
