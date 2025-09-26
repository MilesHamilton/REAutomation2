import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool

from ..config import settings
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session management"""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.engine: Optional[AsyncEngine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._is_initialized = False

    async def initialize(self) -> bool:
        """Initialize database engine and session factory"""
        try:
            # Parse database URL to determine connection parameters
            if "sqlite" in self.database_url:
                # SQLite configuration for development
                self.engine = create_async_engine(
                    self.database_url,
                    poolclass=StaticPool,
                    connect_args={
                        "check_same_thread": False,
                    },
                    echo=settings.debug,
                )
            else:
                # PostgreSQL configuration for production
                self.engine = create_async_engine(
                    self.database_url,
                    poolclass=QueuePool,
                    pool_size=20,  # Max connections in pool
                    max_overflow=10,  # Additional connections beyond pool_size
                    pool_timeout=30,  # Timeout when getting connection from pool
                    pool_recycle=3600,  # Recreate connections after 1 hour
                    pool_pre_ping=True,  # Validate connections before use
                    echo=settings.debug,
                )

            # Create async session factory
            self.SessionLocal = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection
            async with self.engine.begin() as conn:
                await conn.run_sync(lambda _: None)  # Simple connection test

            self._is_initialized = True
            logger.info(f"Database initialized successfully: {self._get_safe_url()}")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False

    def _get_safe_url(self) -> str:
        """Get database URL with credentials masked for logging"""
        if not self.database_url:
            return "None"

        # Mask password in URL for logging
        import re
        masked_url = re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', self.database_url)
        return masked_url

    async def create_tables(self) -> bool:
        """Create all database tables"""
        try:
            if not self.engine:
                raise RuntimeError("Database not initialized")

            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database tables created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

    async def drop_tables(self) -> bool:
        """Drop all database tables (use with caution!)"""
        try:
            if not self.engine:
                raise RuntimeError("Database not initialized")

            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)

            logger.warning("All database tables dropped")
            return True

        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            return False

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self._is_initialized:
            raise RuntimeError("Database not initialized")

        async with self.SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def get_session_sync(self) -> AsyncSession:
        """Get database session for dependency injection"""
        if not self._is_initialized:
            raise RuntimeError("Database not initialized")

        return self.SessionLocal()

    async def health_check(self) -> dict:
        """Check database health"""
        try:
            if not self._is_initialized:
                return {
                    "status": "unhealthy",
                    "error": "Database not initialized"
                }

            # Test connection with simple query
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                await result.fetchone()

            return {
                "status": "healthy",
                "database_url": self._get_safe_url(),
                "pool_size": self.engine.pool.size() if self.engine else 0,
                "checked_out": self.engine.pool.checkedout() if self.engine else 0,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_url": self._get_safe_url()
            }

    async def cleanup(self):
        """Clean up database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
                self.engine = None

            self.SessionLocal = None
            self._is_initialized = False
            logger.info("Database cleanup complete")

        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")

    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._is_initialized

    async def execute_raw_sql(self, sql: str, parameters: dict = None) -> list:
        """Execute raw SQL query (use with caution)"""
        try:
            async with self.get_session() as session:
                result = await session.execute(sql, parameters or {})
                return result.fetchall()

        except Exception as e:
            logger.error(f"Raw SQL execution failed: {e}")
            raise

    async def get_connection_info(self) -> dict:
        """Get connection pool information"""
        if not self.engine:
            return {"status": "not_initialized"}

        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
        }


# Global database manager instance
db_manager = DatabaseManager()


# Dependency for FastAPI
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions"""
    async with db_manager.get_session() as session:
        yield session