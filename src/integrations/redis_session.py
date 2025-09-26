"""
Redis Session Management Service

This module provides comprehensive session management using Redis,
including conversation state persistence, session cleanup, and
distributed session storage.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import os

import redis.asyncio as redis
from redis.asyncio import Redis
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base exception for session management errors"""
    pass


class SessionNotFoundError(SessionError):
    """Exception raised when session is not found"""
    pass


class SessionExpiredError(SessionError):
    """Exception raised when session has expired"""
    pass


class ConversationState(BaseModel):
    """Conversation state data"""
    call_id: str
    phone_number: str
    current_agent: str
    workflow_state: str
    conversation_history: List[Dict[str, Any]] = []
    lead_data: Dict[str, Any] = {}
    qualification_score: float = 0.0
    context_data: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


@dataclass
class SessionConfig:
    """Session configuration"""
    default_ttl: int = 3600  # 1 hour
    max_ttl: int = 86400     # 24 hours
    cleanup_interval: int = 300  # 5 minutes
    max_memory_usage: int = 1024 * 1024 * 100  # 100MB
    compression_enabled: bool = True
    persistence_enabled: bool = True


class RedisSessionManager:
    """
    Redis-based session management for conversation state and persistence

    Handles session lifecycle, cleanup, and distributed storage with
    automatic expiration and memory management.
    """

    def __init__(
        self,
        redis_url: str = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str = None,
        config: Optional[SessionConfig] = None
    ):
        # Redis connection parameters
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", redis_port))
        self.redis_db = int(os.getenv("REDIS_DB", redis_db))
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD")

        # Session configuration
        self.config = config or SessionConfig()

        # Redis client
        self._redis: Optional[Redis] = None

        # Key prefixes for different data types
        self.SESSION_PREFIX = "session:"
        self.CONVERSATION_PREFIX = "conv:"
        self.TEMP_PREFIX = "temp:"
        self.METADATA_PREFIX = "meta:"

        # Statistics
        self.stats = {
            "sessions_created": 0,
            "sessions_retrieved": 0,
            "sessions_updated": 0,
            "sessions_expired": 0,
            "sessions_cleaned": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize Redis connection and start background tasks"""
        try:
            if self.redis_url:
                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False  # We'll handle encoding manually for binary data
                )
            else:
                self._redis = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    password=self.redis_password,
                    encoding="utf-8",
                    decode_responses=False,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    health_check_interval=30
                )

            # Test connection
            await self._redis.ping()

            # Start background cleanup task
            if self.config.cleanup_interval > 0:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("Redis session manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Redis session manager: {e}")
            raise SessionError(f"Redis initialization failed: {e}")

    async def cleanup(self) -> None:
        """Clean up resources and connections"""
        self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._redis:
            await self._redis.close()
            self._redis = None

        logger.info("Redis session manager cleaned up")

    async def create_session(
        self,
        call_id: str,
        phone_number: str,
        initial_data: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> ConversationState:
        """
        Create a new conversation session

        Args:
            call_id: Unique call identifier
            phone_number: Phone number for the session
            initial_data: Initial session data
            ttl: Time-to-live in seconds

        Returns:
            ConversationState object

        Raises:
            SessionError: If session creation fails
        """
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        ttl = ttl or self.config.default_ttl
        ttl = min(ttl, self.config.max_ttl)

        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=ttl)

        # Create conversation state
        session_state = ConversationState(
            call_id=call_id,
            phone_number=phone_number,
            current_agent="conversation",
            workflow_state="greeting",
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            lead_data=initial_data or {},
            metadata={"ttl": ttl}
        )

        try:
            # Store session data
            session_key = f"{self.SESSION_PREFIX}{call_id}"
            session_data = self._serialize_session(session_state)

            await self._redis.setex(session_key, ttl, session_data)

            # Create session index
            await self._add_to_session_index(call_id, phone_number)

            self.stats["sessions_created"] += 1
            logger.info(f"Created session for call {call_id} with TTL {ttl}s")

            return session_state

        except Exception as e:
            logger.error(f"Failed to create session {call_id}: {e}")
            raise SessionError(f"Session creation failed: {e}")

    async def get_session(self, call_id: str) -> Optional[ConversationState]:
        """
        Retrieve session by call ID

        Args:
            call_id: Call identifier

        Returns:
            ConversationState if found, None otherwise

        Raises:
            SessionExpiredError: If session has expired
        """
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            session_key = f"{self.SESSION_PREFIX}{call_id}"
            session_data = await self._redis.get(session_key)

            if session_data is None:
                self.stats["cache_misses"] += 1
                logger.debug(f"Session not found: {call_id}")
                return None

            session_state = self._deserialize_session(session_data)

            # Check if session has expired (double-check beyond Redis TTL)
            if session_state.expires_at and datetime.utcnow() > session_state.expires_at:
                await self.delete_session(call_id)
                self.stats["sessions_expired"] += 1
                raise SessionExpiredError(f"Session {call_id} has expired")

            self.stats["cache_hits"] += 1
            self.stats["sessions_retrieved"] += 1

            return session_state

        except (json.JSONDecodeError, pickle.PickleError) as e:
            logger.error(f"Failed to deserialize session {call_id}: {e}")
            # Clean up corrupted session
            await self.delete_session(call_id)
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve session {call_id}: {e}")
            raise SessionError(f"Session retrieval failed: {e}")

    async def update_session(
        self,
        call_id: str,
        updates: Dict[str, Any],
        extend_ttl: bool = True
    ) -> bool:
        """
        Update session data

        Args:
            call_id: Call identifier
            updates: Dictionary of updates to apply
            extend_ttl: Whether to extend session TTL

        Returns:
            True if session was updated, False if not found

        Raises:
            SessionError: If update fails
        """
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            session_state = await self.get_session(call_id)
            if not session_state:
                return False

            # Apply updates
            for key, value in updates.items():
                if hasattr(session_state, key):
                    setattr(session_state, key, value)
                elif key in ["conversation_history", "lead_data", "context_data", "metadata"]:
                    # Handle nested dictionary updates
                    current_dict = getattr(session_state, key, {})
                    if isinstance(value, dict) and isinstance(current_dict, dict):
                        current_dict.update(value)
                    else:
                        setattr(session_state, key, value)

            # Update timestamp
            session_state.updated_at = datetime.utcnow()

            # Calculate new TTL
            session_key = f"{self.SESSION_PREFIX}{call_id}"
            current_ttl = await self._redis.ttl(session_key)

            if extend_ttl:
                new_ttl = max(current_ttl, self.config.default_ttl)
                session_state.expires_at = datetime.utcnow() + timedelta(seconds=new_ttl)
            else:
                new_ttl = max(current_ttl, 0)

            # Store updated session
            session_data = self._serialize_session(session_state)
            await self._redis.setex(session_key, new_ttl, session_data)

            self.stats["sessions_updated"] += 1
            logger.debug(f"Updated session {call_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to update session {call_id}: {e}")
            raise SessionError(f"Session update failed: {e}")

    async def delete_session(self, call_id: str) -> bool:
        """
        Delete session and related data

        Args:
            call_id: Call identifier

        Returns:
            True if session was deleted, False if not found
        """
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            # Delete main session data
            session_key = f"{self.SESSION_PREFIX}{call_id}"
            result = await self._redis.delete(session_key)

            # Delete related conversation data
            conversation_key = f"{self.CONVERSATION_PREFIX}{call_id}"
            await self._redis.delete(conversation_key)

            # Delete temporary data
            temp_key = f"{self.TEMP_PREFIX}{call_id}"
            await self._redis.delete(temp_key)

            # Remove from session index
            await self._remove_from_session_index(call_id)

            if result > 0:
                logger.debug(f"Deleted session {call_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to delete session {call_id}: {e}")
            raise SessionError(f"Session deletion failed: {e}")

    async def extend_session_ttl(self, call_id: str, additional_seconds: int) -> bool:
        """
        Extend session TTL

        Args:
            call_id: Call identifier
            additional_seconds: Additional seconds to add to TTL

        Returns:
            True if TTL was extended, False if session not found
        """
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            session_key = f"{self.SESSION_PREFIX}{call_id}"
            current_ttl = await self._redis.ttl(session_key)

            if current_ttl == -2:  # Key doesn't exist
                return False

            new_ttl = min(current_ttl + additional_seconds, self.config.max_ttl)
            await self._redis.expire(session_key, new_ttl)

            # Update expires_at in session data
            session_state = await self.get_session(call_id)
            if session_state:
                session_state.expires_at = datetime.utcnow() + timedelta(seconds=new_ttl)
                await self.update_session(call_id, {"expires_at": session_state.expires_at}, False)

            logger.debug(f"Extended TTL for session {call_id} to {new_ttl}s")
            return True

        except Exception as e:
            logger.error(f"Failed to extend TTL for session {call_id}: {e}")
            return False

    async def get_active_sessions(self) -> List[str]:
        """Get list of active session call IDs"""
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            pattern = f"{self.SESSION_PREFIX}*"
            keys = await self._redis.keys(pattern)
            call_ids = [key.decode('utf-8').replace(self.SESSION_PREFIX, '') for key in keys]
            return call_ids

        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []

    async def get_session_by_phone(self, phone_number: str) -> Optional[ConversationState]:
        """Get most recent session for a phone number"""
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            # Search through session index
            index_key = f"phone_index:{phone_number}"
            call_ids = await self._redis.lrange(index_key, 0, -1)

            for call_id_bytes in call_ids:
                call_id = call_id_bytes.decode('utf-8')
                session = await self.get_session(call_id)
                if session:
                    return session

            return None

        except Exception as e:
            logger.error(f"Failed to get session by phone {phone_number}: {e}")
            return None

    async def store_temporary_data(
        self,
        call_id: str,
        key: str,
        data: Any,
        ttl: int = 300
    ) -> bool:
        """
        Store temporary data related to a session

        Args:
            call_id: Call identifier
            key: Data key
            data: Data to store
            ttl: Time-to-live in seconds

        Returns:
            True if data was stored successfully
        """
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            temp_key = f"{self.TEMP_PREFIX}{call_id}:{key}"
            serialized_data = json.dumps(data) if not isinstance(data, bytes) else data

            await self._redis.setex(temp_key, ttl, serialized_data)
            logger.debug(f"Stored temporary data {key} for call {call_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store temporary data for {call_id}: {e}")
            return False

    async def get_temporary_data(self, call_id: str, key: str) -> Optional[Any]:
        """Retrieve temporary data"""
        if not self._redis:
            raise SessionError("Redis session manager not initialized")

        try:
            temp_key = f"{self.TEMP_PREFIX}{call_id}:{key}"
            data = await self._redis.get(temp_key)

            if data is None:
                return None

            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return data

        except Exception as e:
            logger.error(f"Failed to get temporary data {key} for {call_id}: {e}")
            return None

    async def get_session_stats(self) -> Dict[str, Any]:
        """Get session management statistics"""
        if not self._redis:
            return {"error": "Redis not initialized"}

        try:
            # Get Redis info
            redis_info = await self._redis.info()

            # Get active session count
            active_sessions = len(await self.get_active_sessions())

            # Memory usage
            used_memory = redis_info.get('used_memory', 0)
            used_memory_human = redis_info.get('used_memory_human', '0B')

            return {
                **self.stats,
                "active_sessions": active_sessions,
                "redis_memory_used": used_memory,
                "redis_memory_human": used_memory_human,
                "redis_connected_clients": redis_info.get('connected_clients', 0),
                "redis_uptime_seconds": redis_info.get('uptime_in_seconds', 0),
                "cache_hit_rate": (
                    self.stats["cache_hits"] /
                    (self.stats["cache_hits"] + self.stats["cache_misses"])
                    if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0
                )
            }

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            if not self._redis:
                return {"status": "unhealthy", "error": "Redis not initialized"}

            # Test basic operations
            test_key = "health_check_test"
            await self._redis.set(test_key, "test", ex=10)
            result = await self._redis.get(test_key)
            await self._redis.delete(test_key)

            if result != b"test":
                return {"status": "unhealthy", "error": "Redis operations failed"}

            stats = await self.get_session_stats()

            return {
                "status": "healthy",
                "redis_connected": True,
                "stats": stats
            }

        except Exception as e:
            logger.error(f"Session manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e)
            }

    def _serialize_session(self, session_state: ConversationState) -> bytes:
        """Serialize session state for storage"""
        if self.config.compression_enabled:
            # Use pickle for better compression of complex objects
            return pickle.dumps(session_state.dict())
        else:
            return json.dumps(session_state.dict(), default=str).encode('utf-8')

    def _deserialize_session(self, data: bytes) -> ConversationState:
        """Deserialize session state from storage"""
        try:
            if self.config.compression_enabled:
                session_dict = pickle.loads(data)
            else:
                session_dict = json.loads(data.decode('utf-8'))

            # Convert datetime strings back to datetime objects
            for field in ['created_at', 'updated_at', 'expires_at']:
                if field in session_dict and isinstance(session_dict[field], str):
                    session_dict[field] = datetime.fromisoformat(session_dict[field])

            return ConversationState(**session_dict)

        except Exception as e:
            logger.error(f"Failed to deserialize session data: {e}")
            raise

    async def _add_to_session_index(self, call_id: str, phone_number: str) -> None:
        """Add session to phone number index"""
        try:
            index_key = f"phone_index:{phone_number}"
            await self._redis.lpush(index_key, call_id)
            await self._redis.expire(index_key, self.config.max_ttl)
        except Exception as e:
            logger.error(f"Failed to add to session index: {e}")

    async def _remove_from_session_index(self, call_id: str) -> None:
        """Remove session from indexes"""
        try:
            # We'd need to search through phone indexes, but for efficiency
            # we can let them expire naturally
            pass
        except Exception as e:
            logger.error(f"Failed to remove from session index: {e}")

    async def _cleanup_loop(self) -> None:
        """Background cleanup task"""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_cleanup()
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.cleanup_interval
                )
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue cleanup cycle
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _perform_cleanup(self) -> None:
        """Perform cleanup of expired sessions and optimize memory"""
        try:
            # Clean up expired temporary data
            temp_pattern = f"{self.TEMP_PREFIX}*"
            temp_keys = await self._redis.keys(temp_pattern)

            expired_count = 0
            for key in temp_keys:
                ttl = await self._redis.ttl(key)
                if ttl == -2:  # Key expired
                    expired_count += 1

            if expired_count > 0:
                self.stats["sessions_cleaned"] += expired_count

            # Clean up old phone indexes
            phone_pattern = "phone_index:*"
            phone_keys = await self._redis.keys(phone_pattern)

            for key in phone_keys[:10]:  # Limit to avoid blocking
                await self._redis.expire(key, self.config.max_ttl)

            if expired_count > 0:
                logger.debug(f"Cleaned up {expired_count} expired items")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    # Batch operations for performance
    async def batch_update_sessions(
        self,
        updates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Update multiple sessions in batch"""
        results = {}

        for call_id, session_updates in updates.items():
            try:
                result = await self.update_session(call_id, session_updates)
                results[call_id] = result
            except Exception as e:
                logger.error(f"Failed to update session {call_id} in batch: {e}")
                results[call_id] = False

        return results