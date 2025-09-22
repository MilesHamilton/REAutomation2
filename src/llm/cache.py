import hashlib
import json
import time
import logging
from typing import Optional, Dict, Any, Union
import asyncio
from dataclasses import dataclass

import aioredis
from aioredis import Redis

from ..config import settings
from .models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    response: LLMResponse
    timestamp: float
    access_count: int = 0
    ttl: int = 3600  # 1 hour default


class LLMCache:
    """
    Redis-based caching system for LLM responses
    """

    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        self.redis_url = redis_url or settings.redis_url
        self.default_ttl = default_ttl
        self.redis: Optional[Redis] = None
        self.local_cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "evictions": 0
        }
        self.max_local_entries = 1000
        self.is_connected = False

    async def connect(self) -> bool:
        """Connect to Redis cache"""
        try:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False  # Handle binary data
            )

            # Test connection
            await self.redis.ping()
            self.is_connected = True
            logger.info("Connected to Redis cache")
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using local cache only.")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
        self.is_connected = False
        logger.info("Disconnected from Redis cache")

    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for LLM request"""
        # Create deterministic hash based on request parameters
        key_data = {
            "messages": [{"role": msg.role.value, "content": msg.content} for msg in request.messages],
            "system_prompt": request.system_prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "structured_output": request.structured_output,
            "response_format": request.response_format
        }

        key_json = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.sha256(key_json.encode()).hexdigest()
        return f"llm_cache:{cache_key}"

    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Get cached response for LLM request"""
        cache_key = self._generate_cache_key(request)

        try:
            # Try local cache first (fastest)
            if cache_key in self.local_cache:
                entry = self.local_cache[cache_key]

                # Check if entry is still valid
                if time.time() - entry.timestamp < entry.ttl:
                    entry.access_count += 1
                    self.cache_stats["hits"] += 1
                    logger.debug(f"Local cache hit for key: {cache_key[:8]}...")
                    return entry.response
                else:
                    # Expired, remove from local cache
                    del self.local_cache[cache_key]

            # Try Redis cache
            if self.is_connected and self.redis:
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    try:
                        # Deserialize cached response
                        response_data = json.loads(cached_data)
                        response = LLMResponse(**response_data)

                        # Store in local cache for faster future access
                        self._store_local(cache_key, response)

                        self.cache_stats["hits"] += 1
                        logger.debug(f"Redis cache hit for key: {cache_key[:8]}...")
                        return response

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to deserialize cached response: {e}")
                        # Remove corrupted entry
                        await self.redis.delete(cache_key)

            self.cache_stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats["errors"] += 1
            return None

    async def set(
        self,
        request: LLMRequest,
        response: LLMResponse,
        ttl: int = None
    ) -> bool:
        """Cache LLM response"""
        cache_key = self._generate_cache_key(request)
        ttl = ttl or self.default_ttl

        try:
            # Store in local cache
            self._store_local(cache_key, response, ttl)

            # Store in Redis if connected
            if self.is_connected and self.redis:
                # Serialize response
                response_data = response.dict()
                cached_data = json.dumps(response_data)

                await self.redis.setex(cache_key, ttl, cached_data)
                logger.debug(f"Cached response in Redis for key: {cache_key[:8]}...")

            self.cache_stats["sets"] += 1
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.cache_stats["errors"] += 1
            return False

    def _store_local(self, cache_key: str, response: LLMResponse, ttl: int = None) -> None:
        """Store response in local cache"""
        ttl = ttl or self.default_ttl

        # Evict oldest entries if cache is full
        if len(self.local_cache) >= self.max_local_entries:
            self._evict_oldest()

        entry = CacheEntry(
            response=response,
            timestamp=time.time(),
            ttl=ttl
        )

        self.local_cache[cache_key] = entry
        logger.debug(f"Cached response locally for key: {cache_key[:8]}...")

    def _evict_oldest(self):
        """Evict oldest entries from local cache"""
        if not self.local_cache:
            return

        # Sort by timestamp (oldest first)
        sorted_entries = sorted(
            self.local_cache.items(),
            key=lambda x: x[1].timestamp
        )

        # Remove oldest 20% of entries
        num_to_evict = max(1, len(sorted_entries) // 5)
        for i in range(num_to_evict):
            key, _ = sorted_entries[i]
            del self.local_cache[key]
            self.cache_stats["evictions"] += 1

        logger.debug(f"Evicted {num_to_evict} entries from local cache")

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        invalidated_count = 0

        try:
            # Clear local cache entries matching pattern
            local_keys_to_remove = []
            for key in self.local_cache.keys():
                if pattern in key:
                    local_keys_to_remove.append(key)

            for key in local_keys_to_remove:
                del self.local_cache[key]
                invalidated_count += 1

            # Clear Redis cache entries if connected
            if self.is_connected and self.redis:
                # Use SCAN to find matching keys (more memory efficient than KEYS)
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(
                        cursor,
                        match=f"*{pattern}*",
                        count=100
                    )

                    if keys:
                        await self.redis.delete(*keys)
                        invalidated_count += len(keys)

                    if cursor == 0:
                        break

            logger.info(f"Invalidated {invalidated_count} cache entries matching pattern: {pattern}")
            return invalidated_count

        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    async def clear(self) -> bool:
        """Clear all cached responses"""
        try:
            # Clear local cache
            self.local_cache.clear()

            # Clear Redis cache
            if self.is_connected and self.redis:
                await self.redis.flushdb()

            logger.info("Cache cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(1, total_requests)

        return {
            "cache_stats": self.cache_stats.copy(),
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache),
            "redis_connected": self.is_connected,
            "total_requests": total_requests
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check cache health status"""
        health_status = {
            "local_cache": {
                "status": "healthy",
                "entries": len(self.local_cache),
                "max_entries": self.max_local_entries
            },
            "redis_cache": {
                "status": "unknown",
                "connected": self.is_connected
            }
        }

        # Check Redis connection
        if self.is_connected and self.redis:
            try:
                await self.redis.ping()
                health_status["redis_cache"]["status"] = "healthy"

                # Get Redis info
                info = await self.redis.info("memory")
                health_status["redis_cache"]["memory_usage"] = info.get("used_memory_human", "unknown")
                health_status["redis_cache"]["keys"] = await self.redis.dbsize()

            except Exception as e:
                health_status["redis_cache"]["status"] = "unhealthy"
                health_status["redis_cache"]["error"] = str(e)
        else:
            health_status["redis_cache"]["status"] = "disconnected"

        return health_status


# Global cache instance
llm_cache = LLMCache()


class CacheStrategy:
    """
    Define caching strategies for different types of requests
    """

    @staticmethod
    def get_ttl_for_request(request: LLMRequest) -> int:
        """Determine TTL based on request type"""
        # Structured outputs (like qualification analysis) cache longer
        if request.structured_output:
            return 7200  # 2 hours

        # High temperature responses (creative) cache for shorter time
        if request.temperature > 0.8:
            return 1800  # 30 minutes

        # Low temperature responses (factual) cache longer
        if request.temperature < 0.3:
            return 3600  # 1 hour

        # Default TTL
        return 3600  # 1 hour

    @staticmethod
    def should_cache(request: LLMRequest, response: LLMResponse) -> bool:
        """Determine if response should be cached"""
        # Don't cache very short responses (likely errors)
        if len(response.content) < 10:
            return False

        # Don't cache responses that took too long (likely timeouts)
        if response.response_time_ms > 30000:  # 30 seconds
            return False

        # Don't cache high-temperature creative responses
        if request.temperature > 0.9:
            return False

        return True


async def setup_cache() -> bool:
    """Initialize the global cache"""
    return await llm_cache.connect()


async def cleanup_cache():
    """Cleanup the global cache"""
    await llm_cache.disconnect()