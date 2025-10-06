"""Response caching for voice-agent integration"""

import logging
import time
import hashlib
from typing import Optional, Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ResponseCache:
    """LRU cache for agent responses to improve performance"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize response cache

        Args:
            max_size: Maximum number of cached responses
            ttl_seconds: Time-to-live for cached responses in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def _generate_key(self, call_id: str, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from input parameters"""
        # Create a deterministic key from the inputs
        key_parts = [call_id, user_input.lower().strip()]

        if context:
            # Include relevant context in key
            if "workflow_state" in context:
                key_parts.append(str(context["workflow_state"]))
            if "current_agent" in context:
                key_parts.append(str(context["current_agent"]))

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self,
        call_id: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get cached response if available and not expired

        Args:
            call_id: Call identifier
            user_input: User input text
            context: Optional context for cache key generation

        Returns:
            Cached response text or None if not found/expired
        """
        try:
            key = self._generate_key(call_id, user_input, context)

            if key not in self.cache:
                self.misses += 1
                return None

            # Check if expired
            metadata = self.metadata.get(key, {})
            cached_time = metadata.get("timestamp", 0)

            if time.time() - cached_time > self.ttl_seconds:
                # Expired, remove from cache
                del self.cache[key]
                del self.metadata[key]
                self.misses += 1
                logger.debug(f"Cache entry expired for key {key[:8]}...")
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1

            response = self.cache[key]
            logger.debug(f"Cache hit for key {key[:8]}...")

            return response

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(
        self,
        call_id: str,
        user_input: str,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Store response in cache

        Args:
            call_id: Call identifier
            user_input: User input text
            response: Agent response to cache
            context: Optional context for cache key generation
        """
        try:
            key = self._generate_key(call_id, user_input, context)

            # Remove oldest item if cache is full
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.metadata[oldest_key]
                logger.debug(f"Cache full, removed oldest entry: {oldest_key[:8]}...")

            # Add to cache
            self.cache[key] = response
            self.metadata[key] = {
                "timestamp": time.time(),
                "call_id": call_id,
                "input_length": len(user_input),
                "response_length": len(response)
            }

            logger.debug(f"Cached response for key {key[:8]}...")

        except Exception as e:
            logger.error(f"Error storing in cache: {e}")

    def invalidate(self, call_id: Optional[str] = None):
        """
        Invalidate cache entries

        Args:
            call_id: If provided, invalidate only entries for this call.
                    If None, clear entire cache.
        """
        try:
            if call_id is None:
                # Clear entire cache
                self.cache.clear()
                self.metadata.clear()
                logger.info("Entire cache cleared")
            else:
                # Remove entries for specific call
                keys_to_remove = [
                    key for key, meta in self.metadata.items()
                    if meta.get("call_id") == call_id
                ]

                for key in keys_to_remove:
                    del self.cache[key]
                    del self.metadata[key]

                logger.info(f"Invalidated {len(keys_to_remove)} cache entries for call {call_id}")

        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate_pct": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }

    def cleanup_expired(self):
        """Remove all expired entries from cache"""
        try:
            current_time = time.time()
            keys_to_remove = []

            for key, metadata in self.metadata.items():
                cached_time = metadata.get("timestamp", 0)
                if current_time - cached_time > self.ttl_seconds:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.cache[key]
                del self.metadata[key]

            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} expired cache entries")

        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")


# Global response cache instance
response_cache = ResponseCache(max_size=1000, ttl_seconds=300)
