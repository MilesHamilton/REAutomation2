"""
ElevenLabs API Integration Service

This module provides integration with ElevenLabs premium TTS service,
including voice selection, API rate limiting, and fallback mechanisms.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import json

import aiohttp
import backoff
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ElevenLabsError(Exception):
    """Base exception for ElevenLabs API errors"""
    pass


class RateLimitError(ElevenLabsError):
    """Exception raised when API rate limit is exceeded"""
    pass


class QuotaExceededError(ElevenLabsError):
    """Exception raised when API quota is exceeded"""
    pass


class VoiceModel(BaseModel):
    """ElevenLabs voice model"""
    voice_id: str
    name: str
    preview_url: Optional[str] = None
    category: str = "generated"
    description: Optional[str] = None
    labels: Dict[str, Any] = {}
    samples: List[Dict[str, Any]] = []
    settings: Optional[Dict[str, float]] = None


@dataclass
class TTSRequest:
    """Text-to-speech request parameters"""
    text: str
    voice_id: str
    model_id: str = "eleven_monolingual_v1"
    voice_settings: Optional[Dict[str, float]] = None
    pronunciation_dictionary_locators: Optional[List[Dict[str, str]]] = None


@dataclass
class TTSResponse:
    """Text-to-speech response"""
    audio_data: bytes
    character_count: int
    cost_estimate: float
    processing_time: float
    voice_id: str
    model_id: str


class RateLimiter:
    """Rate limiting for ElevenLabs API"""

    def __init__(self, max_requests_per_minute: int = 120):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests: List[float] = []
        self._lock = asyncio.Lock()

    async def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        async with self._lock:
            now = time.time()

            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            # Check if we need to wait
            if len(self.requests) >= self.max_requests_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached, waiting {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Remove the old request
                    self.requests.pop(0)

            # Add current request
            self.requests.append(now)


class ElevenLabsService:
    """
    ElevenLabs API integration service for premium TTS

    Handles voice synthesis, voice management, and API optimization
    with fallback mechanisms and rate limiting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.elevenlabs.io/v1",
        max_requests_per_minute: int = 120,
        default_voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        timeout: float = 30.0
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.base_url = base_url.rstrip("/")
        self.default_voice_id = default_voice_id
        self.timeout = timeout

        # Rate limiting and quota management
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.quota_used = 0
        self.quota_limit = None
        self.quota_reset_date = None

        # Voice cache
        self._voice_cache: Dict[str, VoiceModel] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=24)

        # Session for connection pooling
        self._session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_characters_synthesized": 0,
            "total_cost": 0.0,
            "rate_limit_hits": 0,
            "quota_limit_hits": 0
        }

    async def initialize(self) -> None:
        """Initialize the service and load available voices"""
        if self._session is None:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            timeout = aiohttp.ClientTimeout(total=self.timeout)

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "User-Agent": "REAutomation2/1.0"
                }
            )

        # Load available voices
        await self._load_voices()

        logger.info("ElevenLabs service initialized successfully")

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self._session:
            await self._session.close()
            self._session = None

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=30
    )
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        voice_settings: Optional[Dict[str, float]] = None
    ) -> TTSResponse:
        """
        Synthesize speech from text using ElevenLabs API

        Args:
            text: Text to synthesize
            voice_id: Voice ID to use (defaults to default_voice_id)
            voice_settings: Voice settings (stability, similarity_boost, etc.)

        Returns:
            TTSResponse with audio data and metadata

        Raises:
            RateLimitError: When rate limit is exceeded
            QuotaExceededError: When quota is exceeded
            ElevenLabsError: For other API errors
        """
        if not self._session:
            await self.initialize()

        voice_id = voice_id or self.default_voice_id
        start_time = time.time()

        # Apply rate limiting
        await self.rate_limiter.wait_if_needed()

        # Default voice settings for clear speech
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }

        request_data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": voice_settings
        }

        url = f"{self.base_url}/text-to-speech/{voice_id}"

        try:
            self.stats["total_requests"] += 1

            async with self._session.post(url, json=request_data) as response:
                await self._handle_response_headers(response)

                if response.status == 429:
                    self.stats["rate_limit_hits"] += 1
                    raise RateLimitError("ElevenLabs rate limit exceeded")

                if response.status == 401:
                    raise ElevenLabsError("Invalid API key")

                if response.status == 402:
                    self.stats["quota_limit_hits"] += 1
                    raise QuotaExceededError("ElevenLabs quota exceeded")

                if response.status != 200:
                    error_text = await response.text()
                    raise ElevenLabsError(f"API error {response.status}: {error_text}")

                audio_data = await response.read()
                processing_time = time.time() - start_time

                # Calculate cost estimate (approximate)
                character_count = len(text)
                cost_estimate = character_count * 0.00003  # ~$0.30 per 1K characters

                self.stats["successful_requests"] += 1
                self.stats["total_characters_synthesized"] += character_count
                self.stats["total_cost"] += cost_estimate

                logger.debug(f"Synthesized {character_count} characters in {processing_time:.2f}s")

                return TTSResponse(
                    audio_data=audio_data,
                    character_count=character_count,
                    cost_estimate=cost_estimate,
                    processing_time=processing_time,
                    voice_id=voice_id,
                    model_id="eleven_monolingual_v1"
                )

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self.stats["failed_requests"] += 1
            logger.error(f"ElevenLabs API request failed: {e}")
            raise ElevenLabsError(f"Request failed: {e}")

    async def get_available_voices(self, force_refresh: bool = False) -> List[VoiceModel]:
        """
        Get list of available voices

        Args:
            force_refresh: Force refresh of voice cache

        Returns:
            List of available VoiceModel objects
        """
        if (not force_refresh and
            self._cache_timestamp and
            datetime.now() - self._cache_timestamp < self._cache_ttl and
            self._voice_cache):
            return list(self._voice_cache.values())

        await self._load_voices()
        return list(self._voice_cache.values())

    async def get_voice_by_id(self, voice_id: str) -> Optional[VoiceModel]:
        """Get voice by ID"""
        voices = await self.get_available_voices()
        return self._voice_cache.get(voice_id)

    async def get_voices_by_category(self, category: str) -> List[VoiceModel]:
        """Get voices filtered by category"""
        voices = await self.get_available_voices()
        return [voice for voice in voices if voice.category.lower() == category.lower()]

    async def get_user_info(self) -> Dict[str, Any]:
        """Get user subscription information and quota details"""
        if not self._session:
            await self.initialize()

        url = f"{self.base_url}/user"

        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    raise ElevenLabsError(f"Failed to get user info: {response.status}")

                return await response.json()

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise ElevenLabsError(f"Failed to get user info: {e}")

    async def check_quota_status(self) -> Dict[str, Any]:
        """Check current quota usage and limits"""
        try:
            user_info = await self.get_user_info()

            subscription = user_info.get("subscription", {})
            quota = subscription.get("character_count", 0)
            quota_limit = subscription.get("character_limit", 0)

            return {
                "characters_used": quota,
                "character_limit": quota_limit,
                "characters_remaining": quota_limit - quota,
                "usage_percentage": (quota / quota_limit * 100) if quota_limit > 0 else 0,
                "tier": subscription.get("tier", "unknown"),
                "reset_date": subscription.get("next_character_count_reset_unix", None)
            }

        except Exception as e:
            logger.error(f"Failed to check quota status: {e}")
            return {
                "characters_used": 0,
                "character_limit": 0,
                "characters_remaining": 0,
                "usage_percentage": 0,
                "tier": "unknown",
                "reset_date": None
            }

    async def _load_voices(self) -> None:
        """Load available voices from API"""
        if not self._session:
            await self.initialize()

        url = f"{self.base_url}/voices"

        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to load voices: {response.status}")
                    return

                data = await response.json()
                voices_data = data.get("voices", [])

                self._voice_cache.clear()
                for voice_data in voices_data:
                    voice = VoiceModel(
                        voice_id=voice_data["voice_id"],
                        name=voice_data["name"],
                        preview_url=voice_data.get("preview_url"),
                        category=voice_data.get("category", "generated"),
                        description=voice_data.get("description"),
                        labels=voice_data.get("labels", {}),
                        samples=voice_data.get("samples", []),
                        settings=voice_data.get("settings")
                    )
                    self._voice_cache[voice.voice_id] = voice

                self._cache_timestamp = datetime.now()
                logger.info(f"Loaded {len(self._voice_cache)} voices from ElevenLabs")

        except Exception as e:
            logger.error(f"Failed to load voices: {e}")

    async def _handle_response_headers(self, response: aiohttp.ClientResponse) -> None:
        """Handle response headers for quota tracking"""
        # Update quota information from headers
        if "xi-character-count" in response.headers:
            try:
                self.quota_used = int(response.headers["xi-character-count"])
            except ValueError:
                pass

        if "xi-character-limit" in response.headers:
            try:
                self.quota_limit = int(response.headers["xi-character-limit"])
            except ValueError:
                pass

    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.stats,
            "quota_used": self.quota_used,
            "quota_limit": self.quota_limit,
            "voices_cached": len(self._voice_cache),
            "cache_age": (datetime.now() - self._cache_timestamp).total_seconds()
                        if self._cache_timestamp else None
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the service"""
        try:
            # Try to get user info as a simple API test
            user_info = await self.get_user_info()

            # Check quota status
            quota_status = await self.check_quota_status()

            return {
                "status": "healthy",
                "api_accessible": True,
                "quota_status": quota_status,
                "service_stats": self.get_service_stats()
            }

        except Exception as e:
            logger.error(f"ElevenLabs health check failed: {e}")
            return {
                "status": "unhealthy",
                "api_accessible": False,
                "error": str(e),
                "service_stats": self.get_service_stats()
            }

    # Voice recommendation methods
    async def get_recommended_voice_for_use_case(self, use_case: str) -> Optional[VoiceModel]:
        """Get recommended voice for specific use case"""
        voices = await self.get_available_voices()

        # Define use case mappings
        use_case_mappings = {
            "real_estate": ["Rachel", "Bella", "Antoni"],
            "professional": ["Rachel", "Josh", "Arnold"],
            "friendly": ["Bella", "Rachel", "Sam"],
            "authoritative": ["Josh", "Antoni", "Arnold"],
            "conversational": ["Rachel", "Sam", "Bella"]
        }

        preferred_names = use_case_mappings.get(use_case.lower(), ["Rachel"])

        for name in preferred_names:
            for voice in voices:
                if voice.name.lower() == name.lower():
                    return voice

        # Fallback to default voice
        return await self.get_voice_by_id(self.default_voice_id)

    def estimate_cost(self, text: str) -> float:
        """Estimate cost for synthesizing given text"""
        character_count = len(text)
        return character_count * 0.00003  # Approximate cost per character

    async def batch_synthesize(
        self,
        requests: List[TTSRequest],
        max_concurrent: int = 5
    ) -> List[TTSResponse]:
        """Synthesize multiple texts concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def synthesize_with_semaphore(request: TTSRequest) -> TTSResponse:
            async with semaphore:
                return await self.synthesize_speech(
                    text=request.text,
                    voice_id=request.voice_id,
                    voice_settings=request.voice_settings
                )

        tasks = [synthesize_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)