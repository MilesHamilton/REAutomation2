import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from .ollama_client import OllamaClient
from .models import (
    LLMRequest, LLMResponse, Message, MessageRole,
    ConversationContext, HealthStatus, PerformanceMetrics
)
from .cache import llm_cache, CacheStrategy, setup_cache, cleanup_cache
from .queue_manager import request_queue, RequestPriority

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self._startup_complete = False
        self._health_status: Optional[HealthStatus] = None

    async def startup(self) -> bool:
        try:
            logger.info("Starting LLM service with caching and queue management...")

            # Initialize Ollama client
            await self.ollama_client.connect()
            model_loaded = await self.ollama_client.preload_model()
            if not model_loaded:
                logger.error("Failed to load LLM model")
                return False

            # Initialize cache
            cache_connected = await setup_cache()
            if cache_connected:
                logger.info("Cache system initialized successfully")
            else:
                logger.warning("Cache system failed to initialize, continuing without cache")

            # Start request queue with Ollama client as processor
            await request_queue.start(self._process_request_direct)

            self._startup_complete = True
            logger.info("LLM service started successfully with enhanced features")
            return True

        except Exception as e:
            logger.error(f"LLM service startup failed: {e}")
            return False

    async def shutdown(self):
        logger.info("Shutting down LLM service...")

        # Stop request queue
        await request_queue.stop()

        # Cleanup cache
        await cleanup_cache()

        # Close Ollama client
        await self.ollama_client.close()

        self._startup_complete = False
        logger.info("LLM service shutdown complete")

    @asynccontextmanager
    async def lifespan(self):
        await self.startup()
        try:
            yield self
        finally:
            await self.shutdown()

    async def generate_response(
        self,
        context: ConversationContext,
        user_input: str,
        system_prompt: Optional[str] = None,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> LLMResponse:
        if not self._startup_complete:
            raise RuntimeError("LLM service not initialized")

        # Create a new message from user input
        user_message = Message(role=MessageRole.USER, content=user_input)

        # Prepare request with conversation history
        messages = context.messages + [user_message]

        # Use default system prompt if none provided
        if not system_prompt:
            system_prompt = self._get_default_system_prompt(context)

        request = LLMRequest(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=150,
            temperature=0.7
        )

        # Try cache first
        cached_response = await llm_cache.get(request)
        if cached_response:
            logger.debug("Returning cached response")
            return cached_response

        # Queue the request for processing
        try:
            response_future = await request_queue.enqueue(
                request=request,
                priority=priority,
                agent_type="conversation"
            )

            response = await response_future

            # Cache the response if it should be cached
            if CacheStrategy.should_cache(request, response):
                ttl = CacheStrategy.get_ttl_for_request(request)
                await llm_cache.set(request, response, ttl)

            return response

        except Exception as e:
            logger.error(f"Failed to process LLM request: {e}")
            # Fallback to direct processing if queue fails
            logger.warning("Falling back to direct LLM processing")
            response = await self._process_request_direct(request)

            # Still try to cache on fallback
            if CacheStrategy.should_cache(request, response):
                ttl = CacheStrategy.get_ttl_for_request(request)
                await llm_cache.set(request, response, ttl)

            return response

    async def _process_request_direct(self, request: LLMRequest) -> LLMResponse:
        """
        Process LLM request directly through Ollama client
        This method is used by the queue manager
        """
        return await self.ollama_client.generate(request)

    async def generate_structured_response(
        self,
        context: ConversationContext,
        user_input: str,
        response_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        priority: RequestPriority = RequestPriority.HIGH
    ) -> LLMResponse:
        if not self._startup_complete:
            raise RuntimeError("LLM service not initialized")

        user_message = Message(role=MessageRole.USER, content=user_input)
        messages = context.messages + [user_message]

        if not system_prompt:
            system_prompt = self._get_structured_system_prompt(context, response_schema)

        request = LLMRequest(
            messages=messages,
            system_prompt=system_prompt,
            structured_output=True,
            response_format=response_schema,
            max_tokens=200,
            temperature=0.3  # Lower temperature for structured output
        )

        # Try cache first (structured responses cache well)
        cached_response = await llm_cache.get(request)
        if cached_response:
            logger.debug("Returning cached structured response")
            return cached_response

        # Queue with high priority for structured responses
        try:
            response_future = await request_queue.enqueue(
                request=request,
                priority=priority,
                agent_type="structured"
            )

            response = await response_future

            # Always cache structured responses (they're deterministic)
            ttl = CacheStrategy.get_ttl_for_request(request)
            await llm_cache.set(request, response, ttl)

            return response

        except Exception as e:
            logger.error(f"Failed to process structured LLM request: {e}")
            # Fallback to direct processing
            response = await self._process_request_direct(request)

            # Cache on fallback
            ttl = CacheStrategy.get_ttl_for_request(request)
            await llm_cache.set(request, response, ttl)

            return response

    async def analyze_qualification(
        self, context: ConversationContext
    ) -> Dict[str, Any]:
        qualification_prompt = """
        Analyze this conversation and provide a qualification score and reasoning.

        Consider factors like:
        - Intent to purchase/engage
        - Budget availability
        - Timeline urgency
        - Decision-making authority
        - Specific needs expressed

        Respond with JSON format:
        {
            "qualification_score": 0.0-1.0,
            "confidence": 0.0-1.0,
            "factors": {
                "intent": 0.0-1.0,
                "budget": 0.0-1.0,
                "timeline": 0.0-1.0,
                "authority": 0.0-1.0,
                "needs": 0.0-1.0
            },
            "reasoning": "Brief explanation",
            "recommended_action": "continue|escalate|disqualify"
        }
        """

        schema = {
            "type": "object",
            "properties": {
                "qualification_score": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "factors": {
                    "type": "object",
                    "properties": {
                        "intent": {"type": "number", "minimum": 0, "maximum": 1},
                        "budget": {"type": "number", "minimum": 0, "maximum": 1},
                        "timeline": {"type": "number", "minimum": 0, "maximum": 1},
                        "authority": {"type": "number", "minimum": 0, "maximum": 1},
                        "needs": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "reasoning": {"type": "string"},
                "recommended_action": {"type": "string", "enum": ["continue", "escalate", "disqualify"]}
            }
        }

        response = await self.generate_structured_response(
            context=context,
            user_input="Analyze this conversation for lead qualification.",
            response_schema=schema,
            system_prompt=qualification_prompt
        )

        return response.structured_data or {}

    def _get_default_system_prompt(self, context: ConversationContext) -> str:
        base_prompt = """
        You are an AI assistant conducting outbound sales calls for lead generation.
        Your role is to have natural, conversational interactions while gathering
        qualification information.

        Guidelines:
        - Be friendly, professional, and conversational
        - Keep responses concise (1-2 sentences max)
        - Ask one question at a time
        - Listen actively and respond to what the person says
        - Handle objections gracefully
        - Build rapport naturally
        """

        # Add context-specific information
        if context.lead_info:
            lead_context = f"\nLead information: {context.lead_info}"
            base_prompt += lead_context

        if context.conversation_state:
            state_context = f"\nConversation state: {context.conversation_state}"
            base_prompt += state_context

        return base_prompt

    def _get_structured_system_prompt(
        self, context: ConversationContext, schema: Dict[str, Any]
    ) -> str:
        return f"""
        You are analyzing a sales conversation for structured data extraction.
        Provide your response in valid JSON format only, following this schema:
        {schema}

        Base your analysis on the conversation history and be objective in your assessment.
        """

    async def health_check(self) -> HealthStatus:
        if not self._startup_complete:
            return HealthStatus(
                service="llm_service",
                status="unhealthy",
                response_time_ms=0,
                concurrent_requests=0,
                last_check=0,
                details={"error": "Service not initialized"}
            )

        # Get health from all components
        ollama_health = await self.ollama_client.health_check()
        cache_health = await llm_cache.health_check()
        queue_health = await request_queue.health_check()

        # Determine overall health
        overall_status = ollama_health.status
        if queue_health["status"] != "healthy":
            overall_status = "degraded"

        self._health_status = HealthStatus(
            service="llm_service",
            status=overall_status,
            response_time_ms=ollama_health.response_time_ms,
            concurrent_requests=ollama_health.concurrent_requests,
            last_check=ollama_health.last_check,
            details={
                "ollama": ollama_health.details,
                "cache": cache_health,
                "queue": {
                    "status": queue_health["status"],
                    "total_queued": queue_health["total_queued"],
                    "processing_count": queue_health["processing_count"],
                    "success_rate": queue_health["metrics"]["success_rate"]
                }
            }
        )

        return self._health_status

    async def get_metrics(self) -> PerformanceMetrics:
        ollama_metrics = await self.ollama_client.get_metrics()
        cache_stats = llm_cache.get_stats()
        queue_status = await request_queue.get_queue_status()

        # Enhanced metrics including cache and queue performance
        return PerformanceMetrics(
            avg_response_time_ms=ollama_metrics.avg_response_time_ms,
            requests_per_minute=ollama_metrics.requests_per_minute,
            success_rate=ollama_metrics.success_rate,
            concurrent_peak=ollama_metrics.concurrent_peak,
            memory_peak_mb=ollama_metrics.memory_peak_mb,
            gpu_memory_peak_mb=ollama_metrics.gpu_memory_peak_mb,
            error_count=ollama_metrics.error_count,
            timestamp=ollama_metrics.timestamp
        )

    async def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including cache and queue performance"""
        ollama_metrics = await self.ollama_client.get_metrics()
        cache_stats = llm_cache.get_stats()
        queue_status = await request_queue.get_queue_status()

        return {
            "llm": ollama_metrics.dict(),
            "cache": cache_stats,
            "queue": queue_status["metrics"],
            "overall": {
                "total_requests_processed": queue_status["metrics"]["completed_requests"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "queue_success_rate": queue_status["metrics"]["success_rate"],
                "average_queue_wait_time": queue_status["metrics"].get("average_wait_time", 0.0),
                "concurrent_processing_capacity": queue_status["max_concurrent"]
            }
        }

    def is_ready(self) -> bool:
        return self._startup_complete


# Global service instance
llm_service = LLMService()