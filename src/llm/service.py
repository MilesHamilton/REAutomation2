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
from .batch_processor import RequestBatchProcessor
from .gpu_manager import gpu_manager
from .context_manager import ContextManager, ContextManagerFactory
from .prompt_optimizer import PromptOptimizer, PromptType
from .streaming_handler import StreamingHandler, VoiceStreamAdapter, StreamingMetricsCollector
from ..config import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self._startup_complete = False
        self._health_status: Optional[HealthStatus] = None

        # Initialize batch processor
        self.batch_processor: Optional[RequestBatchProcessor] = None

        # Initialize context manager
        self.context_manager: Optional[ContextManager] = None

        # Initialize prompt optimizer
        self.prompt_optimizer: Optional[PromptOptimizer] = None

        # Initialize streaming components
        self.streaming_metrics_collector = StreamingMetricsCollector()

    async def startup(self) -> bool:
        try:
            logger.info("Starting LLM service with caching, queue management, and batch processing...")

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

            # Initialize batch processor
            self.batch_processor = RequestBatchProcessor(
                processor=self._process_request_direct,
                enabled=settings.ollama_batch_enabled,
                batch_window_ms=settings.ollama_batch_window_ms,
                max_batch_size=settings.ollama_batch_max_size,
                similarity_threshold=settings.ollama_batch_similarity_threshold
            )
            await self.batch_processor.start()

            # Start request queue with batch processor as intermediary
            await request_queue.start(self._process_with_batching)

            # Start GPU manager
            await gpu_manager.start()

            # Initialize context manager
            self.context_manager = ContextManagerFactory.create(
                llm_client=self.ollama_client
            )
            logger.info(f"Context manager initialized with strategy: {self.context_manager.strategy_name}")

            # Initialize prompt optimizer
            self.prompt_optimizer = PromptOptimizer()
            template_stats = self.prompt_optimizer.get_template_stats()
            logger.info(
                f"Prompt optimizer initialized: {template_stats['total_templates']} templates, "
                f"{template_stats['average_tokens']:.0f} avg tokens"
            )

            self._startup_complete = True
            logger.info("LLM service started successfully with all optimizations enabled")
            return True

        except Exception as e:
            logger.error(f"LLM service startup failed: {e}")
            return False

    async def shutdown(self):
        logger.info("Shutting down LLM service...")

        # Stop batch processor
        if self.batch_processor:
            await self.batch_processor.stop()

        # Stop request queue
        await request_queue.stop()

        # Stop GPU manager
        await gpu_manager.stop()

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

        # Manage context window (prune if necessary)
        if self.context_manager and settings.LLM_CONTEXT_MANAGEMENT_ENABLED:
            # Convert messages to dicts for context manager
            message_dicts = [msg.dict() for msg in messages]

            # Manage context
            managed_dicts, context_stats = await self.context_manager.manage_context(
                message_dicts, system_prompt
            )

            # Convert back to Message objects
            messages = [Message(**msg_dict) for msg_dict in managed_dicts]

            # Update context tracking
            if context_stats.pruned:
                context.context_pruned = True
                context.pruning_count += 1
                context.last_pruning_strategy = context_stats.strategy_used
                logger.info(
                    f"Context pruned for call {context.call_id}: "
                    f"{context_stats.total_messages} messages, "
                    f"{context_stats.total_tokens} tokens "
                    f"({context_stats.utilization:.1f}% utilization)"
                )

            context.total_tokens = context_stats.total_tokens

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

    async def generate_response_streaming(
        self,
        context: ConversationContext,
        user_input: str,
        system_prompt: Optional[str] = None,
        for_voice: bool = False
    ):
        """
        Generate streaming LLM response with optimized performance.

        Args:
            context: Conversation context
            user_input: User input message
            system_prompt: Optional system prompt
            for_voice: Whether to optimize chunks for voice synthesis

        Yields:
            String chunks of the response
        """
        if not self._startup_complete:
            raise RuntimeError("LLM service not initialized")

        # Create user message
        user_message = Message(role=MessageRole.USER, content=user_input)
        messages = context.messages + [user_message]

        # Use default system prompt if none provided
        if not system_prompt:
            system_prompt = self._get_default_system_prompt(context)

        # Optimize system prompt
        if self.prompt_optimizer:
            system_prompt, opt_metrics = self.prompt_optimizer.optimize_prompt(system_prompt)
            logger.debug(
                f"Prompt optimized: {opt_metrics.original_tokens} -> {opt_metrics.optimized_tokens} tokens "
                f"({opt_metrics.reduction_percent:.1f}% reduction)"
            )

        # Manage context window
        if self.context_manager and settings.LLM_CONTEXT_MANAGEMENT_ENABLED:
            message_dicts = [msg.dict() for msg in messages]
            managed_dicts, context_stats = await self.context_manager.manage_context(
                message_dicts, system_prompt
            )
            messages = [Message(**msg_dict) for msg_dict in managed_dicts]

            if context_stats.pruned:
                context.context_pruned = True
                context.pruning_count += 1
                context.last_pruning_strategy = context_stats.strategy_used

            context.total_tokens = context_stats.total_tokens

        # Create streaming handler
        streaming_handler = StreamingHandler()

        # Optionally adapt for voice
        if for_voice:
            voice_adapter = VoiceStreamAdapter()

        try:
            # Get streaming generator from Ollama client
            stream_gen = self.ollama_client.generate_stream(
                messages=messages,
                system_prompt=system_prompt,
                max_tokens=150,
                temperature=0.7
            )

            # Apply handlers
            if for_voice:
                buffered_stream = streaming_handler.stream_with_buffer(stream_gen, aggregate_sentences=True)
                final_stream = voice_adapter.adapt_for_voice(buffered_stream)
            else:
                final_stream = streaming_handler.stream_with_buffer(stream_gen)

            # Yield chunks
            async for chunk in final_stream:
                yield chunk

            # Record metrics
            metrics = streaming_handler.get_metrics()
            self.streaming_metrics_collector.record_stream(metrics)

            logger.info(
                f"Streaming completed: {metrics['total_chunks']} chunks, "
                f"{metrics['time_to_first_chunk_ms']:.0f}ms TTFC, "
                f"{metrics['throughput_tokens_per_second']:.1f} tokens/s"
            )

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    async def _process_request_direct(self, request: LLMRequest) -> LLMResponse:
        """
        Process LLM request directly through Ollama client
        This method is used by the batch processor
        """
        # Track model usage for GPU manager
        await gpu_manager.track_model_usage(settings.ollama_model)

        return await self.ollama_client.generate(request)

    async def _process_with_batching(self, request: LLMRequest) -> LLMResponse:
        """
        Process LLM request with batching optimization
        This method is used by the queue manager
        """
        if self.batch_processor and self.batch_processor.enabled:
            return await self.batch_processor.process_request(request)
        else:
            return await self._process_request_direct(request)

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

        # Manage context window (prune if necessary)
        if self.context_manager and settings.LLM_CONTEXT_MANAGEMENT_ENABLED:
            message_dicts = [msg.dict() for msg in messages]
            managed_dicts, context_stats = await self.context_manager.manage_context(
                message_dicts, system_prompt
            )
            messages = [Message(**msg_dict) for msg_dict in managed_dicts]

            if context_stats.pruned:
                context.context_pruned = True
                context.pruning_count += 1
                context.last_pruning_strategy = context_stats.strategy_used

            context.total_tokens = context_stats.total_tokens

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
        # Use optimized prompt template
        if self.prompt_optimizer:
            qualification_prompt = self.prompt_optimizer.get_template("qualification")
        else:
            # Fallback to basic optimized version
            qualification_prompt = """Analyze conversation for lead qualification. Score 0-1 on:
- Intent, Budget, Timeline, Authority, Needs

JSON format:
{
  "qualification_score": 0.0-1.0,
  "confidence": 0.0-1.0,
  "factors": {"intent": 0.0-1.0, "budget": 0.0-1.0, "timeline": 0.0-1.0, "authority": 0.0-1.0, "needs": 0.0-1.0},
  "reasoning": "Brief explanation",
  "recommended_action": "continue|escalate|disqualify"
}"""

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
        # Use optimized prompt template if available
        if self.prompt_optimizer:
            company = context.metadata.get("company", "our company")
            lead_context = str(context.lead_info) if context.lead_info else "No prior info"
            conv_state = context.conversation_state

            base_prompt = self.prompt_optimizer.get_template(
                "conversation",
                company=company,
                lead_context=lead_context,
                conversation_state=conv_state
            )
        else:
            # Fallback to optimized version
            base_prompt = f"""You're an AI sales agent. Be friendly, concise (1-2 sentences), ask one question at a time.

State: {context.conversation_state}"""

            if context.lead_info:
                base_prompt += f"\nLead: {context.lead_info}"

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
        batch_health = await self.batch_processor.health_check() if self.batch_processor else {"status": "disabled"}
        gpu_health = await gpu_manager.health_check()

        # Determine overall health
        overall_status = ollama_health.status
        if queue_health["status"] != "healthy":
            overall_status = "degraded"
        if batch_health["status"] == "degraded":
            overall_status = "degraded"
        if gpu_health["status"] == "degraded":
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
                },
                "batch_processor": batch_health,
                "gpu_manager": gpu_health
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
        """Get comprehensive metrics including cache, queue, batching, and GPU performance"""
        ollama_metrics = await self.ollama_client.get_metrics()
        cache_stats = llm_cache.get_stats()
        queue_status = await request_queue.get_queue_status()
        batch_metrics = await self.batch_processor.get_metrics() if self.batch_processor else {}
        gpu_metrics = await gpu_manager.get_metrics()

        return {
            "llm": ollama_metrics.dict(),
            "cache": cache_stats,
            "queue": queue_status["metrics"],
            "batch_processor": batch_metrics,
            "gpu": gpu_metrics,
            "overall": {
                "total_requests_processed": queue_status["metrics"]["completed_requests"],
                "cache_hit_rate": cache_stats["hit_rate"],
                "queue_success_rate": queue_status["metrics"]["success_rate"],
                "average_queue_wait_time": queue_status["metrics"].get("average_wait_time", 0.0),
                "concurrent_processing_capacity": queue_status["max_concurrent"],
                "batch_saved_llm_calls": batch_metrics.get("saved_llm_calls", 0),
                "batch_saved_percentage": batch_metrics.get("saved_percentage", 0.0),
                "gpu_memory_used_mb": gpu_metrics.get("used_memory_mb", 0.0),
                "gpu_utilization_percent": gpu_metrics.get("utilization_percent", 0.0)
            }
        }

    def is_ready(self) -> bool:
        return self._startup_complete

    async def get_context_stats(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Get context window statistics for a conversation.

        Args:
            context: ConversationContext to analyze

        Returns:
            Dictionary with context statistics
        """
        if not self.context_manager:
            return {"error": "Context manager not initialized"}

        message_dicts = [msg.dict() for msg in context.messages]
        system_prompt = self._get_default_system_prompt(context)

        stats = self.context_manager.get_context_stats(message_dicts, system_prompt)
        capacity = self.context_manager.estimate_remaining_capacity(message_dicts, system_prompt)

        return {
            "current_stats": stats.to_dict(),
            "capacity": capacity,
            "context_info": {
                "total_tokens": context.total_tokens,
                "context_pruned": context.context_pruned,
                "pruning_count": context.pruning_count,
                "last_pruning_strategy": context.last_pruning_strategy
            },
            "manager_summary": self.context_manager.get_stats_summary()
        }

    def get_streaming_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated streaming performance metrics.

        Returns:
            Dictionary with streaming metrics
        """
        return self.streaming_metrics_collector.get_aggregate_metrics()

    def get_prompt_optimization_stats(self) -> Dict[str, Any]:
        """
        Get prompt optimization statistics.

        Returns:
            Dictionary with prompt stats
        """
        if not self.prompt_optimizer:
            return {"error": "Prompt optimizer not initialized"}

        return self.prompt_optimizer.get_template_stats()


# Global service instance
llm_service = LLMService()