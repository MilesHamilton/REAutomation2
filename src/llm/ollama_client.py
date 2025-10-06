import asyncio
import aiohttp
import time
import json
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager

from ..config import settings
from .models import LLMRequest, LLMResponse, Message, HealthStatus, PerformanceMetrics

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self):
        self.base_url = settings.ollama_host
        self.model = settings.ollama_model

        # Adaptive concurrency settings
        self.adaptive_concurrency = settings.ollama_adaptive_concurrency
        self.min_concurrent = settings.ollama_min_concurrent if self.adaptive_concurrency else settings.llm_max_concurrent
        self.max_concurrent = settings.ollama_max_concurrent if self.adaptive_concurrency else settings.llm_max_concurrent
        self.current_concurrent = settings.llm_max_concurrent

        self.semaphore = asyncio.Semaphore(self.current_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None

        self.metrics = {
            "requests_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_response_time": 0.0,
            "concurrent_requests": 0,
            "last_request_time": 0.0,
            "response_times": [],  # Track last 50 response times for adaptive concurrency
        }

        # Adaptive concurrency adjustment
        self._last_adjustment_time = time.time()
        self._adjustment_interval = 30.0  # Adjust every 30 seconds

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        if self._session is None:
            # Create TCP connector with connection pooling
            self._connector = aiohttp.TCPConnector(
                limit=settings.ollama_connection_pool_size,
                limit_per_host=settings.ollama_connection_pool_limit_per_host,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                force_close=False,  # Enable connection reuse
                keepalive_timeout=settings.ollama_keepalive_timeout
            )

            # Configure timeouts
            timeout = aiohttp.ClientTimeout(
                total=settings.ollama_request_timeout,
                connect=settings.ollama_connection_timeout,
                sock_read=settings.ollama_request_timeout
            )

            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout
            )
            logger.info(
                f"Connected to Ollama at {self.base_url} with connection pool "
                f"(size={settings.ollama_connection_pool_size}, per_host={settings.ollama_connection_pool_limit_per_host})"
            )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
            logger.info("Ollama client connection closed")

    @asynccontextmanager
    async def _get_session(self):
        if self._session is None:
            await self.connect()
        yield self._session

    async def generate(self, request: LLMRequest) -> LLMResponse:
        async with self.semaphore:
            self.metrics["concurrent_requests"] += 1
            start_time = time.time()

            try:
                response = await self._make_request(request)
                self.metrics["success_count"] += 1
                return response
            except Exception as e:
                self.metrics["error_count"] += 1
                logger.error(f"LLM generation error: {e}")
                raise
            finally:
                response_time = (time.time() - start_time) * 1000
                self.metrics["total_response_time"] += response_time
                self.metrics["requests_count"] += 1
                self.metrics["concurrent_requests"] -= 1
                self.metrics["last_request_time"] = time.time()

                # Track response times for adaptive concurrency
                self.metrics["response_times"].append(response_time)
                if len(self.metrics["response_times"]) > 50:
                    self.metrics["response_times"].pop(0)

                # Adjust concurrency if enabled
                if self.adaptive_concurrency:
                    await self._maybe_adjust_concurrency()

    async def generate_stream(
        self,
        request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response from LLM

        Yields text chunks as they arrive from the model.
        Useful for real-time applications like voice assistants.
        """
        if not settings.ollama_streaming_enabled:
            # Streaming disabled, fall back to regular generation
            response = await self.generate(request)
            yield response.content
            return

        async with self.semaphore:
            self.metrics["concurrent_requests"] += 1
            start_time = time.time()
            total_content = ""

            try:
                # Prepare messages for Ollama format
                messages = []
                if request.system_prompt:
                    messages.append({"role": "system", "content": request.system_prompt})

                for msg in request.messages:
                    messages.append({"role": msg.role.value, "content": msg.content})

                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,  # Enable streaming
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": request.max_tokens,
                    }
                }

                async with self._get_session() as session:
                    url = f"{self.base_url}/api/chat"

                    async with session.post(url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Ollama API error {response.status}: {error_text}")

                        # Stream response chunks
                        async for line in response.content:
                            if not line:
                                continue

                            try:
                                # Parse JSON chunk
                                chunk_data = json.loads(line.decode('utf-8'))

                                # Extract content from chunk
                                if "message" in chunk_data:
                                    content = chunk_data["message"].get("content", "")
                                    if content:
                                        total_content += content
                                        yield content

                                # Check if done
                                if chunk_data.get("done", False):
                                    break

                            except json.JSONDecodeError:
                                # Skip malformed chunks
                                continue

                self.metrics["success_count"] += 1
                logger.debug(f"Streaming completed, total length: {len(total_content)}")

            except Exception as e:
                self.metrics["error_count"] += 1
                logger.error(f"LLM streaming error: {e}")
                raise

            finally:
                response_time = (time.time() - start_time) * 1000
                self.metrics["total_response_time"] += response_time
                self.metrics["requests_count"] += 1
                self.metrics["concurrent_requests"] -= 1
                self.metrics["last_request_time"] = time.time()

                # Track response times for adaptive concurrency
                self.metrics["response_times"].append(response_time)
                if len(self.metrics["response_times"]) > 50:
                    self.metrics["response_times"].pop(0)

                # Adjust concurrency if enabled
                if self.adaptive_concurrency:
                    await self._maybe_adjust_concurrency()

    async def _make_request(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()

        # Prepare messages for Ollama format
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            messages.append({"role": msg.role.value, "content": msg.content})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            }
        }

        # Add structured output format if requested
        if request.structured_output and request.response_format:
            payload["format"] = "json"
            payload["options"]["temperature"] = 0.1  # Lower temp for structured output

        async with self._get_session() as session:
            url = f"{self.base_url}/api/chat"

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")

                data = await response.json()

                response_time_ms = (time.time() - start_time) * 1000
                content = data.get("message", {}).get("content", "")

                # Parse structured output if requested
                structured_data = None
                if request.structured_output:
                    try:
                        structured_data = json.loads(content)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse structured output as JSON")

                return LLMResponse(
                    content=content,
                    usage_tokens=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
                    response_time_ms=response_time_ms,
                    model_used=self.model,
                    structured_data=structured_data,
                    confidence_score=self._calculate_confidence(data)
                )

    def _calculate_confidence(self, response_data: Dict[str, Any]) -> Optional[float]:
        # Simple confidence calculation based on response metadata
        # This can be enhanced with more sophisticated metrics
        eval_count = response_data.get("eval_count", 0)
        if eval_count > 0:
            # Normalize based on typical response lengths
            return min(1.0, eval_count / 100.0)
        return None

    async def health_check(self) -> HealthStatus:
        start_time = time.time()

        try:
            async with self._get_session() as session:
                # Check if Ollama is running and model is loaded
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status != 200:
                        return HealthStatus(
                            service="ollama",
                            status="unhealthy",
                            response_time_ms=(time.time() - start_time) * 1000,
                            concurrent_requests=self.metrics["concurrent_requests"],
                            last_check=time.time(),
                            details={"error": f"API returned status {response.status}"}
                        )

                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    model_loaded = self.model in models

                    status = "healthy" if model_loaded else "degraded"
                    details = {
                        "model_loaded": model_loaded,
                        "available_models": models,
                        "target_model": self.model
                    }

                    return HealthStatus(
                        service="ollama",
                        status=status,
                        response_time_ms=(time.time() - start_time) * 1000,
                        concurrent_requests=self.metrics["concurrent_requests"],
                        last_check=time.time(),
                        details=details
                    )

        except Exception as e:
            return HealthStatus(
                service="ollama",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                concurrent_requests=self.metrics["concurrent_requests"],
                last_check=time.time(),
                details={"error": str(e)}
            )

    async def get_metrics(self) -> PerformanceMetrics:
        avg_response_time = (
            self.metrics["total_response_time"] / max(1, self.metrics["requests_count"])
        )

        requests_per_minute = 0.0
        if self.metrics["last_request_time"] > 0:
            time_diff = time.time() - self.metrics["last_request_time"]
            if time_diff < 60:  # Only calculate if within last minute
                requests_per_minute = self.metrics["requests_count"] / max(1, time_diff / 60)

        success_rate = (
            self.metrics["success_count"] / max(1, self.metrics["requests_count"])
        )

        return PerformanceMetrics(
            avg_response_time_ms=avg_response_time,
            requests_per_minute=requests_per_minute,
            success_rate=success_rate,
            concurrent_peak=self.max_concurrent,
            memory_peak_mb=0.0,  # Would need system monitoring for actual values
            error_count=self.metrics["error_count"],
            timestamp=time.time()
        )

    async def _maybe_adjust_concurrency(self):
        """Adjust concurrency limits based on performance"""
        current_time = time.time()

        # Only adjust every adjustment_interval seconds
        if current_time - self._last_adjustment_time < self._adjustment_interval:
            return

        # Need at least 10 samples to make adjustment decision
        if len(self.metrics["response_times"]) < 10:
            return

        # Calculate average response time from recent samples
        avg_response_time = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])

        old_concurrent = self.current_concurrent

        # Adjust based on response time
        if avg_response_time > 2000:  # >2s, decrease concurrency
            self.current_concurrent = max(self.min_concurrent, self.current_concurrent - 1)
        elif avg_response_time < 500:  # <500ms, increase concurrency
            self.current_concurrent = min(self.max_concurrent, self.current_concurrent + 1)

        # Update semaphore if concurrency changed
        if old_concurrent != self.current_concurrent:
            self.semaphore = asyncio.Semaphore(self.current_concurrent)
            logger.info(
                f"Adjusted concurrency: {old_concurrent} -> {self.current_concurrent} "
                f"(avg response: {avg_response_time:.0f}ms)"
            )

        self._last_adjustment_time = current_time

    async def preload_model(self) -> bool:
        try:
            # Make a small request to ensure model is loaded
            test_request = LLMRequest(
                messages=[Message(role="user", content="Hello")],
                max_tokens=10
            )
            await self.generate(test_request)
            logger.info(f"Model {self.model} preloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to preload model {self.model}: {e}")
            return False