import asyncio
import aiohttp
import time
import json
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from ..config import settings
from .models import LLMRequest, LLMResponse, Message, HealthStatus, PerformanceMetrics

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self):
        self.base_url = settings.ollama_host
        self.model = settings.ollama_model
        self.max_concurrent = settings.llm_max_concurrent
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
        self.metrics = {
            "requests_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_response_time": 0.0,
            "concurrent_requests": 0,
            "last_request_time": 0.0,
        }

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"Connected to Ollama at {self.base_url}")

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