"""Circuit Breaker implementation for voice-agent integration"""

import logging
import time
from enum import Enum
from typing import Callable, Any, Optional
import asyncio
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure threshold exceeded, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker pattern implementation for resilient service calls"""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before transitioning to half-open
            half_open_max_calls: Number of calls to allow in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_call_count = 0

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func execution

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from func if it fails
        """
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_call_count = 0
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_call_count >= self.half_open_max_calls:
                raise CircuitBreakerOpenError("Circuit breaker HALF_OPEN call limit reached")

        try:
            # Execute the function
            result = await func(*args, **kwargs)

            # Record success
            await self._on_success()

            return result

        except Exception as e:
            # Record failure
            await self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.timeout_seconds

    async def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            self.half_open_call_count += 1

            if self.success_count >= self.half_open_max_calls:
                logger.info("Circuit breaker transitioning to CLOSED after successful recovery")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.half_open_call_count = 0

        elif self.state == CircuitState.CLOSED:
            # Reset failure count on successful calls
            self.failure_count = 0

    async def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker transitioning to OPEN after failure in HALF_OPEN state")
            self.state = CircuitState.OPEN
            self.success_count = 0
            self.half_open_call_count = 0

        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker transitioning to OPEN after {self.failure_count} failures"
                )
                self.state = CircuitState.OPEN

    def get_state(self) -> CircuitState:
        """Get current circuit state"""
        return self.state

    def reset(self):
        """Manually reset circuit breaker to closed state"""
        logger.info("Circuit breaker manually reset to CLOSED")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_call_count = 0
        self.last_failure_time = None

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "half_open_call_count": self.half_open_call_count,
            "last_failure_time": self.last_failure_time,
            "time_since_last_failure": (
                time.time() - self.last_failure_time
                if self.last_failure_time else None
            )
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


def circuit_breaker(
    failure_threshold: int = 5,
    timeout_seconds: int = 60,
    fallback: Optional[Callable] = None
):
    """
    Decorator to apply circuit breaker pattern to async functions

    Args:
        failure_threshold: Number of failures before opening circuit
        timeout_seconds: Time to wait before transitioning to half-open
        fallback: Optional fallback function to call when circuit is open

    Example:
        @circuit_breaker(failure_threshold=3, timeout_seconds=30)
        async def call_external_service():
            # Service call implementation
            pass
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        timeout_seconds=timeout_seconds
    )

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await breaker.call(func, *args, **kwargs)
            except CircuitBreakerOpenError as e:
                logger.warning(f"Circuit breaker open for {func.__name__}: {e}")

                if fallback:
                    logger.info(f"Using fallback for {func.__name__}")
                    return await fallback(*args, **kwargs)

                raise e

        # Attach breaker to wrapper for access to metrics
        wrapper.circuit_breaker = breaker
        return wrapper

    return decorator


# Global circuit breakers for different services
agent_orchestrator_breaker = CircuitBreaker(
    failure_threshold=5,
    timeout_seconds=60,
    half_open_max_calls=3
)

llm_service_breaker = CircuitBreaker(
    failure_threshold=10,
    timeout_seconds=30,
    half_open_max_calls=5
)
