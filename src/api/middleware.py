import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url}")

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {response.status_code} - "
            f"{process_time:.3f}s - "
            f"{request.method} {request.url}"
        )

        # Add processing time to headers
        response.headers["X-Process-Time"] = str(process_time)

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.total_time = 0.0
        self.error_count = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        try:
            response = await call_next(request)

            # Update metrics
            process_time = time.time() - start_time
            self.request_count += 1
            self.total_time += process_time

            if response.status_code >= 400:
                self.error_count += 1

            return response

        except Exception as e:
            # Handle errors
            self.error_count += 1
            logger.error(f"Request failed: {e}")
            raise

    def get_metrics(self):
        avg_time = self.total_time / max(1, self.request_count)
        return {
            "request_count": self.request_count,
            "average_response_time": avg_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count)
        }