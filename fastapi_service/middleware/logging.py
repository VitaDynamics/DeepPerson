"""Request/response logging middleware with structured logging support.

Implements Observability principle (Constitution Principle V).
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all HTTP requests and responses with request IDs."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with logging.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response
        """
        # Generate unique request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log incoming request
        start_time = time.time()
        logger.info(
            f"request_id={request_id} method={request.method} "
            f"path={request.url.path} client={request.client.host if request.client else 'unknown'}"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                f"request_id={request_id} method={request.method} "
                f"path={request.url.path} status={response.status_code} "
                f"duration={duration:.3f}s"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                f"request_id={request_id} method={request.method} "
                f"path={request.url.path} error={str(e)} "
                f"duration={duration:.3f}s"
            )
            raise
