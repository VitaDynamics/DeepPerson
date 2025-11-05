"""Global error handling middleware.

Implements Observability principle (Constitution Principle V).
"""

import logging
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    """Base exception for service errors.

    Attributes:
        code: Machine-readable error code
        message: Human-readable error message
        details: Additional error context
        status_code: HTTP status code
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        """Initialize service error.

        Args:
            code: Machine-readable error code
            message: Human-readable error message
            details: Additional error context
            status_code: HTTP status code
        """
        self.code = code
        self.message = message
        self.details = details or {}
        self.status_code = status_code
        super().__init__(self.message)


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    """Handle ServiceError exceptions.

    Args:
        request: FastAPI request
        exc: Service error exception

    Returns:
        JSON response with standardized error format
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"request_id={request_id} error_code={exc.code} "
        f"message={exc.message} details={exc.details}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
            "request_id": request_id,
        },
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions.

    Args:
        request: FastAPI request
        exc: HTTP exception

    Returns:
        JSON response with standardized error format
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"request_id={request_id} status={exc.status_code} detail={exc.detail}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
                "details": {},
            },
            "request_id": request_id,
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors.

    Args:
        request: FastAPI request
        exc: Validation error

    Returns:
        JSON response with standardized error format
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        f"request_id={request_id} validation_error={exc.errors()}"
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Request validation failed",
                "details": {"validation_errors": exc.errors()},
            },
            "request_id": request_id,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: FastAPI request
        exc: Generic exception

    Returns:
        JSON response with standardized error format
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"request_id={request_id} unexpected_error={str(exc)}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {},
            },
            "request_id": request_id,
        },
    )


# Specific error types for common scenarios
class InvalidImageFormatError(ServiceError):
    """Image format not supported."""

    def __init__(self, received: str, supported: list[str]):
        """Initialize invalid image format error."""
        super().__init__(
            code="INVALID_IMAGE_FORMAT",
            message="The provided image format is not supported",
            details={
                "supported_formats": supported,
                "received": received,
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class ImageTooLargeError(ServiceError):
    """Image exceeds size limit."""

    def __init__(self, max_size_mb: float, received_size_mb: float):
        """Initialize image too large error."""
        super().__init__(
            code="IMAGE_TOO_LARGE",
            message="Image exceeds maximum size limit",
            details={
                "max_size_mb": max_size_mb,
                "received_size_mb": received_size_mb,
            },
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )


class NoPersonDetectedError(ServiceError):
    """No persons detected in image."""

    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize no person detected error."""
        super().__init__(
            code="NO_PERSON_DETECTED",
            message="No persons detected in the provided image",
            details={
                "confidence_threshold": confidence_threshold,
                "detections_count": 0,
            },
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ModelNotLoadedError(ServiceError):
    """Model failed to load."""

    def __init__(self, model_name: str, error_message: str):
        """Initialize model not loaded error."""
        super().__init__(
            code="MODEL_NOT_LOADED",
            message=f"Failed to load model: {model_name}",
            details={"model": model_name, "error": error_message},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class GalleryNotFoundError(ServiceError):
    """Gallery does not exist."""

    def __init__(self, gallery_id: str):
        """Initialize gallery not found error."""
        super().__init__(
            code="GALLERY_NOT_FOUND",
            message="Gallery does not exist",
            details={"gallery_id": gallery_id},
            status_code=status.HTTP_404_NOT_FOUND,
        )


class GalleryAlreadyExistsError(ServiceError):
    """Gallery already exists."""

    def __init__(self, gallery_id: str):
        """Initialize gallery already exists error."""
        super().__init__(
            code="GALLERY_ALREADY_EXISTS",
            message="Gallery already exists",
            details={"gallery_id": gallery_id},
            status_code=status.HTTP_409_CONFLICT,
        )
