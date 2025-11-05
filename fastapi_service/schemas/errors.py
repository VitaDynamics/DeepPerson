"""Pydantic schemas for error responses.

This module defines standardized error response schemas following RFC 7807
Problem Details pattern. All API errors use this consistent format.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Error detail information.

    Attributes:
        code: Machine-readable error code (e.g., INVALID_IMAGE_FORMAT)
        message: Human-readable error message
        details: Additional context for debugging
    """

    code: str = Field(
        ...,
        description="Machine-readable error code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context"
    )


class ErrorResponse(BaseModel):
    """Standardized error response format.

    All API errors use this consistent structure for predictable error handling.
    """

    error: ErrorDetail = Field(
        ...,
        description="Error detail information"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request identifier for tracing"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": {
                        "code": "INVALID_IMAGE_FORMAT",
                        "message": "The provided image format is not supported",
                        "details": {
                            "supported_formats": ["JPEG", "PNG", "WEBP"],
                            "received": "GIF"
                        }
                    },
                    "request_id": "req-12345"
                },
                {
                    "error": {
                        "code": "IMAGE_TOO_LARGE",
                        "message": "Image exceeds maximum size limit",
                        "details": {
                            "max_size_mb": 10,
                            "received_size_mb": 15
                        }
                    },
                    "request_id": "req-67890"
                },
                {
                    "error": {
                        "code": "NO_PERSON_DETECTED",
                        "message": "No persons detected in the provided image",
                        "details": {
                            "confidence_threshold": 0.5,
                            "detections_count": 0
                        }
                    },
                    "request_id": "req-24680"
                }
            ]
        }
    }


# Error code constants for consistent usage
class ErrorCodes:
    """Standard error codes used throughout the API."""

    # Image validation errors
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    INVALID_BASE64_IMAGE = "INVALID_BASE64_IMAGE"
    IMAGE_DECODE_ERROR = "IMAGE_DECODE_ERROR"

    # Detection errors
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
    DETECTION_FAILED = "DETECTION_FAILED"

    # Model errors
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"

    # Gallery errors
    GALLERY_NOT_FOUND = "GALLERY_NOT_FOUND"
    GALLERY_ALREADY_EXISTS = "GALLERY_ALREADY_EXISTS"
    GALLERY_EMPTY = "GALLERY_EMPTY"

    # Request errors
    INVALID_REQUEST = "INVALID_REQUEST"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_PARAMETER = "INVALID_PARAMETER"

    # Server errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
