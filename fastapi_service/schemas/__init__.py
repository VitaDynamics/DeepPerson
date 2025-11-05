"""Pydantic schemas for request/response validation."""

from .represent import (
    RepresentRequest,
    Base64RepresentRequest,
    EmbeddingMetadata,
    PersonEmbedding,
    ModelInfo,
    RepresentResponse,
)
from .verify import (
    VerifyRequest,
    Base64VerifyRequest,
    VerifyResponse,
)
from .errors import (
    ErrorDetail,
    ErrorResponse,
    ErrorCodes,
)
from .health import (
    ModelStatus,
    HardwareInfo,
    HealthResponse,
)

__all__ = [
    # Represent schemas
    "RepresentRequest",
    "Base64RepresentRequest",
    "EmbeddingMetadata",
    "PersonEmbedding",
    "ModelInfo",
    "RepresentResponse",
    # Verify schemas
    "VerifyRequest",
    "Base64VerifyRequest",
    "VerifyResponse",
    # Error schemas
    "ErrorDetail",
    "ErrorResponse",
    "ErrorCodes",
    # Health schemas
    "ModelStatus",
    "HardwareInfo",
    "HealthResponse",
]
