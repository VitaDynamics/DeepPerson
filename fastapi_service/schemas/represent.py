"""Pydantic schemas for /represent endpoint.

This module defines request and response schemas for person embedding generation.
Supports both multipart/form-data and base64-encoded images.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RepresentRequest(BaseModel):
    """Request schema for person embedding generation.

    Supports two input methods:
    1. multipart/form-data with 'image' field (UploadFile)
    2. JSON with base64_image string
    """

    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch identifier for tracking"
    )
    detector_backend: Optional[str] = Field(
        default=None,
        description="Person detection backend (yolo, ultralytics, fasterrcnn, torchvision)"
    )
    normalization: str = Field(
        default="resnet",
        description="Embedding normalization method (base, resnet, circle)"
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=32,
        description="Batch size for embedding generation"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence"
    )
    generate_face_embeddings: bool = Field(
        default=False,
        description="Whether to generate face embeddings"
    )
    face_model_name: str = Field(
        default="Facenet",
        description="Face recognition model name"
    )
    face_detector_backend: str = Field(
        default="opencv",
        description="Face detection backend"
    )
    return_multi_modal: bool = Field(
        default=False,
        description="Return full PersonEmbedding objects"
    )

    @field_validator("detector_backend")
    @classmethod
    def validate_detector_backend(cls, v: Optional[str]) -> Optional[str]:
        """Validate detector backend."""
        if v is not None:
            valid_backends = ["yolo", "ultralytics", "fasterrcnn", "torchvision"]
            if v not in valid_backends:
                raise ValueError(
                    f"detector_backend must be one of {valid_backends}, got '{v}'"
                )
        return v

    @field_validator("normalization")
    @classmethod
    def validate_normalization(cls, v: str) -> str:
        """Validate normalization method."""
        valid_norms = ["base", "resnet", "circle"]
        if v not in valid_norms:
            raise ValueError(
                f"normalization must be one of {valid_norms}, got '{v}'"
            )
        return v


class Base64RepresentRequest(RepresentRequest):
    """Request schema for /represent with base64-encoded image."""

    base64_image: str = Field(
        ...,
        description="Base64-encoded image string"
    )


class EmbeddingMetadata(BaseModel):
    """Metadata for person embedding."""

    bbox: List[int] = Field(
        ...,
        description="Bounding box [x, y, width, height]",
        min_length=4,
        max_length=4
    )
    confidence: float = Field(
        ...,
        description="Detection confidence (0-1)",
        ge=0.0,
        le=1.0
    )
    hardware: Optional[str] = Field(
        default=None,
        description="Hardware device used"
    )
    model_profile_id: Optional[str] = Field(
        default=None,
        description="Model profile identifier"
    )
    normalization: Optional[str] = Field(
        default=None,
        description="Normalization method used"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Generation timestamp"
    )
    source_image: str = Field(
        ...,
        description="Source image identifier"
    )
    modality: str = Field(
        ...,
        description="Modality type (BODY_ONLY, BODY_FACE)"
    )
    face_confidence: Optional[float] = Field(
        default=None,
        description="Face detection confidence",
        ge=0.0,
        le=1.0
    )
    face_bbox: Optional[List[int]] = Field(
        default=None,
        description="Face bounding box [x, y, width, height]",
        min_length=4,
        max_length=4
    )


class PersonEmbedding(BaseModel):
    """Person embedding with body and optional face embedding."""

    embedding: List[float] = Field(
        ...,
        description="Body embedding vector (2048-dim)",
        min_length=2048,
        max_length=2048
    )
    face_embedding: Optional[List[float]] = Field(
        default=None,
        description="Face embedding vector (512-dim)",
        min_length=512,
        max_length=512
    )
    metadata: EmbeddingMetadata = Field(
        ...,
        description="Embedding metadata"
    )
    person_embedding: Optional[dict] = Field(
        default=None,
        description="Full PersonEmbedding object (if return_multi_modal=true)"
    )


class ModelInfo(BaseModel):
    """Model and hardware information."""

    name: str = Field(
        ...,
        description="Model name"
    )
    device: str = Field(
        ...,
        description="Device used (cpu/cuda:0)"
    )
    feature_dim: int = Field(
        ...,
        description="Embedding dimension"
    )
    detector_backend: Optional[str] = Field(
        default=None,
        description="Detector backend used"
    )


class RepresentResponse(BaseModel):
    """Response schema for person embedding generation."""

    subjects: List[PersonEmbedding] = Field(
        ...,
        description="List of detected persons with embeddings"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Warning messages (e.g., no detection)"
    )
    model_info: ModelInfo = Field(
        ...,
        description="Model and hardware information"
    )
    face_model_info: Optional[ModelInfo] = Field(
        default=None,
        description="Face model information (if generated)"
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "subjects": [
                        {
                            "embedding": [0.123, -0.456] + [0.0] * 2046,
                            "face_embedding": [0.789, -0.321] + [0.0] * 510,
                            "metadata": {
                                "bbox": [100, 150, 200, 300],
                                "confidence": 0.95,
                                "modality": "BODY_FACE",
                                "source_image": "image.jpg"
                            }
                        }
                    ],
                    "model_info": {
                        "name": "resnet50_circle_dg",
                        "device": "cuda:0",
                        "feature_dim": 2048
                    },
                    "request_id": "req-12345"
                }
            ]
        }
    }
