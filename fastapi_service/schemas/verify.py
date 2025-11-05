"""Pydantic schemas for /verify endpoint.

This module defines request and response schemas for identity verification.
Supports both multipart/form-data and base64-encoded images.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class VerifyRequest(BaseModel):
    """Request schema for identity verification.

    Supports two input methods:
    1. multipart/form-data with 'img1_path' and 'img2_path' fields (UploadFile)
    2. JSON with base64_img1 and base64_img2 strings
    """

    model_name: Optional[str] = Field(
        default=None,
        description="Override model name"
    )
    detector_backend: Optional[str] = Field(
        default=None,
        description="Override detector backend (yolo, ultralytics, fasterrcnn, torchvision)"
    )
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for comparison (cosine, euclidean, euclidean_l2)"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Verification threshold (None for model default)"
    )
    normalization: str = Field(
        default="resnet",
        description="Embedding normalization method (base, resnet, circle)"
    )
    enforce_detection: bool = Field(
        default=True,
        description="Require person detection or return unverified"
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

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        valid_metrics = ["cosine", "euclidean", "euclidean_l2"]
        if v not in valid_metrics:
            raise ValueError(
                f"distance_metric must be one of {valid_metrics}, got '{v}'"
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


class Base64VerifyRequest(VerifyRequest):
    """Request schema for /verify with base64-encoded images."""

    base64_img1: str = Field(
        ...,
        description="First image as base64 string"
    )
    base64_img2: str = Field(
        ...,
        description="Second image as base64 string"
    )


class VerifyResponse(BaseModel):
    """Response schema for identity verification."""

    verified: bool = Field(
        ...,
        description="Whether the images show the same person"
    )
    distance: float = Field(
        ...,
        description="Computed distance between embeddings (backward compatibility)"
    )
    threshold: float = Field(
        ...,
        description="Threshold used for verification"
    )
    distance_metric: str = Field(
        ...,
        description="Metric used for distance calculation"
    )
    model: str = Field(
        ...,
        description="Model name used"
    )
    detector_backend: str = Field(
        ...,
        description="Detector backend used"
    )
    facial_areas: Optional[Dict[str, List[int]]] = Field(
        default=None,
        description="Bounding boxes for both images"
    )
    body_distance: float = Field(
        ...,
        description="Body embedding distance"
    )
    face_distance: Optional[float] = Field(
        default=None,
        description="Face embedding distance (if available)"
    )
    fusion_score: Optional[float] = Field(
        default=None,
        description="Multi-modal fusion similarity score (0-1)"
    )
    face_weight: Optional[float] = Field(
        default=None,
        description="Actual face weight used in fusion"
    )
    body_weight: Optional[float] = Field(
        default=None,
        description="Actual body weight used in fusion"
    )
    used_fusion: bool = Field(
        ...,
        description="Whether fusion scoring was used"
    )
    modality_available: Dict[str, bool] = Field(
        ...,
        description="Available modalities"
    )
    warnings: Optional[List[str]] = Field(
        default=None,
        description="Warning messages"
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "verified": True,
                    "distance": 0.234,
                    "threshold": 0.5,
                    "distance_metric": "cosine",
                    "model": "resnet50_circle_dg",
                    "detector_backend": "yolo",
                    "facial_areas": {
                        "img1": [100, 150, 200, 300],
                        "img2": [110, 160, 210, 310]
                    },
                    "body_distance": 0.234,
                    "face_distance": 0.123,
                    "fusion_score": 0.87,
                    "face_weight": 0.6,
                    "body_weight": 0.4,
                    "used_fusion": True,
                    "modality_available": {
                        "body": True,
                        "face": True
                    },
                    "request_id": "req-67890"
                }
            ]
        }
    }
