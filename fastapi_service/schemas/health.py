"""Pydantic schemas for health check endpoints.

This module defines response schemas for service health monitoring,
including model status, hardware utilization, and system metrics.
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


class ModelStatus(BaseModel):
    """Individual model loading status.

    Attributes:
        status: Model loading status (loaded, loading, failed)
        load_time_ms: Load duration in milliseconds
        error: Error message if failed
    """

    status: Literal["loaded", "loading", "failed"] = Field(
        ...,
        description="Model loading status"
    )
    load_time_ms: Optional[int] = Field(
        default=None,
        description="Load duration in milliseconds"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )


class HardwareInfo(BaseModel):
    """Hardware utilization information.

    Attributes:
        device: Primary device (cpu/cuda:0)
        memory_used: Memory used (e.g., '2.1GB')
        memory_total: Total memory (e.g., '8.0GB')
        gpu_utilization: GPU utilization percentage (0-100)
    """

    device: str = Field(
        ...,
        description="Primary device (cpu/cuda:0)"
    )
    memory_used: str = Field(
        ...,
        description="Memory used (e.g., '2.1GB')"
    )
    memory_total: str = Field(
        ...,
        description="Total memory (e.g., '8.0GB')"
    )
    gpu_utilization: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="GPU utilization percentage"
    )


class HealthResponse(BaseModel):
    """Service health status response.

    Provides comprehensive health information including:
    - Overall service status
    - Model loading status
    - Hardware utilization
    - Service uptime
    - Version information
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall service status"
    )
    timestamp: str = Field(
        ...,
        description="Health check timestamp (ISO 8601)"
    )
    uptime_seconds: int = Field(
        ...,
        description="Service uptime in seconds"
    )
    models: Dict[str, ModelStatus] = Field(
        ...,
        description="Model loading status (body_model, face_model)"
    )
    hardware: HardwareInfo = Field(
        ...,
        description="Hardware utilization information"
    )
    version: str = Field(
        ...,
        description="Service version"
    )
    gallery_storage_path: Optional[str] = Field(
        default=None,
        description="Configured gallery storage path"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2025-11-05T10:30:45Z",
                    "uptime_seconds": 3600,
                    "models": {
                        "body_model": {
                            "status": "loaded",
                            "load_time_ms": 1234
                        },
                        "face_model": {
                            "status": "loaded",
                            "load_time_ms": 567
                        }
                    },
                    "hardware": {
                        "device": "cuda:0",
                        "memory_used": "2.1GB",
                        "memory_total": "8.0GB",
                        "gpu_utilization": 45
                    },
                    "version": "1.0.0",
                    "gallery_storage_path": "/data/galleries"
                }
            ]
        }
    }
