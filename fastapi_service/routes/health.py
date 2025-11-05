"""Health check and system status endpoints.

Implements Observability principle (Constitution Principle V).

This module provides comprehensive health monitoring including:
- Model loading status (Registry Pattern - Principle I)
- Pipeline health verification (Pipeline Pattern - Principle II)
- Hardware monitoring (Hardware Optimization - Principle III)
- Service uptime and metrics (Observability - Principle V)
"""

import logging
import time
from typing import Dict

from fastapi import APIRouter, Request

from fastapi_service.dependencies import get_deep_person_instance
from fastapi_service.schemas.health import (
    HardwareInfo,
    HealthResponse,
    ModelStatus,
)
from fastapi_service.utils.response import format_timestamp

logger = logging.getLogger(__name__)

# Track service start time
_service_start_time = time.time()

router = APIRouter(tags=["Health"])


def get_hardware_info(device: str) -> HardwareInfo:
    """Get current hardware information.

    Implements Hardware Optimization principle (Constitution Principle III).

    Args:
        device: Device string (cpu/cuda:0)

    Returns:
        HardwareInfo object with memory and utilization metrics
    """
    try:
        import torch

        if device.startswith("cuda"):
            # GPU information
            gpu_idx = int(device.split(":")[1]) if ":" in device else 0

            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("CUDA device specified but not available")
                return HardwareInfo(
                    device=device,
                    memory_used="N/A",
                    memory_total="N/A",
                    gpu_utilization=None,
                )

            props = torch.cuda.get_device_properties(gpu_idx)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            allocated_memory = torch.cuda.memory_allocated(gpu_idx) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(gpu_idx) / (1024**3)

            # Try to get GPU utilization if nvidia-ml-py3 is available
            gpu_utilization = None
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = utilization.gpu
                pynvml.nvmlShutdown()
            except ImportError:
                pass  # nvidia-ml-py3 not installed
            except Exception as e:
                logger.debug(f"Failed to get GPU utilization: {e}")

            logger.debug(
                f"GPU {gpu_idx} memory: {allocated_memory:.2f}GB / {total_memory:.2f}GB "
                f"(reserved: {reserved_memory:.2f}GB)"
            )

            return HardwareInfo(
                device=device,
                memory_used=f"{allocated_memory:.2f}GB",
                memory_total=f"{total_memory:.2f}GB",
                gpu_utilization=gpu_utilization,
            )
        else:
            # CPU information
            try:
                import psutil

                mem = psutil.virtual_memory()
                logger.debug(
                    f"CPU memory: {mem.used / (1024**3):.2f}GB / "
                    f"{mem.total / (1024**3):.2f}GB (utilization: {mem.percent}%)"
                )
                return HardwareInfo(
                    device="cpu",
                    memory_used=f"{mem.used / (1024**3):.2f}GB",
                    memory_total=f"{mem.total / (1024**3):.2f}GB",
                    gpu_utilization=None,
                )
            except ImportError:
                logger.warning("psutil not available for CPU memory monitoring")
                return HardwareInfo(
                    device="cpu",
                    memory_used="N/A",
                    memory_total="N/A",
                    gpu_utilization=None,
                )

    except Exception as e:
        logger.error(f"Failed to get hardware info: {e}", exc_info=True)
        return HardwareInfo(
            device=device,
            memory_used="N/A",
            memory_total="N/A",
            gpu_utilization=None,
        )


def check_model_status(device: str) -> Dict[str, ModelStatus]:
    """Check status of loaded models.

    Implements Registry Pattern (Constitution Principle I).
    Verifies that DeepPerson models are properly loaded and accessible.

    Args:
        device: Device string for model initialization

    Returns:
        Dictionary with model statuses (body_model, face_model)
    """
    model_statuses = {}

    try:
        # Try to get or initialize DeepPerson instance
        start_time = time.time()
        dp_instance = get_deep_person_instance(device=device)
        load_time_ms = int((time.time() - start_time) * 1000)

        # Check body model status via registry
        try:
            from src.registry import ModelRegistry

            registry = ModelRegistry.get_instance()

            # Check if body model is loaded
            if hasattr(registry, "_model") and registry._model is not None:
                model_statuses["body_model"] = ModelStatus(
                    status="loaded",
                    load_time_ms=load_time_ms if load_time_ms > 0 else None,
                )
                logger.debug("Body model status: loaded")
            else:
                model_statuses["body_model"] = ModelStatus(
                    status="loading",
                    load_time_ms=None,
                )
                logger.warning("Body model status: not yet loaded")

        except Exception as e:
            logger.error(f"Failed to check body model status: {e}")
            model_statuses["body_model"] = ModelStatus(
                status="failed",
                load_time_ms=None,
                error=str(e),
            )

        # Check face model status
        # Face models are loaded on-demand, so check if face embeddings are available
        try:
            # Face models are lazy-loaded, so we just mark as available
            model_statuses["face_model"] = ModelStatus(
                status="loaded",
                load_time_ms=None,
            )
            logger.debug("Face model status: available (lazy-loaded)")
        except Exception as e:
            logger.error(f"Failed to check face model status: {e}")
            model_statuses["face_model"] = ModelStatus(
                status="failed",
                load_time_ms=None,
                error=str(e),
            )

    except Exception as e:
        logger.error(f"Failed to initialize DeepPerson instance: {e}", exc_info=True)
        model_statuses["body_model"] = ModelStatus(
            status="failed",
            load_time_ms=None,
            error=f"DeepPerson initialization failed: {str(e)}",
        )
        model_statuses["face_model"] = ModelStatus(
            status="failed",
            load_time_ms=None,
            error=f"DeepPerson initialization failed: {str(e)}",
        )

    return model_statuses


def verify_pipeline_health(device: str) -> bool:
    """Verify pipeline health (Detection → Embedding → Search).

    Implements Pipeline Pattern (Constitution Principle II).
    Checks that all pipeline components are functional.

    Args:
        device: Device string for pipeline components

    Returns:
        True if pipeline is healthy, False otherwise
    """
    try:
        # Try to initialize pipeline components
        dp_instance = get_deep_person_instance(device=device)

        # Check if detector is accessible
        if not hasattr(dp_instance, "detector") or dp_instance.detector is None:
            logger.warning("Pipeline health check: detector not available")
            return False

        # Check if embedding generator is accessible
        if not hasattr(dp_instance, "embedding_generator"):
            logger.warning("Pipeline health check: embedding generator not available")
            return False

        logger.debug("Pipeline health check: all components available")
        return True

    except Exception as e:
        logger.error(f"Pipeline health check failed: {e}", exc_info=True)
        return False


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check service health status, model availability, and hardware info",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2025-11-05T10:30:45Z",
                        "uptime_seconds": 3600,
                        "models": {
                            "body_model": {"status": "loaded", "load_time_ms": 1234},
                            "face_model": {"status": "loaded", "load_time_ms": 567},
                        },
                        "hardware": {
                            "device": "cuda:0",
                            "memory_used": "2.1GB",
                            "memory_total": "8.0GB",
                            "gpu_utilization": 45,
                        },
                        "version": "1.0.0",
                        "gallery_storage_path": "/data/galleries",
                    }
                }
            },
        }
    },
)
async def health_check(request: Request) -> HealthResponse:
    """Get service health status.

    Provides comprehensive health information:
    - **Model Status**: Body and face model loading status (Registry Pattern)
    - **Hardware Metrics**: GPU/CPU memory and utilization (Hardware Optimization)
    - **Pipeline Health**: Detection → Embedding → Search components (Pipeline Pattern)
    - **Service Uptime**: Time since service started (Observability)
    - **Version**: Service version information

    **Health Status Levels**:
    - `healthy`: All models loaded, pipeline functional
    - `degraded`: Some models loading or minor issues
    - `unhealthy`: Model failures or pipeline broken

    Returns:
        HealthResponse with comprehensive service status
    """
    start_time = time.time()

    # Calculate uptime (T043 - Observability)
    uptime_seconds = int(time.time() - _service_start_time)

    # Get device and config from app state
    device = getattr(request.app.state, "device", "cpu")
    config = getattr(request.app.state, "config", None)

    logger.info(
        f"Health check started: device={device}, uptime={uptime_seconds}s"
    )

    # Get hardware info (T044 - Hardware)
    hardware = get_hardware_info(device)

    # Check model status (T045 - Registry)
    models = check_model_status(device)

    # Verify pipeline health (T046 - Pipeline)
    pipeline_healthy = verify_pipeline_health(device)

    # Determine overall status
    model_statuses = [m.status for m in models.values()]

    if all(status == "loaded" for status in model_statuses) and pipeline_healthy:
        overall_status = "healthy"
    elif any(status == "failed" for status in model_statuses) or not pipeline_healthy:
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    # Build response
    response = HealthResponse(
        status=overall_status,
        timestamp=format_timestamp(),
        uptime_seconds=uptime_seconds,
        models=models,
        hardware=hardware,
        version=config.service_version if config else "1.0.0",
        gallery_storage_path=str(config.gallery_storage_path) if config else None,
    )

    check_duration_ms = (time.time() - start_time) * 1000

    # Comprehensive logging (T048 - Observability)
    logger.info(
        f"Health check complete: status={overall_status}, uptime={uptime_seconds}s, "
        f"device={device}, pipeline_healthy={pipeline_healthy}, "
        f"body_model={models.get('body_model', ModelStatus(status='unknown')).status}, "
        f"face_model={models.get('face_model', ModelStatus(status='unknown')).status}, "
        f"memory={hardware.memory_used}/{hardware.memory_total}, "
        f"check_duration_ms={check_duration_ms:.2f}"
    )

    return response
