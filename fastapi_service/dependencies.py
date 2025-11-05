"""FastAPI dependency injection for DeepPerson instance management.

Implements Registry Pattern (Constitution Principle I).
"""

import logging
from typing import Annotated

from fastapi import Depends, Request

from src.api import DeepPerson

logger = logging.getLogger(__name__)

# Global DeepPerson instance (singleton pattern)
_deep_person_instance: DeepPerson | None = None


def get_deep_person_instance(device: str | None = None) -> DeepPerson:
    """Get or create DeepPerson instance (singleton).

    This implements the Registry Pattern by maintaining a single instance
    of DeepPerson with centralized model management.

    Args:
        device: Device to use ("cpu" or "cuda:0")

    Returns:
        DeepPerson instance
    """
    global _deep_person_instance

    if _deep_person_instance is None:
        logger.info(f"Initializing DeepPerson instance with device: {device}")
        _deep_person_instance = DeepPerson(
            detector_name="yolo",
            device=device,
        )
        logger.info("DeepPerson instance initialized successfully")

    return _deep_person_instance


def get_deep_person(request: Request) -> DeepPerson:
    """FastAPI dependency for DeepPerson instance.

    Args:
        request: FastAPI request

    Returns:
        DeepPerson instance
    """
    # Get device from request state (set during app initialization)
    device = getattr(request.app.state, "device", "cpu")
    return get_deep_person_instance(device=device)


def get_request_id(request: Request) -> str:
    """Extract request ID from request state.

    Args:
        request: FastAPI request

    Returns:
        Request ID string
    """
    return getattr(request.state, "request_id", "unknown")


# Type aliases for dependency injection
DeepPersonDep = Annotated[DeepPerson, Depends(get_deep_person)]
RequestIDDep = Annotated[str, Depends(get_request_id)]


def reset_deep_person_instance() -> None:
    """Reset DeepPerson instance (useful for testing).

    WARNING: This should only be used in test environments.
    """
    global _deep_person_instance
    _deep_person_instance = None
    logger.warning("DeepPerson instance has been reset")
