"""Custom validation functions for Pydantic schemas."""

import re
from typing import Any


def validate_user_id(value: str) -> str:
    """Validate user_id format.

    Args:
        value: User ID string

    Returns:
        Validated user ID

    Raises:
        ValueError: If user ID format is invalid
    """
    # User ID must be 3-64 characters, alphanumeric + underscore
    if not 3 <= len(value) <= 64:
        raise ValueError("User ID must be 3-64 characters")

    if not re.match(r"^[a-zA-Z0-9_]+$", value):
        raise ValueError(
            "User ID must contain only alphanumeric characters and underscores"
        )

    return value


def validate_gallery_name(value: str) -> str:
    """Validate gallery name format.

    Args:
        value: Gallery name string

    Returns:
        Validated gallery name

    Raises:
        ValueError: If gallery name format is invalid
    """
    # Gallery name must be 0-128 characters
    if len(value) > 128:
        raise ValueError("Gallery name must be 0-128 characters")

    return value


def validate_distance_metric(value: str) -> str:
    """Validate distance metric value.

    Args:
        value: Distance metric string

    Returns:
        Validated distance metric

    Raises:
        ValueError: If distance metric is invalid
    """
    valid_metrics = {"cosine", "euclidean", "euclidean_l2"}

    if value not in valid_metrics:
        raise ValueError(
            f"Invalid distance metric: {value}. "
            f"Must be one of: {', '.join(valid_metrics)}"
        )

    return value


def validate_modality(value: str) -> str:
    """Validate modality value.

    Args:
        value: Modality string

    Returns:
        Validated modality

    Raises:
        ValueError: If modality is invalid
    """
    valid_modalities = {"BODY", "FACE", "BODY_FACE"}

    if value.upper() not in valid_modalities:
        raise ValueError(
            f"Invalid modality: {value}. "
            f"Must be one of: {', '.join(valid_modalities)}"
        )

    return value.upper()


def validate_top_k(value: int) -> int:
    """Validate top_k value for search results.

    Args:
        value: Top K value

    Returns:
        Validated top_k

    Raises:
        ValueError: If top_k is invalid
    """
    if not 1 <= value <= 100:
        raise ValueError("top_k must be between 1 and 100")

    return value


def validate_confidence_threshold(value: float) -> float:
    """Validate confidence threshold value.

    Args:
        value: Confidence threshold

    Returns:
        Validated threshold

    Raises:
        ValueError: If threshold is invalid
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    return value


def validate_fusion_weights(weights: dict[str, float]) -> dict[str, float]:
    """Validate fusion weights dictionary.

    Args:
        weights: Dictionary with modality weights

    Returns:
        Validated weights

    Raises:
        ValueError: If weights are invalid
    """
    # Check required keys
    required_keys = {"face", "body"}
    if not required_keys.issubset(weights.keys()):
        raise ValueError(
            f"Fusion weights must contain keys: {', '.join(required_keys)}"
        )

    # Check weight values
    for modality, weight in weights.items():
        if not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"Weight for {modality} must be between 0.0 and 1.0, got {weight}"
            )

    # Check sum equals 1.0 (with small tolerance)
    total = sum(weights.values())
    if not 0.99 <= total <= 1.01:
        raise ValueError(
            f"Fusion weights must sum to 1.0, got {total:.3f}"
        )

    return weights


def validate_batch_size(value: int, max_batch_size: int = 32) -> int:
    """Validate batch size.

    Args:
        value: Batch size
        max_batch_size: Maximum allowed batch size

    Returns:
        Validated batch size

    Raises:
        ValueError: If batch size is invalid
    """
    if not 1 <= value <= max_batch_size:
        raise ValueError(
            f"Batch size must be between 1 and {max_batch_size}"
        )

    return value
