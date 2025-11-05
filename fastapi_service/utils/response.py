"""Response formatting utilities.

Implements Observability principle (Constitution Principle V).
"""

from datetime import datetime
from typing import Any


def format_success_response(
    data: dict[str, Any],
    request_id: str | None = None,
) -> dict[str, Any]:
    """Format successful API response.

    Args:
        data: Response data
        request_id: Optional request ID for tracing

    Returns:
        Formatted response dictionary
    """
    response = {**data}

    if request_id:
        response["request_id"] = request_id

    return response


def format_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as ISO 8601 string.

    Args:
        dt: Datetime to format (defaults to current time)

    Returns:
        ISO 8601 formatted string
    """
    if dt is None:
        dt = datetime.utcnow()
    return dt.isoformat() + "Z"


def format_bytes_to_mb(size_bytes: int) -> float:
    """Convert bytes to megabytes.

    Args:
        size_bytes: Size in bytes

    Returns:
        Size in megabytes (rounded to 2 decimal places)
    """
    return round(size_bytes / (1024 * 1024), 2)


def format_seconds_to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds.

    Args:
        seconds: Time in seconds

    Returns:
        Time in milliseconds (rounded to 2 decimal places)
    """
    return round(seconds * 1000, 2)


def sanitize_path(path: str) -> str:
    """Sanitize file path to prevent directory traversal.

    Args:
        path: File path to sanitize

    Returns:
        Sanitized path

    Raises:
        ValueError: If path contains suspicious patterns
    """
    # Remove potentially dangerous characters and patterns
    if ".." in path or path.startswith("/"):
        raise ValueError("Invalid path: contains directory traversal or absolute path")

    # Remove leading/trailing whitespace
    sanitized = path.strip()

    return sanitized
