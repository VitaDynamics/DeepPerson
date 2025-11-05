"""CORS middleware configuration.

Implements Observability principle (Constitution Principle V).
"""

from typing import List


def get_cors_config(
    allow_origins: List[str] | str = "*",
    allow_credentials: bool = True,
    allow_methods: List[str] | None = None,
    allow_headers: List[str] | None = None,
) -> dict:
    """Get CORS configuration for FastAPI CORSMiddleware.

    Args:
        allow_origins: List of allowed origins or "*" for all
        allow_credentials: Whether to allow credentials
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers

    Returns:
        Dictionary with CORS configuration
    """
    if allow_methods is None:
        allow_methods = ["*"]
    if allow_headers is None:
        allow_headers = ["*"]

    # Convert string to list if needed
    if isinstance(allow_origins, str):
        if allow_origins == "*":
            origins = ["*"]
        else:
            origins = [origin.strip() for origin in allow_origins.split(",")]
    else:
        origins = allow_origins

    return {
        "allow_origins": origins,
        "allow_credentials": allow_credentials,
        "allow_methods": allow_methods,
        "allow_headers": allow_headers,
    }
