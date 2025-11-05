"""Main entry point for the FastAPI service.

This module provides the main entry point for running the DeepPerson FastAPI service.

Usage:
    python -m fastapi_service.main
    or
    uvicorn fastapi_service.main:app --host 0.0.0.0 --port 8000
"""

from fastapi_service.app import create_app
from fastapi_service.config import load_config

# Create app instance for uvicorn
config = load_config()
app = create_app(config)


if __name__ == "__main__":
    # If run directly, start the server
    from fastapi_service.server import run_server
    run_server()
