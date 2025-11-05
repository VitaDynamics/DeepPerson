"""Server startup and shutdown logic."""

import logging
import sys

import uvicorn

from fastapi_service.app import create_app
from fastapi_service.config import load_config

logger = logging.getLogger(__name__)


def run_server() -> None:
    """Start the FastAPI server with Uvicorn.

    Loads configuration, creates the FastAPI app, and starts the Uvicorn server.
    """
    try:
        # Load configuration
        logger.info("Loading service configuration...")
        config = load_config()

        # Create FastAPI application
        logger.info("Creating FastAPI application...")
        app = create_app(config)

        # Start Uvicorn server
        logger.info(
            f"Starting Uvicorn server on {config.host}:{config.port} "
            f"with {config.workers} worker(s)..."
        )

        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            workers=config.workers,
            log_level=config.log_level.lower(),
            access_log=True,
        )

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """Main entry point for the server."""
    run_server()


if __name__ == "__main__":
    main()
