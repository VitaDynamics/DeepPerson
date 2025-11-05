"""FastAPI application factory.

Creates and configures the FastAPI application with middleware, error handlers,
and routes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastapi_service.config import ServiceConfig
from fastapi_service.middleware.cors import get_cors_config
from fastapi_service.middleware.errors import (
    ServiceError,
    generic_exception_handler,
    http_exception_handler,
    service_error_handler,
    validation_exception_handler,
)
from fastapi_service.middleware.logging import LoggingMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown logic.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting FastAPI service...")
    logger.info(f"Device: {app.state.device}")
    logger.info(f"Gallery storage: {app.state.config.gallery_storage_path}")

    yield

    # Shutdown
    logger.info("Shutting down FastAPI service...")


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Service configuration (loads from environment if None)

    Returns:
        Configured FastAPI application
    """
    # Load configuration if not provided
    if config is None:
        from fastapi_service.config import load_config
        config = load_config()

    # Detect device
    device = config.get_device()

    # Create FastAPI app
    app = FastAPI(
        title="DeepPerson API",
        description=(
            "Person re-identification and embedding generation service. "
            "Provides endpoints for person recognition, identity verification, "
            "and gallery management."
        ),
        version=config.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Store configuration and device in app state
    app.state.config = config
    app.state.device = device

    # Add CORS middleware (if enabled)
    if config.enable_cors:
        cors_config = get_cors_config(allow_origins=config.cors_origins)
        app.add_middleware(CORSMiddleware, **cors_config)
        logger.info("CORS middleware enabled")

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    logger.info("Logging middleware enabled")

    # Register error handlers
    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    logger.info("Error handlers registered")

    # Import and register routes
    from fastapi_service.routes import core, health

    app.include_router(health.router)
    app.include_router(core.router)
    logger.info("Routes registered: health, core")

    logger.info(
        f"FastAPI application created: "
        f"version={config.service_version}, device={device}"
    )

    return app
