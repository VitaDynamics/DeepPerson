"""Service configuration with environment-based settings.

Implements Hardware Optimization principle (Constitution Principle III).
"""

import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class ServiceConfig(BaseSettings):
    """FastAPI service configuration.

    Loads settings from environment variables with fallback defaults.
    """

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of Uvicorn workers")

    # Storage Configuration
    gallery_storage_path: Path = Field(
        default=Path("./galleries"),
        description="Gallery data storage directory",
    )

    # Image Processing Configuration
    max_image_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum image upload size in bytes",
    )
    max_batch_size: int = Field(
        default=32,
        description="Maximum batch size for image processing",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # CORS Configuration
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS middleware",
    )
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed origins",
    )

    # Model Configuration
    model_cache_enabled: bool = Field(
        default=True,
        description="Enable model caching",
    )

    # Service Metadata
    service_version: str = Field(
        default="1.0.0",
        description="Service version",
    )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def setup_logging(self) -> None:
        """Configure logging based on log_level setting."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        logger.info(f"Logging configured at {self.log_level} level")

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        # Create gallery storage directory
        self.gallery_storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Gallery storage path: {self.gallery_storage_path}")

        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        logger.info(f"Logs directory: {logs_dir}")

    def get_device(self) -> str:
        """Detect and return available device (GPU or CPU).

        Implements Hardware Optimization principle.

        Returns:
            Device string: "cuda:0" if GPU available, "cpu" otherwise
        """
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda:0"
                logger.info(
                    f"GPU detected: {torch.cuda.get_device_name(0)} "
                    f"(Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)"
                )
            else:
                device = "cpu"
                logger.info("No GPU detected, using CPU")
        except ImportError:
            device = "cpu"
            logger.warning("PyTorch not available, defaulting to CPU")

        return device

    def validate_config(self) -> None:
        """Validate configuration settings.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate port
        if not (1024 <= self.port <= 65535):
            raise ValueError(f"Invalid port: {self.port} (must be 1024-65535)")

        # Validate workers
        if self.workers < 1:
            raise ValueError(f"Invalid workers count: {self.workers} (must be >= 1)")

        # Validate max_image_size
        if self.max_image_size <= 0:
            raise ValueError(f"Invalid max_image_size: {self.max_image_size}")

        # Validate max_batch_size
        if self.max_batch_size <= 0:
            raise ValueError(f"Invalid max_batch_size: {self.max_batch_size}")

        logger.info("Configuration validation passed")


def load_config() -> ServiceConfig:
    """Load and validate service configuration.

    Returns:
        Validated ServiceConfig instance
    """
    config = ServiceConfig()
    config.validate_config()
    config.setup_logging()
    config.ensure_directories()

    logger.info(
        f"Service configuration loaded: "
        f"host={config.host}, port={config.port}, "
        f"workers={config.workers}, log_level={config.log_level}"
    )

    return config
