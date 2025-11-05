"""Image validation and processing utilities.

Implements Pipeline Pattern (Constitution Principle II).
"""

import base64
import io
import logging
from pathlib import Path
from typing import BinaryIO

from fastapi import UploadFile
from PIL import Image

from fastapi_service.middleware.errors import (
    ImageTooLargeError,
    InvalidImageFormatError,
)

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Image constraints
MIN_IMAGE_DIMENSION = 32  # pixels
MAX_IMAGE_DIMENSION = 4096  # pixels


def validate_image_format(image: Image.Image, filename: str = "unknown") -> None:
    """Validate image format is supported.

    Args:
        image: PIL Image object
        filename: Original filename for error reporting

    Raises:
        InvalidImageFormatError: If format is not supported
    """
    if image.format not in SUPPORTED_FORMATS:
        logger.warning(
            f"Invalid image format: {image.format} in file {filename}"
        )
        raise InvalidImageFormatError(
            received=image.format or "unknown",
            supported=list(SUPPORTED_FORMATS),
        )


def validate_image_size(
    size_bytes: int,
    max_size_bytes: int,
    filename: str = "unknown",
) -> None:
    """Validate image file size.

    Args:
        size_bytes: Image size in bytes
        max_size_bytes: Maximum allowed size in bytes
        filename: Original filename for error reporting

    Raises:
        ImageTooLargeError: If image exceeds size limit
    """
    if size_bytes > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        received_mb = size_bytes / (1024 * 1024)
        logger.warning(
            f"Image too large: {received_mb:.2f}MB (max: {max_mb:.2f}MB) "
            f"in file {filename}"
        )
        raise ImageTooLargeError(
            max_size_mb=max_mb,
            received_size_mb=received_mb,
        )


def validate_image_dimensions(image: Image.Image, filename: str = "unknown") -> None:
    """Validate image dimensions.

    Args:
        image: PIL Image object
        filename: Original filename for error reporting

    Raises:
        ValueError: If dimensions are invalid
    """
    width, height = image.size

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions too small: {width}x{height} "
            f"(minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})"
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Image dimensions too large: {width}x{height} "
            f"(maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
        )


async def process_upload_file(
    file: UploadFile,
    max_size_bytes: int,
) -> Image.Image:
    """Process and validate uploaded file.

    Args:
        file: FastAPI UploadFile
        max_size_bytes: Maximum allowed file size in bytes

    Returns:
        PIL Image object

    Raises:
        InvalidImageFormatError: If format is not supported
        ImageTooLargeError: If image exceeds size limit
        ValueError: If image dimensions are invalid
    """
    filename = file.filename or "unknown"
    logger.debug(f"Processing upload file: {filename}")

    # Read file content
    content = await file.read()
    size_bytes = len(content)

    # Validate size
    validate_image_size(size_bytes, max_size_bytes, filename)

    # Open and validate image
    image = Image.open(io.BytesIO(content))
    validate_image_format(image, filename)
    validate_image_dimensions(image, filename)

    logger.debug(
        f"Image validated: {filename} "
        f"({image.format}, {image.size[0]}x{image.size[1]}, "
        f"{size_bytes / 1024:.2f}KB)"
    )

    return image


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image.

    Args:
        base64_string: Base64 encoded image string
            (with or without data URI prefix)

    Returns:
        PIL Image object

    Raises:
        InvalidImageFormatError: If format is not supported
        ValueError: If base64 string is invalid
    """
    try:
        # Remove data URI prefix if present
        if base64_string.startswith("data:"):
            # Format: data:image/jpeg;base64,/9j/4AAQ...
            base64_string = base64_string.split(",", 1)[1]

        # Decode base64
        image_bytes = base64.b64decode(base64_string)

        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        validate_image_format(image, "base64_image")
        validate_image_dimensions(image, "base64_image")

        logger.debug(
            f"Base64 image decoded: {image.format}, "
            f"{image.size[0]}x{image.size[1]}"
        )

        return image

    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        raise ValueError(f"Invalid base64 image string: {str(e)}") from e


def save_image_to_file(image: Image.Image, output_path: Path) -> None:
    """Save PIL Image to file.

    Args:
        image: PIL Image object
        output_path: Path to save the image

    Raises:
        IOError: If save fails
    """
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save image
        image.save(output_path)
        logger.debug(f"Image saved to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to save image to {output_path}: {str(e)}")
        raise IOError(f"Failed to save image: {str(e)}") from e


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """Convert PIL Image to bytes.

    Args:
        image: PIL Image object
        format: Output format (JPEG, PNG, WEBP)

    Returns:
        Image bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()
