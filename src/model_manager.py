"""
Simplified model management for DeepPerson library.

Provides essential downloading, caching, and lifecycle management for:
- ResNet-50 backbone weights from Google Drive
- YOLO detection weights from Ultralytics releases
- Model cache directory management

Features:
- Environment-based cache directory configuration
- Simple file-based validation (no complex structure validation)
- Basic dependency management
- Global model manager instance
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Union

import gdown
import requests

logger = logging.getLogger(__name__)

# A Mapping between Model Name and its weight file path
NAME_WEIGHT_MAPPING = {
    "resnet50_circle_dg": "ft_ResNet50/net_last.pth"
}


class ModelManager:
    """
    Simplified model management interface for DeepPerson.

    Provides unified access to:
    - Cache directory management
    - Backbone weight downloading (ResNet-50)
    - YOLO weight management
    - Basic download utilities
    """

    # Google Drive configuration for ResNet-50 backbone
    WEIGHTS_GOOGLE_DRIVE_ID = "1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE"
    WEIGHTS_ZIP_NAME = "model.zip"
    WEIGHTS_EXTRACTED_DIR = "backbones"

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Cache directory (if None, uses environment variable or default)
        """
        if cache_dir is None:
            # Check DEEP_PERSON_CACHE environment variable first
            env_cache = os.environ.get("DEEP_PERSON_CACHE")
            if env_cache:
                self.cache_dir = Path(env_cache)
                logger.info(f"Using cache directory from DEEP_PERSON_CACHE: {self.cache_dir}")
            else:
                # Default to user's cache directory
                self.cache_dir = Path.home() / ".cache" / "deep_person"
                logger.info(f"Using default cache directory: {self.cache_dir}")
        else:
            self.cache_dir = Path(cache_dir)

        # Create directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Model manager initialized with cache directory: {self.cache_dir}")

        # Initialize YOLO manager
        self.yolo_manager = YOLOManager(self.cache_dir)

    def is_backbone_file_available(self, name: str) -> bool:
        """
        Check if the backbone model file exists based on NAME_WEIGHT_MAPPING.

        Args:
            name: Name of the backbone model

        Returns:
            True if model file exists, False otherwise
        """
        if name not in NAME_WEIGHT_MAPPING:
            logger.error(f"Unknown backbone model: {name}")
            return False

        weight_path = NAME_WEIGHT_MAPPING[name]
        backbone_file = self.cache_dir / self.WEIGHTS_EXTRACTED_DIR / weight_path

        exists = backbone_file.exists() and backbone_file.stat().st_size > 0
        logger.debug(f"Backbone file check for {name} at {backbone_file}: {'found' if exists else 'not found'}")
        return exists

    def download_from_google_drive(self, file_id: str, output_path: Path, filename: str) -> Path:
        """
        Download file from Google Drive using gdown.

        Args:
            file_id: Google Drive file ID
            output_path: Output directory
            filename: Filename to save as

        Returns:
            Path to downloaded file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename
        url = f"https://drive.google.com/uc?id={file_id}"

        logger.info(f"Downloading from Google Drive: file_id={file_id}")
        logger.info(f"Output path: {file_path}")

        try:
            downloaded_path = gdown.download(
                url=url,
                output=str(file_path),
                fuzzy=True,
                quiet=False
            )

            if downloaded_path is None:
                raise RuntimeError("gdown failed to download the file")

            downloaded_path = Path(downloaded_path)
            logger.info(f"Successfully downloaded to: {downloaded_path}")
            return downloaded_path

        except Exception as e:
            logger.error(f"Failed to download from Google Drive: {e}")
            raise

    def extract_zip_archive(self, archive_path: Path, extract_to: Path) -> Path:
        """
        Extract zip archive to specified directory.

        Args:
            archive_path: Path to zip archive
            extract_to: Directory to extract to

        Returns:
            Path to extraction directory
        """
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)

        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        extract_to.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting archive: {archive_path}")

        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            logger.info(f"Successfully extracted to: {extract_to}")
            return extract_to

        except Exception as e:
            logger.error(f"Failed to extract archive: {e}")
            raise

    def ensure_backbone_weights(self, name: str, force_download: bool = False) -> Path:
        """
        Ensure backbone weights are available using simple file validation.

        Args:
            name: Name of the backbone model to ensure
            force_download: Force re-download even if weights exist

        Returns:
            Path to backbone weights file
        """
        if name not in NAME_WEIGHT_MAPPING:
            raise ValueError(f"Unknown backbone model: {name}")

        weight_path = NAME_WEIGHT_MAPPING[name]
        backbone_file = self.cache_dir / self.WEIGHTS_EXTRACTED_DIR / weight_path

        # Check if file exists and is valid
        if not force_download and self.is_backbone_file_available(name):
            logger.info(f"Valid backbone weights found: {backbone_file}")
            return backbone_file

        if force_download:
            logger.info("Force download requested, clearing existing cache")
            if backbone_file.exists():
                backbone_file.unlink()
                logger.info(f"Removed existing backbone file: {backbone_file}")

        logger.info("Downloading ResNet-50 backbone weights from Google Drive")

        # Download zip file
        zip_path = self.cache_dir / self.WEIGHTS_ZIP_NAME
        if not zip_path.exists() or force_download:
            downloaded_path = self.download_from_google_drive(
                file_id=self.WEIGHTS_GOOGLE_DRIVE_ID,
                output_path=self.cache_dir,
                filename=self.WEIGHTS_ZIP_NAME
            )
            zip_path = downloaded_path

        # Extract archive
        try:
            extracted_dir = self.extract_zip_archive(
                archive_path=zip_path,
                extract_to=self.cache_dir
            )
            
            # Move the content in extracted dir into WEIGHTS_EXTRACTED_DIR
            # The zip contains a folder named "model" which we need to move
            os.rename(extracted_dir / "model", self.cache_dir / self.WEIGHTS_EXTRACTED_DIR)
            

            # Clean up zip file
            zip_path.unlink()
            logger.info(f"Removed archive file: {zip_path}")

            # Validate that the expected file exists
            if not self.is_backbone_file_available(name):
                raise FileNotFoundError(
                    f"Expected backbone file not found after extraction: {backbone_file}"
                )

            logger.info(f"Successfully extracted backbone weights. File available at: {backbone_file}")
            return backbone_file

        except Exception as e:
            logger.error(f"Failed to extract backbone weights: {e}")
            raise

    def ensure_yolo_weights(self, model_name: str) -> Path:
        """
        Ensure YOLO weights are available.

        Args:
            model_name: YOLO model name

        Returns:
            Path to YOLO weights file
        """
        return self.yolo_manager.ensure_yolo_weights(model_name)

    def get_cache_info(self) -> dict:
        """
        Get basic information about cache usage.

        Returns:
            Dictionary with cache statistics
        """
        backbone_dir = self.cache_dir / self.WEIGHTS_EXTRACTED_DIR
        backbone_exists = backbone_dir.exists()

        # Check available backbone models
        backbone_models = []
        if backbone_exists:
            for name, weight_path in NAME_WEIGHT_MAPPING.items():
                backbone_file = backbone_dir / weight_path
                if backbone_file.exists():
                    backbone_models.append(name)

        info = {
            "cache_dir": str(self.cache_dir),
            "backbone_models_available": sorted(backbone_models),
            "yolo_models_cached": self.yolo_manager.list_cached_models(),
        }

        return info

    def clear_cache(self, keep_backbone: bool = False) -> None:
        """
        Clear model cache.

        Args:
            keep_backbone: Keep backbone weights
        """
        logger.info("Clearing model cache")

        # Clear YOLO cache
        yolo_cache = self.cache_dir / "detection"
        if yolo_cache.exists():
            cached_models = list(yolo_cache.glob("*.pt"))
            shutil.rmtree(yolo_cache)
            logger.info(f"Cleared YOLO cache ({len(cached_models)} models)")

        # Optionally keep backbone
        backbone_dir = self.cache_dir / self.WEIGHTS_EXTRACTED_DIR
        if not keep_backbone and backbone_dir.exists():
            shutil.rmtree(backbone_dir)
            logger.info("Cleared backbone cache")
        elif keep_backbone and backbone_dir.exists():
            logger.info("Kept backbone cache")

        # Clear any remaining zip files
        zip_files = list(self.cache_dir.glob("*.zip"))
        for zip_file in zip_files:
            zip_file.unlink()
        if zip_files:
            logger.info(f"Cleared {len(zip_files)} archive files")

        logger.info("Model cache clearing completed")


class YOLOManager:
    """
    Simplified YOLO model weights manager.

    Automatically downloads and caches YOLO weights:
    - Supports basic YOLOv8, YOLOv9, YOLOv10, YOLOv11 variants
    - Downloads from GitHub releases
    - Simple cache management
    """

    # Simplified YOLO model configurations
    YOLO_CONFIGS = {
        "yolov8": {"variants": ["n", "s", "m", "l", "x"]},
        "yolov9": {"variants": ["t", "s", "m", "c", "e"]},
        "yolov10": {"variants": ["n", "s", "m", "b", "l", "x"]},
        "yolov11": {"variants": ["n", "s", "m", "l", "x"]}
    }
    BASE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"

    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize YOLO manager.

        Args:
            cache_dir: Cache directory for YOLO weights
        """
        self.cache_dir = Path(cache_dir) / "detection"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ensure_yolo_weights(self, model_name: str) -> Path:
        """
        Ensure YOLO weights are available, download if necessary.

        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt', 'yolov11s.pt')

        Returns:
            Path to YOLO weights file
        """
        model_path = self.cache_dir / model_name

        if model_path.exists():
            logger.debug(f"YOLO weights already cached: {model_path}")
            return model_path

        logger.info(f"YOLO weights not found, downloading: {model_name}")

        # Parse model name to determine family and variant
        model_info = self._parse_model_name(model_name)
        if not model_info:
            raise ValueError(f"Unsupported YOLO model: {model_name}")

        family, variant = model_info
        if family not in self.YOLO_CONFIGS:
            raise ValueError(f"Unsupported YOLO family: {family}")

        config = self.YOLO_CONFIGS[family]
        if variant not in config["variants"]:
            raise ValueError(f"Unsupported {family} variant: {variant}")

        download_url = f"{self.BASE_URL}{model_name}"

        try:
            # Download weights using requests
            response = requests.get(download_url, stream=True)
            response.raise_for_status()

            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Successfully downloaded YOLO weights: {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to download YOLO weights {model_name}: {e}")
            raise

    def _parse_model_name(self, model_name: str) -> Optional[tuple[str, str]]:
        """
        Parse YOLO model name to extract family and variant.

        Args:
            model_name: Model name like 'yolov8n.pt'

        Returns:
            Tuple of (family, variant) or None if not supported
        """
        # Remove extension if present
        name = model_name.replace('.pt', '')

        # Try to match YOLO patterns
        for family in self.YOLO_CONFIGS.keys():
            family_prefix = family
            if name.startswith(family_prefix):
                variant = name[len(family_prefix):]
                if variant and len(variant) == 1:  # Single letter variants
                    return family, variant

        return None

    def list_cached_models(self) -> list[str]:
        """
        List all cached YOLO models.

        Returns:
            List of cached model names
        """
        cached = []
        for file_path in self.cache_dir.glob("*.pt"):
            cached.append(file_path.name)
        return sorted(cached)


# Global model manager instance
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """
    Get global model manager instance.

    Returns:
        Shared ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager