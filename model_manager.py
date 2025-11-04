"""
Model management for DeepPerson library.

Provides automatic downloading, caching, and lifecycle management for:
- ResNet-50 backbone weights from Google Drive
- YOLO detection weights from Ultralytics releases
- Model cache directory management

Features:
- Environment-based cache directory configuration
- Resume capability for interrupted downloads
- Integrity verification
- Automatic dependency management
"""

import hashlib
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import gdown
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CacheDirectoryResolver:
    """
    Resolve model cache directory based on environment variables.

    Priority order:
    1. DEEP_PERSON_CACHE environment variable (if set)
    2. Default: ~/.cache/deep_person
    """

    DEFAULT_CACHE_NAME = "deep_person"

    @classmethod
    def get_cache_dir(cls) -> Path:
        """
        Get the model cache directory.

        Returns:
            Path to cache directory
        """
        # Check DEEP_PERSON_CACHE environment variable first
        env_cache = os.environ.get("DEEP_PERSON_CACHE")
        if env_cache:
            cache_dir = Path(env_cache)
            logger.info(f"Using cache directory from DEEP_PERSON_CACHE: {cache_dir}")
        else:
            # Default to user's cache directory
            cache_dir = Path.home() / ".cache" / cls.DEFAULT_CACHE_NAME
            logger.info(f"Using default cache directory: {cache_dir}")

        # Create directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


class ModelDownloader:
    """
    Download models from various sources with resume capability.

    Supports:
    - Google Drive URLs via gdown
    - Direct HTTP/HTTPS URLs with progress tracking
    - Resume capability for interrupted downloads
    """

    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize model downloader.

        Args:
            cache_dir: Cache directory for downloads
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_from_google_drive(
        self,
        file_id: str,
        output_path: Optional[Path] = None,
        filename: Optional[str] = None,
        resume: bool = True
    ) -> Path:
        """
        Download file from Google Drive using gdown.

        Args:
            file_id: Google Drive file ID
            output_path: Output path (if None, uses cache_dir)
            filename: Filename to save as
            resume: Enable resume capability

        Returns:
            Path to downloaded file
        """
        if output_path is None:
            output_path = self.cache_dir

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Construct Google Drive URL
        url = f"https://drive.google.com/uc?id={file_id}"

        logger.info(f"Downloading from Google Drive: file_id={file_id}")

        try:
            # Use gdown to download with progress tracking
            downloaded_path = gdown.download(
                url=url,
                output=str(output_path),
                fuzzy=True,
                resume=resume,
                quiet=False
            )

            if downloaded_path is None:
                raise RuntimeError("gdown failed to download the file")

            downloaded_path = Path(downloaded_path)

            # Rename if filename specified
            if filename and downloaded_path.name != filename:
                target_path = output_path / filename
                downloaded_path.rename(target_path)
                downloaded_path = target_path

            logger.info(f"Successfully downloaded to: {downloaded_path}")
            return downloaded_path

        except Exception as e:
            logger.error(f"Failed to download from Google Drive: {e}")
            raise

    def download_from_url(
        self,
        url: str,
        output_path: Path,
        filename: Optional[str] = None,
        resume: bool = True
    ) -> Path:
        """
        Download file from direct URL with progress tracking.

        Args:
            url: Direct URL to download
            output_path: Output directory
            filename: Filename to save as
            resume: Enable resume capability

        Returns:
            Path to downloaded file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            # Extract filename from URL
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name

        if not filename:
            raise ValueError("Cannot determine filename from URL")

        file_path = output_path / filename

        # Check if file exists and resume is enabled
        mode = 'ab' if resume and file_path.exists() else 'wb'
        initial_pos = file_path.stat().st_size if mode == 'ab' else 0

        logger.info(f"Downloading from {url}")

        try:
            # Get file info for resume support
            headers = {}
            if resume and file_path.exists():
                headers['Range'] = f'bytes={initial_pos}-'

            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            # Get total file size
            total_size = int(response.headers.get('content-length', 0))
            if resume and file_path.exists():
                total_size += initial_pos

            # Download with progress bar
            with open(file_path, mode) as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=filename,
                    initial=initial_pos
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Successfully downloaded to: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Failed to download from {url}: {e}")
            raise

    def verify_file_integrity(
        self,
        file_path: Path,
        expected_hash: Optional[str] = None,
        hash_algorithm: str = "md5"
    ) -> bool:
        """
        Verify file integrity using hash checksum.

        Args:
            file_path: Path to file to verify
            expected_hash: Expected hash value (if None, just compute hash)
            hash_algorithm: Hash algorithm (md5, sha1, sha256)

        Returns:
            True if file is valid
        """
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        try:
            # Compute hash
            hash_func = getattr(hashlib, hash_algorithm)()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)

            computed_hash = hash_func.hexdigest()

            if expected_hash:
                is_valid = computed_hash == expected_hash.lower()
                if is_valid:
                    logger.info(f"File integrity verified: {file_path}")
                else:
                    logger.error(
                        f"File integrity check failed: {file_path} "
                        f"(expected: {expected_hash}, got: {computed_hash})"
                    )
                return is_valid
            else:
                logger.debug(f"File hash ({hash_algorithm}): {computed_hash}")
                return True

        except Exception as e:
            logger.error(f"Failed to verify file integrity: {e}")
            return False


class ZipExtractor:
    """
    Extract zip archives with progress tracking.
    """

    @staticmethod
    def extract_archive(
        archive_path: Path,
        extract_to: Path,
        delete_after: bool = False
    ) -> Path:
        """
        Extract zip archive to specified directory.

        Args:
            archive_path: Path to zip archive
            extract_to: Directory to extract to
            delete_after: Delete archive after extraction

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
                # Get list of files for progress tracking
                file_list = zip_ref.namelist()
                total_files = len(file_list)

                with tqdm(
                    total=total_files,
                    unit='files',
                    desc=f"Extracting {archive_path.name}"
                ) as pbar:
                    for file in file_list:
                        zip_ref.extract(file, extract_to)
                        pbar.update(1)

            logger.info(f"Successfully extracted to: {extract_to}")

            # Delete archive if requested
            if delete_after:
                archive_path.unlink()
                logger.info(f"Deleted archive: {archive_path}")

            return extract_to

        except Exception as e:
            logger.error(f"Failed to extract archive: {e}")
            raise


class YOLOManager:
    """
    Manage YOLO model weights from Ultralytics.

    Automatically downloads and caches YOLO weights:
    - Supports YOLOv8, YOLOv9, YOLOv10, YOLOv11 variants
    - Downloads from GitHub releases
    - Manages cache directory structure
    """

    # YOLO model configurations
    YOLO_CONFIGS = {
        "yolov8": {
            "variants": ["n", "s", "m", "l", "x"],
            "base_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        },
        "yolov9": {
            "variants": ["t", "s", "m", "c", "e"],
            "base_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        },
        "yolov10": {
            "variants": ["n", "s", "m", "b", "l", "x"],
            "base_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        },
        "yolov11": {
            "variants": ["n", "s", "m", "l", "x"],
            "base_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
        }
    }

    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize YOLO manager.

        Args:
            cache_dir: Cache directory for YOLO weights
        """
        self.cache_dir = Path(cache_dir) / "detection"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = ModelDownloader(self.cache_dir)

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

        # Get download URL
        if family not in self.YOLO_CONFIGS:
            raise ValueError(f"Unsupported YOLO family: {family}")

        config = self.YOLO_CONFIGS[family]
        if variant not in config["variants"]:
            raise ValueError(f"Unsupported {family} variant: {variant}")

        download_url = f"{config['base_url']}{model_name}"

        try:
            # Download weights
            downloaded_path = self.downloader.download_from_url(
                url=download_url,
                output_path=self.cache_dir,
                filename=model_name
            )

            # Verify download
            if not self.downloader.verify_file_integrity(downloaded_path):
                logger.warning(f"YOLO weights integrity check failed: {downloaded_path}")

            logger.info(f"Successfully downloaded YOLO weights: {downloaded_path}")
            return downloaded_path

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

    def list_available_models(self) -> list[str]:
        """
        List all supported YOLO model variants.

        Returns:
            List of supported model names
        """
        models = []
        for family, config in self.YOLO_CONFIGS.items():
            for variant in config["variants"]:
                models.append(f"{family}{variant}.pt")
        return sorted(models)

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


class ModelManager:
    """
    Main model management interface for DeepPerson.

    Provides unified access to:
    - Cache directory management
    - Backbone weight downloading (ResNet-50)
    - YOLO weight management
    - Download and extraction utilities
    """

    # Google Drive configuration for ResNet-50 backbone
    RESNET50_GOOGLE_DRIVE_ID = "1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE"
    RESNET50_ZIP_NAME = "resnet50_circle_dg.zip"
    RESNET50_EXTRACTED_DIR = "backbones"

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize model manager.

        Args:
            cache_dir: Cache directory (if None, uses environment variable or default)
        """
        if cache_dir is None:
            self.cache_dir = CacheDirectoryResolver.get_cache_dir()
        else:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Model manager initialized with cache directory: {self.cache_dir}")

        # Initialize components
        self.downloader = ModelDownloader(self.cache_dir)
        self.yolo_manager = YOLOManager(self.cache_dir)

    def ensure_backbone_weights(self, force_download: bool = False) -> Path:
        """
        Ensure ResNet-50 backbone weights are available.

        Args:
            force_download: Force re-download even if weights exist

        Returns:
            Path to backbone weights directory
        """
        backbone_dir = self.cache_dir / self.RESNET50_EXTRACTED_DIR

        if backbone_dir.exists() and not force_download:
            logger.debug(f"Backbone weights already available: {backbone_dir}")
            return backbone_dir

        logger.info("Downloading ResNet-50 backbone weights from Google Drive")

        # Download zip file
        zip_path = self.cache_dir / self.RESNET50_ZIP_NAME
        if not zip_path.exists() or force_download:
            downloaded_path = self.downloader.download_from_google_drive(
                file_id=self.RESNET50_GOOGLE_DRIVE_ID,
                output_path=self.cache_dir,
                filename=self.RESNET50_ZIP_NAME
            )
            zip_path = downloaded_path

        # Extract archive
        try:
            extracted_dir = ZipExtractor.extract_archive(
                archive_path=zip_path,
                extract_to=backbone_dir,
                delete_after=True
            )

            logger.info(f"Successfully extracted backbone weights to: {extracted_dir}")
            return extracted_dir

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
        Get information about cache usage.

        Returns:
            Dictionary with cache statistics
        """
        info = {
            "cache_dir": str(self.cache_dir),
            "backbone_available": (self.cache_dir / self.RESNET50_EXTRACTED_DIR).exists(),
            "yolo_models_cached": self.yolo_manager.list_cached_models(),
            "cache_size_bytes": 0
        }

        # Calculate total cache size
        try:
            for file_path in self.cache_dir.rglob("*"):
                if file_path.is_file():
                    info["cache_size_bytes"] += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate cache size: {e}")

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
            shutil.rmtree(yolo_cache)
            logger.info("Cleared YOLO cache")

        # Optionally keep backbone
        backbone_dir = self.cache_dir / self.RESNET50_EXTRACTED_DIR
        if not keep_backbone and backbone_dir.exists():
            shutil.rmtree(backbone_dir)
            logger.info("Cleared backbone cache")

        logger.info("Model cache cleared")


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