"""
Gallery storage and persistence layer for user galleries.

This module provides high-level storage operations for user galleries,
managing the complete lifecycle of gallery data including creation,
updates, retrieval, and deletion.
"""

from __future__ import annotations

import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..entities import PersonEmbedding
from .config import UserGalleryConfig
from .models import (
    GalleryStatus,
    ImageAsset,
    UserGallery,
    VariantCluster,
)
from .utils import (
    create_gallery_storage_structure,
    deserialize_person_embeddings,
    deserialize_image_assets,
    deserialize_user_gallery,
    enforce_gallery_business_rules,
    serialize_person_embeddings,
    serialize_image_assets,
    serialize_user_gallery,
)

logger = logging.getLogger(__name__)


class GalleryStorageManager:
    """
    Manages storage and persistence operations for user galleries.

    Provides thread-safe operations for creating, reading, updating,
    and deleting user galleries with complete data integrity.
    """

    def __init__(
        self, base_storage_path: Path, config: Optional[UserGalleryConfig] = None
    ):
        """
        Initialize gallery storage manager.

        Args:
            base_storage_path: Base directory for all gallery storage
            config: Optional configuration for gallery management

        Raises:
            OSError: If base storage path cannot be created
        """
        self.base_storage_path = Path(base_storage_path)
        self.base_storage_path.mkdir(parents=True, exist_ok=True)

        self.config = config if config else UserGalleryConfig()
        self._lock = threading.RLock()

        # Cache for loaded galleries (user_id -> UserGallery)
        self._gallery_cache: dict[str, UserGallery] = {}

        logger.info(f"Initialized GalleryStorageManager at {base_storage_path}")

    def create_gallery(
        self,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        images: Optional[list[ImageAsset]] = None,
        variant_clusters: Optional[list[VariantCluster]] = None,
    ) -> UserGallery:
        """
        Create a new user gallery with initial data.

        Args:
            user_id: Unique user identifier
            name: Optional user display name
            metadata: Optional user metadata
            images: Optional initial image assets
            variant_clusters: Optional initial variant clusters

        Returns:
            Created UserGallery instance

        Raises:
            ValueError: If user_id already exists or validation fails
            OSError: If storage creation fails
        """
        with self._lock:
            # Check if gallery already exists
            if self.gallery_exists(user_id):
                raise ValueError(f"Gallery for user '{user_id}' already exists")

            # Create gallery instance
            gallery = UserGallery(
                user_id=user_id,
                name=name,
                metadata=metadata or {},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=GalleryStatus.PENDING_VERIFICATION,
                variant_clusters=variant_clusters or [],
            )

            # Validate gallery
            is_valid, errors = enforce_gallery_business_rules(
                gallery,
                max_clusters=self.config.max_clusters_per_gallery,
                max_images=self.config.max_images_per_gallery,
            )

            if not is_valid:
                raise ValueError(f"Gallery validation failed: {errors}")

            # Create storage structure
            storage_structure = create_gallery_storage_structure(
                self.base_storage_path, user_id
            )

            # Serialize gallery metadata
            serialize_user_gallery(
                gallery, storage_structure["metadata"], include_images=False
            )

            # Save images if provided
            if images:
                images_file = storage_structure["images"] / "image_assets.json"
                serialize_image_assets(images, images_file)

            # Cache the gallery
            self._gallery_cache[user_id] = gallery

            logger.info(
                f"Created gallery for user '{user_id}' with {len(variant_clusters or [])} clusters"
            )
            return gallery

    def get_gallery(
        self, user_id: str, use_cache: bool = True
    ) -> Optional[UserGallery]:
        """
        Retrieve a user gallery by ID.

        Args:
            user_id: User identifier
            use_cache: Whether to use cached gallery if available

        Returns:
            UserGallery instance or None if not found

        Raises:
            FileNotFoundError: If gallery directory doesn't exist
            ValueError: If gallery data is corrupted
        """
        with self._lock:
            # Check cache first
            if use_cache and user_id in self._gallery_cache:
                logger.debug(f"Retrieved gallery '{user_id}' from cache")
                return self._gallery_cache[user_id]

            # Load from disk
            gallery_dir = self.base_storage_path / user_id

            if not gallery_dir.exists():
                logger.warning(f"Gallery directory not found for user '{user_id}'")
                return None

            metadata_dir = gallery_dir / "metadata"
            gallery = deserialize_user_gallery(metadata_dir)

            # Cache the loaded gallery
            self._gallery_cache[user_id] = gallery

            logger.debug(f"Loaded gallery '{user_id}' from disk")
            return gallery

    def update_gallery(
        self,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        status: Optional[GalleryStatus] = None,
        variant_clusters: Optional[list[VariantCluster]] = None,
    ) -> UserGallery:
        """
        Update an existing user gallery.

        Args:
            user_id: User identifier
            name: Optional new name
            metadata: Optional metadata to merge
            status: Optional new status
            variant_clusters: Optional new variant clusters

        Returns:
            Updated UserGallery instance

        Raises:
            ValueError: If gallery doesn't exist or validation fails
        """
        with self._lock:
            # Load existing gallery
            gallery = self.get_gallery(user_id, use_cache=False)

            if gallery is None:
                raise ValueError(f"Gallery for user '{user_id}' not found")

            # Update fields
            if name is not None:
                gallery.name = name

            if metadata is not None:
                gallery.metadata.update(metadata)

            if status is not None:
                gallery.status = status

            if variant_clusters is not None:
                gallery.variant_clusters = variant_clusters

            gallery.updated_at = datetime.now()

            # Validate updated gallery
            is_valid, errors = enforce_gallery_business_rules(
                gallery,
                max_clusters=self.config.max_clusters_per_gallery,
                max_images=self.config.max_images_per_gallery,
            )

            if not is_valid:
                raise ValueError(f"Gallery validation failed: {errors}")

            # Save updated gallery
            gallery_dir = self.base_storage_path / user_id
            metadata_dir = gallery_dir / "metadata"
            serialize_user_gallery(gallery, metadata_dir, include_images=False)

            # Update cache
            self._gallery_cache[user_id] = gallery

            logger.info(f"Updated gallery for user '{user_id}'")
            return gallery

    def delete_gallery(self, user_id: str, permanent: bool = False) -> bool:
        """
        Delete a user gallery.

        Args:
            user_id: User identifier
            permanent: If True, permanently delete; if False, mark as inactive

        Returns:
            True if deleted successfully, False if not found

        Raises:
            OSError: If permanent deletion fails
        """
        with self._lock:
            if not self.gallery_exists(user_id):
                logger.warning(f"Gallery '{user_id}' not found for deletion")
                return False

            if permanent:
                # Permanently delete from disk
                gallery_dir = self.base_storage_path / user_id
                shutil.rmtree(gallery_dir)

                # Remove from cache
                if user_id in self._gallery_cache:
                    del self._gallery_cache[user_id]

                logger.info(f"Permanently deleted gallery '{user_id}'")
            else:
            # Mark as inactive
                self.update_gallery(user_id, status=GalleryStatus.INACTIVE)
                logger.info(f"Marked gallery '{user_id}' as inactive")

            return True

    def gallery_exists(self, user_id: str) -> bool:
        """
        Check if a gallery exists for the given user ID.

        Args:
            user_id: User identifier

        Returns:
            True if gallery exists, False otherwise
        """
        gallery_dir = self.base_storage_path / user_id
        return gallery_dir.exists() and (gallery_dir / "metadata").exists()

    def list_galleries(
        self, status_filter: Optional[GalleryStatus] = None
    ) -> list[str]:
        """
        List all gallery user IDs.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of user IDs
        """
        with self._lock:
            user_ids = []

            for item in self.base_storage_path.iterdir():
                if item.is_dir() and (item / "metadata").exists():
                    if status_filter is not None:
                        # Load gallery to check status
                        gallery = self.get_gallery(item.name, use_cache=True)
                        if gallery and gallery.status == status_filter:
                            user_ids.append(item.name)
                    else:
                        user_ids.append(item.name)

            return sorted(user_ids)

    def save_images(self, user_id: str, images: list[ImageAsset]) -> None:
        """
        Save image assets for a user gallery.

        Args:
            user_id: User identifier
            images: List of image assets to save

        Raises:
            ValueError: If gallery doesn't exist
        """
        with self._lock:
            if not self.gallery_exists(user_id):
                raise ValueError(f"Gallery for user '{user_id}' not found")

            gallery_dir = self.base_storage_path / user_id
            images_file = gallery_dir / "images" / "image_assets.json"

            serialize_image_assets(images, images_file)
            logger.debug(f"Saved {len(images)} image assets for user '{user_id}'")

    def load_images(self, user_id: str) -> list[ImageAsset]:
        """
        Load image assets for a user gallery.

        Args:
            user_id: User identifier

        Returns:
            List of image assets

        Raises:
            ValueError: If gallery doesn't exist
            FileNotFoundError: If images file doesn't exist
        """
        with self._lock:
            if not self.gallery_exists(user_id):
                raise ValueError(f"Gallery for user '{user_id}' not found")

            gallery_dir = self.base_storage_path / user_id
            images_file = gallery_dir / "images" / "image_assets.json"

            if not images_file.exists():
                logger.debug(f"No image assets found for user '{user_id}'")
                return []

            images = deserialize_image_assets(images_file)
            logger.debug(f"Loaded {len(images)} image assets for user '{user_id}'")
            return images

    def save_embeddings(self, user_id: str, embeddings: list[PersonEmbedding]) -> None:
        """
        Save person embeddings for a user gallery.

        Args:
            user_id: User identifier
            embeddings: List of person embeddings to save

        Raises:
            ValueError: If gallery doesn't exist
        """
        with self._lock:
            if not self.gallery_exists(user_id):
                raise ValueError(f"Gallery for user '{user_id}' not found")

            gallery_dir = self.base_storage_path / user_id
            embeddings_dir = gallery_dir / "embeddings"

            serialize_person_embeddings(embeddings, embeddings_dir, save_format="npz")
            logger.debug(f"Saved {len(embeddings)} person embeddings for user '{user_id}'")

    def load_embeddings(self, user_id: str) -> list[PersonEmbedding]:
        """
        Load person embeddings for a user gallery.

        Args:
            user_id: User identifier

        Returns:
            List of person embeddings

        Raises:
            ValueError: If gallery doesn't exist
        """
        with self._lock:
            if not self.gallery_exists(user_id):
                raise ValueError(f"Gallery for user '{user_id}' not found")

            gallery_dir = self.base_storage_path / user_id
            embeddings_dir = gallery_dir / "embeddings"

            # Check if embeddings exist
            embeddings_file = embeddings_dir / "embeddings.npz"
            if not embeddings_file.exists():
                logger.debug(f"No embeddings found for user '{user_id}'")
                return []

            embeddings = deserialize_person_embeddings(embeddings_dir, load_format="npz")
            logger.debug(
                f"Loaded {len(embeddings)} person embeddings for user '{user_id}'"
            )
            return embeddings

    def get_gallery_stats(self, user_id: str) -> dict[str, Any]:
        """
        Get statistics for a user gallery.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with gallery statistics

        Raises:
            ValueError: If gallery doesn't exist
        """
        with self._lock:
            gallery = self.get_gallery(user_id)

            if gallery is None:
                raise ValueError(f"Gallery for user '{user_id}' not found")

            images = self.load_images(user_id)
            embeddings = self.load_embeddings(user_id)

            return {
                "user_id": user_id,
                "name": gallery.name,
                "status": gallery.status.value,
                "created_at": gallery.created_at.isoformat(),
                "updated_at": gallery.updated_at.isoformat(),
                "total_clusters": gallery.total_clusters,
                "total_images": len(images),
                "total_embeddings": len(embeddings),
                "has_face_embeddings": any(
                    emb.has_face_embedding for emb in embeddings
                ),
            }

    def clear_cache(self, user_id: Optional[str] = None) -> None:
        """
        Clear the gallery cache.

        Args:
            user_id: Optional specific user ID to clear; if None, clears all
        """
        with self._lock:
            if user_id is not None:
                if user_id in self._gallery_cache:
                    del self._gallery_cache[user_id]
                    logger.debug(f"Cleared cache for gallery '{user_id}'")
            else:
                self._gallery_cache.clear()
                logger.debug("Cleared all gallery cache")

    def get_storage_path(self, user_id: str) -> Path:
        """
        Get the storage path for a user gallery.

        Args:
            user_id: User identifier

        Returns:
            Path to gallery storage directory
        """
        return self.base_storage_path / user_id
