"""
User Gallery API - Public interface for user gallery operations.

This module provides the main API for creating and managing user galleries
with multi-modal (body + face) person re-identification capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .config import UserGalleryConfig
from .models import GalleryStatus, Modality
from .services import GalleryRegistrationService, GalleryUpdateService
from .storage import GalleryStorageManager

logger = logging.getLogger(__name__)


class UserGalleryAPI:
    """
    Main API facade for user gallery operations.

    Provides high-level methods for:
    - Creating user galleries
    - Adding/updating images
    - Managing gallery metadata
    - Querying gallery information
    """

    def __init__(
        self,
        storage_path: str | Path = "galleries/",
        config: Optional[UserGalleryConfig] = None,
    ):
        """
        Initialize User Gallery API.

        Args:
            storage_path: Base path for gallery storage
            config: Optional configuration for gallery management

        Examples:
            >>> api = UserGalleryAPI(storage_path="./my_galleries")
            >>> gallery, info = api.create_gallery(
            ...     user_id="user_001",
            ...     image_paths=["img1.jpg", "img2.jpg"],
            ...     name="John Doe"
            ... )
        """
        self.storage_path = Path(storage_path)
        self.config = config if config else UserGalleryConfig()

        # Initialize components
        self.storage_manager = GalleryStorageManager(self.storage_path, self.config)
        self.registration_service = GalleryRegistrationService(
            self.storage_manager, self.config
        )
        self.update_service = GalleryUpdateService(self.storage_manager, self.config)

        logger.info(f"Initialized UserGalleryAPI with storage at {self.storage_path}")

    def create_gallery(
        self,
        user_id: str,
        image_paths: list[str | Path],
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        modality_hints: Optional[dict[str, str]] = None,
        enable_clustering: bool = True,
    ) -> dict[str, Any]:
        """
        Create a new user gallery with images.

        This is the primary method for registering a new user with their
        associated images. Images are validated, processed, and organized
        into variant clusters.

        Args:
            user_id: Unique identifier for the user
            image_paths: List of paths to user images (body and/or face)
            name: Optional display name for the user
            metadata: Optional metadata dictionary (e.g., demographics, status)
            modality_hints: Optional dict mapping image paths to modality
                           ("BODY", "FACE", or "UNKNOWN")
            enable_clustering: Whether to automatically cluster images into variants

        Returns:
            Dictionary containing:
                - user_id: User identifier
                - status: Gallery status
                - total_images: Number of images processed
                - clusters_created: Number of variant clusters
                - modality_breakdown: Count of images by modality
                - created_at: Creation timestamp

        Raises:
            ValueError: If user_id already exists or validation fails

        Examples:
            >>> api = UserGalleryAPI()
            >>> result = api.create_gallery(
            ...     user_id="user_001",
            ...     image_paths=["body1.jpg", "body2.jpg", "face1.jpg"],
            ...     name="John Doe",
            ...     metadata={"department": "Security"},
            ...     modality_hints={
            ...         "body1.jpg": "BODY",
            ...         "body2.jpg": "BODY",
            ...         "face1.jpg": "FACE"
            ...     }
            ... )
            >>> print(f"Created gallery with {result['total_images']} images")
        """
        logger.info(f"API: Creating gallery for user '{user_id}'")

        # Convert modality hints to Modality enum
        modality_map = None
        if modality_hints:
            modality_map = {}
            for path, modality_str in modality_hints.items():
                try:
                    modality_map[path] = Modality(modality_str.upper())
                except ValueError:
                    logger.warning(
                        f"Invalid modality '{modality_str}' for {path}, using UNKNOWN"
                    )
                    modality_map[path] = Modality.UNKNOWN

        # Register gallery
        gallery, registration_info = self.registration_service.register_gallery(
            user_id=user_id,
            image_paths=image_paths,
            name=name,
            metadata=metadata,
            modality_hints=modality_map,
            enable_clustering=enable_clustering,
        )

        # Prepare response
        response = {
            "user_id": gallery.user_id,
            "name": gallery.name,
            "status": gallery.status.value,
            "total_images": registration_info["valid_images"],
            "invalid_images": registration_info["invalid_images"],
            "clusters_created": registration_info["clusters_created"],
            "modality_breakdown": registration_info["modality_breakdown"],
            "created_at": registration_info["created_at"],
            "metadata": gallery.metadata,
        }

        logger.info(f"API: Successfully created gallery for user '{user_id}'")
        return response

    def get_gallery(self, user_id: str) -> dict[str, Any]:
        """
        Get detailed information about a user gallery.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with comprehensive gallery information

        Raises:
            ValueError: If gallery doesn't exist

        Examples:
            >>> api = UserGalleryAPI()
            >>> gallery_info = api.get_gallery("user_001")
            >>> print(f"Gallery has {gallery_info['total_images']} images")
        """
        logger.debug(f"API: Getting gallery info for user '{user_id}'")
        return self.registration_service.get_gallery_info(user_id)

    def update_gallery(
        self,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        status: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Update gallery metadata and status.

        Args:
            user_id: User identifier
            name: Optional new display name
            metadata: Optional metadata to merge with existing
            status: Optional new status ("ACTIVE", "INACTIVE", "PENDING_VERIFICATION")

        Returns:
            Updated gallery information

        Raises:
            ValueError: If gallery doesn't exist or invalid status

        Examples:
            >>> api = UserGalleryAPI()
            >>> result = api.update_gallery(
            ...     user_id="user_001",
            ...     name="John Smith",
            ...     status="ACTIVE"
            ... )
        """
        logger.info(f"API: Updating gallery for user '{user_id}'")

        # Convert status string to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = GalleryStatus(status.upper())
            except ValueError:
                raise ValueError(
                    f"Invalid status '{status}'. Must be one of: "
                    "ACTIVE, INACTIVE, PENDING_VERIFICATION"
                )

        # Update gallery
        gallery = self.storage_manager.update_gallery(
            user_id=user_id, name=name, metadata=metadata, status=status_enum
        )

        # Return updated info
        return self.registration_service.get_gallery_info(user_id)

    def add_images(
        self,
        user_id: str,
        image_paths: list[str | Path],
        modality_hints: Optional[dict[str, str]] = None,
        recluster: bool = False,
    ) -> dict[str, Any]:
        """
        Add new images to an existing gallery.

        Args:
            user_id: User identifier
            image_paths: List of new image paths
            modality_hints: Optional modality hints for new images
            recluster: Whether to recluster after adding images

        Returns:
            Dictionary with update information

        Raises:
            ValueError: If gallery doesn't exist or validation fails

        Examples:
            >>> api = UserGalleryAPI()
            >>> result = api.add_images(
            ...     user_id="user_001",
            ...     image_paths=["new_body.jpg", "new_face.jpg"],
            ...     modality_hints={"new_body.jpg": "BODY", "new_face.jpg": "FACE"}
            ... )
            >>> print(f"Added {result['images_added']} images")
        """
        logger.info(f"API: Adding images to gallery '{user_id}'")

        # Convert modality hints
        modality_map = None
        if modality_hints:
            modality_map = {}
            for path, modality_str in modality_hints.items():
                try:
                    modality_map[path] = Modality(modality_str.upper())
                except ValueError:
                    modality_map[path] = Modality.UNKNOWN

        # Add images
        gallery, update_info = self.registration_service.add_images_to_gallery(
            user_id=user_id,
            image_paths=image_paths,
            modality_hints=modality_map,
            recluster=recluster,
        )

        return update_info

    def delete_gallery(self, user_id: str, permanent: bool = False) -> dict[str, Any]:
        """
        Delete a user gallery.

        Args:
            user_id: User identifier
            permanent: If True, permanently delete; if False, mark as inactive

        Returns:
            Dictionary with deletion status

        Raises:
            ValueError: If gallery doesn't exist

        Examples:
            >>> api = UserGalleryAPI()
            >>> result = api.delete_gallery("user_001", permanent=False)
            >>> print(f"Gallery deleted: {result['success']}")
        """
        logger.info(f"API: Deleting gallery '{user_id}' (permanent={permanent})")

        success = self.registration_service.delete_gallery(user_id, permanent=permanent)

        return {
            "user_id": user_id,
            "success": success,
            "permanent": permanent,
            "action": "permanently_deleted" if permanent else "marked_inactive",
        }

    def list_galleries(
        self, status_filter: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        List all galleries with summary information.

        Args:
            status_filter: Optional status to filter by
                          ("ACTIVE", "INACTIVE", "PENDING_VERIFICATION")

        Returns:
            List of gallery summary dictionaries

        Examples:
            >>> api = UserGalleryAPI()
            >>> active_galleries = api.list_galleries(status_filter="ACTIVE")
            >>> print(f"Found {len(active_galleries)} active galleries")
        """
        logger.debug(f"API: Listing galleries (filter={status_filter})")

        # Convert status filter
        status_enum = None
        if status_filter:
            try:
                status_enum = GalleryStatus(status_filter.upper())
            except ValueError:
                raise ValueError(f"Invalid status filter: {status_filter}")

        return self.registration_service.list_all_galleries(status_filter=status_enum)

    def recluster_gallery(
        self, user_id: str, algorithm: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Recluster an existing gallery.

        Requires embeddings to be generated first.

        Args:
            user_id: User identifier
            algorithm: Optional clustering algorithm override

        Returns:
            Dictionary with clustering information

        Raises:
            ValueError: If gallery doesn't exist or no embeddings found

        Examples:
            >>> api = UserGalleryAPI()
            >>> result = api.recluster_gallery("user_001", algorithm="DBSCAN")
            >>> print(f"Created {result['clusters_created']} clusters")
        """
        logger.info(f"API: Reclustering gallery '{user_id}'")

        gallery, clustering_info = self.update_service.recluster_gallery(
            user_id=user_id, algorithm=algorithm
        )

        return clustering_info

    def get_gallery_stats(self, user_id: str) -> dict[str, Any]:
        """
        Get statistics for a user gallery.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with gallery statistics

        Raises:
            ValueError: If gallery doesn't exist

        Examples:
            >>> api = UserGalleryAPI()
            >>> stats = api.get_gallery_stats("user_001")
            >>> print(f"Gallery has {stats['total_embeddings']} embeddings")
        """
        logger.debug(f"API: Getting stats for gallery '{user_id}'")
        return self.storage_manager.get_gallery_stats(user_id)

    def gallery_exists(self, user_id: str) -> bool:
        """
        Check if a gallery exists for the given user.

        Args:
            user_id: User identifier

        Returns:
            True if gallery exists, False otherwise

        Examples:
            >>> api = UserGalleryAPI()
            >>> if api.gallery_exists("user_001"):
            ...     print("Gallery exists")
        """
        return self.storage_manager.gallery_exists(user_id)
