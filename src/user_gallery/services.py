"""
Gallery registration and management services.

This module provides high-level services for creating and managing user galleries,
coordinating between storage, clustering, and validation components.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .clustering import VariantClusterer, create_single_cluster
from .config import UserGalleryConfig
from .models import GalleryStatus, ImageAsset, Modality, UserGallery, VariantCluster
from .storage import GalleryStorageManager
from .utils import (
    batch_validate_images,
    group_images_by_modality,
    infer_modality_from_path,
    preprocess_image_asset,
)

logger = logging.getLogger(__name__)


class GalleryRegistrationService:
    """
    Service for registering and managing user galleries.

    Coordinates gallery creation, image validation, clustering,
    and storage operations.
    """

    def __init__(
        self,
        storage_manager: GalleryStorageManager,
        config: Optional[UserGalleryConfig] = None,
    ):
        """
        Initialize gallery registration service.

        Args:
            storage_manager: Gallery storage manager instance
            config: Optional configuration
        """
        self.storage_manager = storage_manager
        self.config = config if config else UserGalleryConfig()
        self.clusterer = VariantClusterer(self.config.clustering)

        logger.info("Initialized GalleryRegistrationService")

    def register_gallery(
        self,
        user_id: str,
        image_paths: list[str | Path],
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        modality_hints: Optional[dict[str, Modality]] = None,
        enable_clustering: bool = True,
    ) -> tuple[UserGallery, dict[str, Any]]:
        """
        Register a new user gallery with images.

        This is the main entry point for gallery creation. It performs:
        1. Image validation
        2. ImageAsset creation
        3. Variant clustering (optional)
        4. Gallery creation and storage

        Args:
            user_id: Unique user identifier
            image_paths: List of paths to user images
            name: Optional user display name
            metadata: Optional user metadata
            modality_hints: Optional mapping of image paths to modalities
            enable_clustering: Whether to perform automatic clustering

        Returns:
            Tuple of (UserGallery, registration_info)

        Raises:
            ValueError: If validation fails or user already exists
        """
        logger.info(
            f"Registering gallery for user '{user_id}' with {len(image_paths)} images"
        )

        # Check if user already exists
        if self.storage_manager.gallery_exists(user_id):
            raise ValueError(f"Gallery for user '{user_id}' already exists")

        # Step 1: Validate all images
        valid_paths, invalid_paths = batch_validate_images(image_paths)

        if len(valid_paths) == 0:
            raise ValueError(
                f"No valid images found. {len(invalid_paths)} images failed validation"
            )

        if len(invalid_paths) > 0:
            logger.warning(
                f"{len(invalid_paths)} images failed validation and will be skipped"
            )

        # Step 2: Create ImageAsset objects
        image_assets = []
        failed_assets = []

        for image_path in valid_paths:
            # Determine modality
            if modality_hints and str(image_path) in modality_hints:
                modality = modality_hints[str(image_path)]
            else:
                modality = infer_modality_from_path(image_path)

            # Create ImageAsset
            asset, errors = preprocess_image_asset(
                image_path=image_path,
                user_id=user_id,
                modality=modality,
                metadata={"original_path": str(image_path)},
            )

            if asset:
                image_assets.append(asset)
            else:
                failed_assets.append((image_path, errors))
                logger.warning(f"Failed to create asset for {image_path}: {errors}")

        if len(image_assets) == 0:
            raise ValueError("Failed to create any valid image assets")

        # Step 3: Create variant clusters
        # FIXME: Carefully think about How to process cluster
        if enable_clustering and self.config.clustering.enable_auto_cluster:
            # Note: Clustering requires embeddings, which we don't have yet
            # For now, create a single default cluster
            # Actual clustering will happen after embedding generation
            variant_clusters = [create_single_cluster(image_assets, user_id, "default")]
            logger.info("Created single default cluster (clustering deferred)")
        else:
            # Create single cluster without clustering
            variant_clusters = [create_single_cluster(image_assets, user_id, "all")]
            logger.info("Clustering disabled, created single cluster")

        # Step 4: Create gallery
        gallery = self.storage_manager.create_gallery(
            user_id=user_id,
            name=name,
            metadata=metadata,
            images=image_assets,
            variant_clusters=variant_clusters,
        )

        # Step 5: Save images
        self.storage_manager.save_images(user_id, image_assets)

        # Prepare registration info
        registration_info = {
            "user_id": user_id,
            "total_images_provided": len(image_paths),
            "valid_images": len(image_assets),
            "invalid_images": len(invalid_paths) + len(failed_assets),
            "clusters_created": len(variant_clusters),
            "status": gallery.status.value,
            "created_at": gallery.created_at.isoformat(),
            "modality_breakdown": self._get_modality_breakdown(image_assets),
        }

        logger.info(
            f"Successfully registered gallery for user '{user_id}': "
            f"{len(image_assets)} images, {len(variant_clusters)} clusters"
        )

        return gallery, registration_info

    def add_images_to_gallery(
        self,
        user_id: str,
        image_paths: list[str | Path],
        modality_hints: Optional[dict[str, Modality]] = None,
        recluster: bool = False,
    ) -> tuple[UserGallery, dict[str, Any]]:
        """
        Add new images to an existing gallery.

        Args:
            user_id: User identifier
            image_paths: List of new image paths
            modality_hints: Optional modality hints
            recluster: Whether to recluster after adding images

        Returns:
            Tuple of (updated_gallery, update_info)

        Raises:
            ValueError: If gallery doesn't exist or validation fails
        """
        logger.info(f"Adding {len(image_paths)} images to gallery '{user_id}'")

        # Load existing gallery
        gallery = self.storage_manager.get_gallery(user_id)
        if gallery is None:
            raise ValueError(f"Gallery for user '{user_id}' not found")

        # Load existing images
        existing_images = self.storage_manager.load_images(user_id)

        # Validate and create new image assets
        valid_paths, invalid_paths = batch_validate_images(image_paths)

        new_assets = []
        for image_path in valid_paths:
            modality = (
                modality_hints.get(str(image_path), Modality.UNKNOWN)
                if modality_hints
                else infer_modality_from_path(image_path)
            )

            asset, errors = preprocess_image_asset(
                image_path=image_path,
                user_id=user_id,
                modality=modality,
            )

            if asset:
                new_assets.append(asset)

        if len(new_assets) == 0:
            raise ValueError("No valid images to add")

        # Combine with existing images
        all_images = existing_images + new_assets

        # Update clusters if needed
        if recluster:
            # TODO: Implement reclustering with embeddings
            logger.warning(
                "Reclustering not yet implemented, keeping existing clusters"
            )

        # Save updated images
        self.storage_manager.save_images(user_id, all_images)

        # Update gallery metadata
        gallery = self.storage_manager.update_gallery(
            user_id=user_id,
            metadata={"last_image_addition": datetime.now().isoformat()},
        )

        update_info = {
            "user_id": user_id,
            "images_added": len(new_assets),
            "total_images": len(all_images),
            "invalid_images": len(invalid_paths),
        }

        logger.info(f"Added {len(new_assets)} images to gallery '{user_id}'")
        return gallery, update_info

    def update_gallery_status(self, user_id: str, status: GalleryStatus) -> UserGallery:
        """
        Update gallery status.

        Args:
            user_id: User identifier
            status: New gallery status

        Returns:
            Updated UserGallery

        Raises:
            ValueError: If gallery doesn't exist
        """
        gallery = self.storage_manager.update_gallery(user_id=user_id, status=status)
        logger.info(f"Updated gallery '{user_id}' status to {status.value}")
        return gallery

    def get_gallery_info(self, user_id: str) -> dict[str, Any]:
        """
        Get comprehensive information about a gallery.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with gallery information

        Raises:
            ValueError: If gallery doesn't exist
        """
        gallery = self.storage_manager.get_gallery(user_id)
        if gallery is None:
            raise ValueError(f"Gallery for user '{user_id}' not found")

        images = self.storage_manager.load_images(user_id)
        embeddings = self.storage_manager.load_embeddings(user_id)

        return {
            "user_id": user_id,
            "name": gallery.name,
            "status": gallery.status.value,
            "created_at": gallery.created_at.isoformat(),
            "updated_at": gallery.updated_at.isoformat(),
            "total_clusters": gallery.total_clusters,
            "total_images": len(images),
            "total_embeddings": len(embeddings),
            "modality_breakdown": self._get_modality_breakdown(images),
            "has_face_embeddings": any(emb.has_face_embedding for emb in embeddings),
            "metadata": gallery.metadata,
        }

    def list_all_galleries(
        self, status_filter: Optional[GalleryStatus] = None
    ) -> list[dict[str, Any]]:
        """
        List all galleries with summary information.

        Args:
            status_filter: Optional status to filter by

        Returns:
            List of gallery summary dictionaries
        """
        user_ids = self.storage_manager.list_galleries(status_filter=status_filter)

        summaries = []
        for user_id in user_ids:
            try:
                stats = self.storage_manager.get_gallery_stats(user_id)
                summaries.append(stats)
            except Exception as e:
                logger.warning(f"Failed to get stats for gallery '{user_id}': {e}")

        return summaries

    def delete_gallery(self, user_id: str, permanent: bool = False) -> bool:
        """
        Delete a user gallery.

        Args:
            user_id: User identifier
            permanent: Whether to permanently delete

        Returns:
            True if deleted successfully

        Raises:
            ValueError: If gallery doesn't exist
        """
        success = self.storage_manager.delete_gallery(user_id, permanent=permanent)

        if success:
            action = "permanently deleted" if permanent else "marked as inactive"
            logger.info(f"Gallery '{user_id}' {action}")
        else:
            logger.warning(f"Failed to delete gallery '{user_id}'")

        return success

    def _get_modality_breakdown(self, images: list[ImageAsset]) -> dict[str, int]:
        """
        Get breakdown of images by modality.

        Args:
            images: List of image assets

        Returns:
            Dictionary mapping modality to count
        """
        grouped = group_images_by_modality(images)
        return {
            "body": len(grouped[Modality.BODY]),
            "face": len(grouped[Modality.FACE]),
            "unknown": len(grouped[Modality.UNKNOWN]),
        }


class GalleryUpdateService:
    """
    Service for updating existing galleries.

    Handles operations like adding/removing images, updating metadata,
    and reclustering.
    """

    def __init__(
        self,
        storage_manager: GalleryStorageManager,
        config: Optional[UserGalleryConfig] = None,
    ):
        """
        Initialize gallery update service.

        Args:
            storage_manager: Gallery storage manager instance
            config: Optional configuration
        """
        self.storage_manager = storage_manager
        self.config = config if config else UserGalleryConfig()
        self.clusterer = VariantClusterer(self.config.clustering)

        logger.info("Initialized GalleryUpdateService")

    def recluster_gallery(
        self, user_id: str, algorithm: Optional[str] = None
    ) -> tuple[UserGallery, dict[str, Any]]:
        """
        Recluster an existing gallery.

        Requires embeddings to be generated first.

        Args:
            user_id: User identifier
            algorithm: Optional clustering algorithm override

        Returns:
            Tuple of (updated_gallery, clustering_info)

        Raises:
            ValueError: If gallery doesn't exist or no embeddings found
        """
        logger.info(f"Reclustering gallery '{user_id}'")

        # Load gallery and data
        gallery = self.storage_manager.get_gallery(user_id)
        if gallery is None:
            raise ValueError(f"Gallery for user '{user_id}' not found")

        images = self.storage_manager.load_images(user_id)
        embeddings = self.storage_manager.load_embeddings(user_id)

        if len(embeddings) == 0:
            raise ValueError(
                f"No embeddings found for gallery '{user_id}'. "
                "Generate embeddings first."
            )

        # Perform clustering
        new_clusters = self.clusterer.cluster_images(embeddings, images, user_id)

        # Update gallery with new clusters
        gallery = self.storage_manager.update_gallery(
            user_id=user_id, variant_clusters=new_clusters
        )

        # Save updated images (with cluster assignments)
        self.storage_manager.save_images(user_id, images)

        clustering_info = {
            "user_id": user_id,
            "clusters_created": len(new_clusters),
            "algorithm": algorithm or self.config.clustering.algorithm.value,
            "reclustered_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Reclustered gallery '{user_id}': {len(new_clusters)} clusters created"
        )
        return gallery, clustering_info

    def update_cluster_metadata(
        self,
        user_id: str,
        cluster_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> VariantCluster:
        """
        Update metadata for a specific cluster.

        Args:
            user_id: User identifier
            cluster_id: Cluster identifier
            name: Optional new cluster name
            description: Optional new description

        Returns:
            Updated VariantCluster

        Raises:
            ValueError: If gallery or cluster doesn't exist
        """
        gallery = self.storage_manager.get_gallery(user_id)
        if gallery is None:
            raise ValueError(f"Gallery for user '{user_id}' not found")

        cluster = gallery.get_cluster(cluster_id)
        if cluster is None:
            raise ValueError(f"Cluster '{cluster_id}' not found in gallery '{user_id}'")

        if name is not None:
            cluster.cluster_name = name
        if description is not None:
            cluster.description = description

        # Save updated gallery
        self.storage_manager.update_gallery(
            user_id=user_id, variant_clusters=gallery.variant_clusters
        )

        logger.info(f"Updated cluster '{cluster_id}' metadata for user '{user_id}'")
        return cluster


class EmbeddingGenerationService:
    """
    Service for generating and managing embeddings for user galleries.

    Handles batch processing of body and face embeddings with quality scoring,
    error handling, and audit trail generation.

    Now uses the enhanced core components with multi-modal support.
    """

    def __init__(
        self,
        storage_manager: GalleryStorageManager,
        config: Optional[UserGalleryConfig] = None,
        model_name: str = "resnet50_circle_dg",
        device: Optional[str] = None,
    ):
        """
        Initialize embedding generation service.

        Args:
            storage_manager: Gallery storage manager instance
            config: Optional configuration
            model_name: Body embedding model name (default: "resnet50_circle_dg")
            device: Device to use ("cuda", "cpu", or None for auto-detection)
        """
        self.storage_manager = storage_manager
        self.config = config if config else UserGalleryConfig()
        self.model_name = model_name
        self.device_str = device

        # Lazy initialization of embedding generators
        self._body_embedding_pipeline = None
        self._face_embedding_generator = None
        self._device = None

        logger.info(
            f"Initialized EmbeddingGenerationService: model={model_name}, device={device or 'auto'}"
        )

    def _get_device(self):
        """Get or select device."""
        if self._device is None:
            from ..utils import select_device
            import torch

            if self.device_str:
                self._device = torch.device(self.device_str)
            else:
                self._device = select_device(prefer_cuda=True)
            logger.debug(f"Selected device: {self._device}")
        return self._device

    def _get_body_embedding_pipeline(self):
        """Lazy load body embedding pipeline (now using configurable model)."""
        if self._body_embedding_pipeline is None:
            from ..embeddings import BodyEmbeddingGenerator

            device = self._get_device()
            self._body_embedding_pipeline = BodyEmbeddingGenerator(
                model_name=self.model_name,  # Now configurable!
                device=device
            )
            logger.debug(f"Initialized body embedding pipeline: {self.model_name}")
        return self._body_embedding_pipeline

    def _get_face_embedding_generator(self):
        """Lazy load face embedding generator (now using core implementation)."""
        if self._face_embedding_generator is None:
            # Use core FaceEmbeddingGenerator instead of user_gallery version
            from ..face_embeddings import FaceEmbeddingGenerator

            self._face_embedding_generator = FaceEmbeddingGenerator(
                model_name=self.config.face_embedding_model,
                detector_backend=self.config.face_detector_backend,
                enforce_detection=False,
            )
            logger.debug(
                f"Initialized core face embedding generator: {self.config.face_embedding_model}"
            )
        return self._face_embedding_generator

    def generate_embeddings_for_gallery(
        self,
        user_id: str,
        generate_face_embeddings: bool = False,
        force_regenerate: bool = False,
        batch_size: int = 16,
    ) -> tuple[list[PersonEmbedding], dict[str, Any]]:
        """
        Generate embeddings for all images in a user gallery.

        Args:
            user_id: User identifier
            generate_face_embeddings: Whether to generate face embeddings
            force_regenerate: Force regeneration of existing embeddings
            batch_size: Batch size for processing

        Returns:
            Tuple of (person_embeddings, generation_info)

        Raises:
            ValueError: If gallery doesn't exist

        Examples:
            >>> service = EmbeddingGenerationService(storage_manager)
            >>> embeddings, info = service.generate_embeddings_for_gallery(
            ...     user_id="user_001",
            ...     generate_face_embeddings=True,
            ...     batch_size=32
            ... )
            >>> print(f"Generated {len(embeddings)} embeddings")
            >>> print(f"Face embeddings: {info['face_embeddings_generated']}")
        """
        import time
        from ..detectors import DetectorFactory
        from ..entities import PersonEmbedding
        from ..utils import select_device
        from .models import ProcessingStatus
        from .utils import (
            batch_process_with_error_handling,
            handle_face_detection_failure,
            log_embedding_generation_error,
        )

        start_time = time.time()

        logger.info(
            f"Generating embeddings for gallery '{user_id}' "
            f"(face_embeddings={generate_face_embeddings}, force={force_regenerate})"
        )

        # Load gallery and images
        gallery = self.storage_manager.get_gallery(user_id)
        if gallery is None:
            raise ValueError(f"Gallery for user '{user_id}' not found")

        images = self.storage_manager.load_images(user_id)
        if len(images) == 0:
            raise ValueError(f"No images found for gallery '{user_id}'")

        # Check existing embeddings
        existing_embeddings = self.storage_manager.load_embeddings(user_id)
        existing_image_ids = {emb.source_image_id for emb in existing_embeddings}

        # Filter images to process
        if force_regenerate:
            images_to_process = images
        else:
            images_to_process = [
                img for img in images if img.image_id not in existing_image_ids
            ]

        if len(images_to_process) == 0:
            logger.info(f"No new images to process for gallery '{user_id}'")
            return existing_embeddings, {
                "user_id": user_id,
                "processed_images": 0,
                "generated_embeddings": len(existing_embeddings),
                "face_embeddings_generated": sum(
                    1 for emb in existing_embeddings if emb.has_face_embedding
                ),
                "processing_time_ms": 0,
                "errors": [],
            }

        # Initialize components
        device = select_device(prefer_cuda=True)
        detector = DetectorFactory.create_detector(backend="yolo", device=device)
        body_pipeline = self._get_body_embedding_pipeline()

        # Generate body embeddings
        logger.info(f"Generating body embeddings for {len(images_to_process)} images")

        person_embeddings = []
        errors = []
        face_embeddings_count = 0

        for image_asset in images_to_process:
            try:
                # Update processing status
                image_asset.processing_status = ProcessingStatus.PROCESSING

                # Detect and crop person
                detections = detector.detect(
                    image=image_asset.image_path, confidence_threshold=0.5
                )

                if len(detections) == 0:
                    error_msg = f"No person detected in {image_asset.image_id}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    image_asset.processing_status = ProcessingStatus.FAILED
                    continue

                # Use first detection
                detection = detections[0]
                cropped_persons = detector.crop_persons(
                    image=image_asset.image_path, detections=[detection]
                )

                # Generate body embedding using core pipeline
                body_embeddings = body_pipeline.generate_embeddings_batch(
                    images=cropped_persons,
                    bboxes=[detection.bbox],
                    confidences=[detection.confidence],
                    normalize_method="resnet",
                    source_image_ids=[image_asset.image_id],
                    batch_size=1,
                    show_progress=False,
                )

                if len(body_embeddings) == 0:
                    error_msg = f"Failed to generate body embedding for {image_asset.image_id}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    image_asset.processing_status = ProcessingStatus.FAILED
                    continue

                # Get PersonEmbedding with body embedding
                person_embedding = body_embeddings[0]

                # Generate face embedding if requested
                if generate_face_embeddings:
                    try:
                        face_generator = self._get_face_embedding_generator()

                        # Use core FaceEmbeddingGenerator (returns PersonEmbedding)
                        from PIL import Image
                        pil_image = Image.open(image_asset.image_path)

                        face_emb_result = face_generator.generate_embedding(
                            image=pil_image,
                            bbox=detection.bbox,
                            confidence=detection.confidence,
                            source_image_id=image_asset.image_id,
                        )

                        # Merge face embedding into person_embedding using helper method
                        if face_emb_result.face_embedding is not None:
                            person_embedding = person_embedding.with_face_embedding(
                                face_embedding=face_emb_result.face_embedding,
                                face_confidence=face_emb_result.face_confidence,
                                face_bbox=face_emb_result.face_bbox,
                                user_id=user_id,
                                cluster_id=image_asset.cluster_id,
                                embedding_provider=self.model_name,
                            )

                            face_embeddings_count += 1
                            image_asset.face_detection_confidence = face_emb_result.face_confidence
                            logger.debug(
                                f"Generated face embedding for {image_asset.image_id} "
                                f"(confidence={face_emb_result.face_confidence:.3f})"
                            )
                        else:
                            # No face detected - set user gallery fields manually
                            person_embedding.user_id = user_id
                            person_embedding.cluster_id = image_asset.cluster_id
                            person_embedding.embedding_provider = self.model_name
                            logger.debug(
                                f"No face detected for {image_asset.image_id}"
                            )

                    except Exception as e:
                        # Handle face detection failure
                        should_continue, fallback_msg = handle_face_detection_failure(
                            image_asset.image_path, e, fallback_strategy="use_body_only"
                        )
                        errors.append(fallback_msg)

                        # Log error for audit trail
                        error_record = log_embedding_generation_error(
                            image_id=image_asset.image_id,
                            user_id=user_id,
                            error=e,
                            modality="face",
                            context={
                                "model": self.config.face_embedding_model,
                                "detector": self.config.face_detector_backend,
                            },
                        )
                        logger.debug(f"Face embedding error logged: {error_record}")

                # Compute quality score using PersonEmbedding method
                quality_score = person_embedding.compute_quality_score(
                    detection_confidence=detection.confidence,
                    normalization_check=True,
                )

                # Set quality score and ensure embedding_id is set
                person_embedding.quality_score = quality_score
                if not person_embedding.embedding_id:
                    import uuid
                    person_embedding.embedding_id = f"{user_id}_{uuid.uuid4().hex[:12]}"

                # Add additional metadata
                person_embedding.metadata.update({
                    "detection_confidence": detection.confidence,
                    "face_confidence": person_embedding.face_confidence,
                    "bbox": list(detection.bbox) if detection.bbox else None,
                })

                person_embeddings.append(person_embedding)
                image_asset.processing_status = ProcessingStatus.COMPLETED

                logger.debug(
                    f"Generated embedding for {image_asset.image_id} "
                    f"(quality={quality_score:.3f})"
                )

            except Exception as e:
                error_msg = f"Failed to process {image_asset.image_id}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                image_asset.processing_status = ProcessingStatus.FAILED

        # Save embeddings
        if len(person_embeddings) > 0:
            # Combine with existing embeddings if not force regenerate
            if not force_regenerate:
                all_embeddings = existing_embeddings + person_embeddings
            else:
                all_embeddings = person_embeddings

            self.storage_manager.save_embeddings(user_id, all_embeddings)
            logger.info(f"Saved {len(person_embeddings)} new embeddings for '{user_id}'")

        # Update image statuses
        self.storage_manager.save_images(user_id, images)

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Update gallery metadata with embedding generation audit trail
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": "embedding_generation",
            "processed_images": len(images_to_process),
            "generated_embeddings": len(person_embeddings),
            "face_embeddings_generated": face_embeddings_count,
            "processing_time_ms": processing_time_ms,
            "force_regenerate": force_regenerate,
            "batch_size": batch_size,
            "errors_count": len(errors),
            "success_rate": (
                len(person_embeddings) / len(images_to_process)
                if len(images_to_process) > 0
                else 0.0
            ),
        }

        # Get existing audit trail or create new one
        gallery_metadata = gallery.metadata.copy()
        if "embedding_audit_trail" not in gallery_metadata:
            gallery_metadata["embedding_audit_trail"] = []

        gallery_metadata["embedding_audit_trail"].append(audit_entry)

        # Keep only last 50 audit entries to prevent unbounded growth
        if len(gallery_metadata["embedding_audit_trail"]) > 50:
            gallery_metadata["embedding_audit_trail"] = gallery_metadata[
                "embedding_audit_trail"
            ][-50:]

        # Update last embedding generation timestamp
        gallery_metadata["last_embedding_generation"] = datetime.now().isoformat()
        gallery_metadata["total_embeddings"] = len(all_embeddings) if len(person_embeddings) > 0 else len(existing_embeddings)
        gallery_metadata["face_embeddings_enabled"] = generate_face_embeddings

        # Save updated gallery metadata
        self.storage_manager.update_gallery(user_id=user_id, metadata=gallery_metadata)
        logger.debug(f"Updated gallery metadata with audit trail for '{user_id}'")

        # Prepare generation info
        generation_info = {
            "user_id": user_id,
            "processed_images": len(images_to_process),
            "generated_embeddings": len(person_embeddings),
            "face_embeddings_generated": face_embeddings_count,
            "processing_time_ms": processing_time_ms,
            "errors": errors if errors else None,
            "force_regenerate": force_regenerate,
            "batch_size": batch_size,
        }

        logger.info(
            f"Embedding generation complete for '{user_id}': "
            f"{len(person_embeddings)} embeddings generated in {processing_time_ms}ms"
        )

        return person_embeddings, generation_info

    def regenerate_embeddings_for_images(
        self,
        user_id: str,
        image_ids: list[str],
        generate_face_embeddings: bool = False,
    ) -> tuple[list[PersonEmbedding], dict[str, Any]]:
        """
        Regenerate embeddings for specific images.

        Args:
            user_id: User identifier
            image_ids: List of image IDs to regenerate embeddings for
            generate_face_embeddings: Whether to generate face embeddings

        Returns:
            Tuple of (new_person_embeddings, regeneration_info)

        Raises:
            ValueError: If gallery doesn't exist or images not found
        """
        logger.info(
            f"Regenerating embeddings for {len(image_ids)} images in gallery '{user_id}'"
        )

        # Load gallery and images
        gallery = self.storage_manager.get_gallery(user_id)
        if gallery is None:
            raise ValueError(f"Gallery for user '{user_id}' not found")

        all_images = self.storage_manager.load_images(user_id)
        images_to_process = [img for img in all_images if img.image_id in image_ids]

        if len(images_to_process) == 0:
            raise ValueError(f"No images found with IDs: {image_ids}")

        # Use main generation method with force regenerate
        # This is a simplified approach - in production, you might want more granular control
        return self.generate_embeddings_for_gallery(
            user_id=user_id,
            generate_face_embeddings=generate_face_embeddings,
            force_regenerate=True,
        )

    def get_embedding_statistics(self, user_id: str) -> dict[str, Any]:
        """
        Get statistics about embeddings for a gallery.

        Args:
            user_id: User identifier

        Returns:
            Dictionary with embedding statistics

        Raises:
            ValueError: If gallery doesn't exist
        """
        gallery = self.storage_manager.get_gallery(user_id)
        if gallery is None:
            raise ValueError(f"Gallery for user '{user_id}' not found")

        embeddings = self.storage_manager.load_embeddings(user_id)
        images = self.storage_manager.load_images(user_id)

        # Calculate statistics
        total_embeddings = len(embeddings)
        face_embeddings = sum(1 for emb in embeddings if emb.has_face_embedding)
        avg_quality = (
            sum(emb.quality_score for emb in embeddings) / total_embeddings
            if total_embeddings > 0
            else 0.0
        )

        # Processing status breakdown
        status_breakdown = {
            "completed": sum(
                1 for img in images if img.processing_status == ProcessingStatus.COMPLETED
            ),
            "pending": sum(
                1 for img in images if img.processing_status == ProcessingStatus.PENDING
            ),
            "processing": sum(
                1 for img in images if img.processing_status == ProcessingStatus.PROCESSING
            ),
            "failed": sum(
                1 for img in images if img.processing_status == ProcessingStatus.FAILED
            ),
        }

        return {
            "user_id": user_id,
            "total_images": len(images),
            "total_embeddings": total_embeddings,
            "face_embeddings": face_embeddings,
            "body_only_embeddings": total_embeddings - face_embeddings,
            "average_quality_score": avg_quality,
            "processing_status": status_breakdown,
            "coverage": (
                total_embeddings / len(images) if len(images) > 0 else 0.0
            ),
        }
