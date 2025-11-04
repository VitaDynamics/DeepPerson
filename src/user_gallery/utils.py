"""
Utility functions for user gallery management and serialization.

This module provides utilities for serializing/deserializing user galleries with
multi-modal embeddings, validation helpers, and storage management for the
user gallery fusion component.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    EmbeddingSet,
    GalleryStatus,
    ImageAsset,
    Modality,
    ProcessingStatus,
    UserGallery,
    VariantCluster,
)

logger = logging.getLogger(__name__)


# ==================== Serialization Utilities ====================


def serialize_user_gallery(
    gallery: UserGallery, output_dir: Path, include_images: bool = False
) -> Path:
    """
    Serialize UserGallery to disk with all associated data.

    Creates a directory structure:
    {output_dir}/{user_id}/
        ├── gallery_metadata.json      # Gallery metadata and structure
        ├── variant_clusters.json      # Cluster definitions
        └── embeddings/                # Embedding data (if any)

    Args:
        gallery: UserGallery instance to serialize
        output_dir: Base directory for gallery storage
        include_images: Whether to copy image files (not implemented)

    Returns:
        Path to the serialized gallery directory

    Raises:
        ValueError: If gallery validation fails
        OSError: If directory creation or file writing fails
    """
    output_dir = Path(output_dir)
    gallery_dir = output_dir / gallery.user_id
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Serialize gallery metadata
    metadata = {
        "user_id": gallery.user_id,
        "name": gallery.name,
        "metadata": gallery.metadata,
        "created_at": gallery.created_at.isoformat(),
        "updated_at": gallery.updated_at.isoformat(),
        "status": gallery.status.value,
        "total_clusters": gallery.total_clusters,
        "total_images": gallery.total_images,
    }

    metadata_path = gallery_dir / "gallery_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.debug(f"Saved gallery metadata to {metadata_path}")

    # Serialize variant clusters
    if gallery.variant_clusters:
        clusters_data = []
        for cluster in gallery.variant_clusters:
            cluster_dict = {
                "cluster_id": cluster.cluster_id,
                "user_id": cluster.user_id,
                "cluster_name": cluster.cluster_name,
                "description": cluster.description,
                "created_at": cluster.created_at.isoformat(),
                "image_count": cluster.image_count,
                "primary_image_id": cluster.primary_image_id,
                "metadata": cluster.metadata,
            }
            clusters_data.append(cluster_dict)

        clusters_path = gallery_dir / "variant_clusters.json"
        with open(clusters_path, "w") as f:
            json.dump(clusters_data, f, indent=2, default=str)
        logger.debug(f"Saved {len(clusters_data)} variant clusters to {clusters_path}")

    logger.info(f"Serialized user gallery '{gallery.user_id}' to {gallery_dir}")
    return gallery_dir


def deserialize_user_gallery(gallery_dir: Path) -> UserGallery:
    """
    Deserialize UserGallery from disk.

    Args:
        gallery_dir: Directory containing serialized gallery

    Returns:
        Reconstructed UserGallery instance

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data is invalid or corrupted
    """
    gallery_dir = Path(gallery_dir)

    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

    # Load gallery metadata
    metadata_path = gallery_dir / "gallery_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Gallery metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Reconstruct variant clusters
    variant_clusters = []
    clusters_path = gallery_dir / "variant_clusters.json"
    if clusters_path.exists():
        with open(clusters_path) as f:
            clusters_data = json.load(f)

        for cluster_dict in clusters_data:
            cluster = VariantCluster(
                cluster_id=cluster_dict["cluster_id"],
                user_id=cluster_dict["user_id"],
                cluster_name=cluster_dict.get("cluster_name"),
                description=cluster_dict.get("description"),
                created_at=datetime.fromisoformat(cluster_dict["created_at"]),
                image_count=cluster_dict.get("image_count", 0),
                primary_image_id=cluster_dict.get("primary_image_id"),
                metadata=cluster_dict.get("metadata", {}),
            )
            variant_clusters.append(cluster)
        logger.debug(f"Loaded {len(variant_clusters)} variant clusters")

    # Reconstruct UserGallery
    gallery = UserGallery(
        user_id=metadata["user_id"],
        name=metadata.get("name"),
        metadata=metadata.get("metadata", {}),
        created_at=datetime.fromisoformat(metadata["created_at"]),
        updated_at=datetime.fromisoformat(metadata["updated_at"]),
        status=GalleryStatus(metadata["status"]),
        variant_clusters=variant_clusters,
    )

    logger.info(f"Deserialized user gallery '{gallery.user_id}' from {gallery_dir}")
    return gallery


def serialize_image_assets(images: list[ImageAsset], output_file: Path) -> None:
    """
    Serialize list of ImageAsset objects to JSON.

    Args:
        images: List of ImageAsset instances
        output_file: Path to output JSON file

    Raises:
        OSError: If file writing fails
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    images_data = []
    for img in images:
        img_dict = {
            "image_id": img.image_id,
            "user_id": img.user_id,
            "image_path": img.image_path,
            "modality": img.modality.value,
            "source_camera": img.source_camera,
            "capture_time": img.capture_time.isoformat() if img.capture_time else None,
            "metadata": img.metadata,
            "processing_status": img.processing_status.value,
            "face_detection_confidence": img.face_detection_confidence,
            "created_at": img.created_at.isoformat(),
            "cluster_id": img.cluster_id,
        }
        images_data.append(img_dict)

    with open(output_file, "w") as f:
        json.dump(images_data, f, indent=2, default=str)

    logger.debug(f"Serialized {len(images)} image assets to {output_file}")


def deserialize_image_assets(input_file: Path) -> list[ImageAsset]:
    """
    Deserialize list of ImageAsset objects from JSON.

    Args:
        input_file: Path to input JSON file

    Returns:
        List of reconstructed ImageAsset instances

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data is invalid
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Image assets file not found: {input_file}")

    with open(input_file) as f:
        images_data = json.load(f)

    images = []
    for img_dict in images_data:
        img = ImageAsset(
            image_id=img_dict["image_id"],
            user_id=img_dict["user_id"],
            image_path=img_dict["image_path"],
            modality=Modality(img_dict["modality"]),
            source_camera=img_dict.get("source_camera"),
            capture_time=datetime.fromisoformat(img_dict["capture_time"])
            if img_dict.get("capture_time")
            else None,
            metadata=img_dict.get("metadata", {}),
            processing_status=ProcessingStatus(img_dict["processing_status"]),
            face_detection_confidence=img_dict.get("face_detection_confidence"),
            created_at=datetime.fromisoformat(img_dict["created_at"]),
            cluster_id=img_dict.get("cluster_id"),
        )
        images.append(img)

    logger.debug(f"Deserialized {len(images)} image assets from {input_file}")
    return images


def serialize_embedding_sets(
    embeddings: list[EmbeddingSet], output_dir: Path, save_format: str = "npz"
) -> Path:
    """
    Serialize list of EmbeddingSet objects with multi-modal embeddings.

    Creates:
        - embeddings.npz: Compressed numpy arrays for body/face embeddings
        - embeddings_metadata.json: Metadata for each embedding set

    Args:
        embeddings: List of EmbeddingSet instances
        output_dir: Directory to save embeddings
        save_format: Format for numpy arrays ('npz' or 'pkl')

    Returns:
        Path to output directory

    Raises:
        ValueError: If embeddings list is empty or save_format is invalid
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty")

    if save_format not in ["npz", "pkl"]:
        raise ValueError(f"Invalid save format: {save_format}. Must be 'npz' or 'pkl'")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arrays
    body_embeddings = []
    face_embeddings = []
    embedding_ids = []
    metadata_list = []

    for emb in embeddings:
        body_embeddings.append(emb.body_embedding)
        face_embeddings.append(
            emb.face_embedding
            if emb.face_embedding is not None
            else np.zeros_like(emb.body_embedding)
        )
        embedding_ids.append(emb.embedding_id)

        meta_dict = {
            "embedding_id": emb.embedding_id,
            "user_id": emb.user_id,
            "image_id": emb.image_id,
            "cluster_id": emb.cluster_id,
            "embedding_provider": emb.embedding_provider,
            "embedding_version": emb.embedding_version,
            "quality_score": emb.quality_score,
            "generated_at": emb.generated_at.isoformat(),
            "has_face_embedding": emb.has_face_embedding,
            "metadata": emb.metadata,
        }
        metadata_list.append(meta_dict)

    # Save embeddings
    if save_format == "npz":
        embeddings_path = output_dir / "embeddings.npz"
        np.savez_compressed(
            embeddings_path,
            body_embeddings=np.array(body_embeddings),
            face_embeddings=np.array(face_embeddings),
            embedding_ids=np.array(embedding_ids),
        )
        logger.debug(f"Saved embeddings to {embeddings_path} (compressed npz)")
    else:  # pkl
        embeddings_path = output_dir / "embeddings.pkl"
        with open(embeddings_path, "wb") as f:
            pickle.dump(
                {
                    "body_embeddings": np.array(body_embeddings),
                    "face_embeddings": np.array(face_embeddings),
                    "embedding_ids": embedding_ids,
                },
                f,
            )
        logger.debug(f"Saved embeddings to {embeddings_path} (pickle)")

    # Save metadata
    metadata_path = output_dir / "embeddings_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(
            {"count": len(embeddings), "embeddings": metadata_list},
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Serialized {len(embeddings)} embedding sets to {output_dir}")
    return output_dir


def deserialize_embedding_sets(
    input_dir: Path, load_format: str = "npz"
) -> list[EmbeddingSet]:
    """
    Deserialize list of EmbeddingSet objects from saved files.

    Args:
        input_dir: Directory containing saved embeddings
        load_format: Format to load from ('npz' or 'pkl')

    Returns:
        List of reconstructed EmbeddingSet instances

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If data is invalid or format is unsupported
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {input_dir}")

    # Load metadata
    metadata_path = input_dir / "embeddings_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Embeddings metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        metadata_container = json.load(f)
        metadata_list = metadata_container["embeddings"]

    # Load embeddings
    if load_format == "npz":
        embeddings_path = input_dir / "embeddings.npz"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        data = np.load(embeddings_path)
        body_embeddings = data["body_embeddings"]
        face_embeddings = data["face_embeddings"]
    elif load_format == "pkl":
        embeddings_path = input_dir / "embeddings.pkl"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)
        body_embeddings = data["body_embeddings"]
        face_embeddings = data["face_embeddings"]
    else:
        raise ValueError(f"Invalid load format: {load_format}")

    # Reconstruct EmbeddingSet objects
    embeddings = []
    for i, meta in enumerate(metadata_list):
        face_emb = face_embeddings[i] if meta["has_face_embedding"] else None
        if face_emb is not None and np.allclose(face_emb, 0):
            face_emb = None

        emb = EmbeddingSet(
            embedding_id=meta["embedding_id"],
            user_id=meta["user_id"],
            body_embedding=body_embeddings[i],
            face_embedding=face_emb,
            image_id=meta.get("image_id"),
            cluster_id=meta.get("cluster_id"),
            embedding_provider=meta["embedding_provider"],
            embedding_version=meta["embedding_version"],
            quality_score=meta["quality_score"],
            generated_at=datetime.fromisoformat(meta["generated_at"]),
            metadata=meta.get("metadata", {}),
        )
        embeddings.append(emb)

    logger.info(f"Deserialized {len(embeddings)} embedding sets from {input_dir}")
    return embeddings


# ==================== Validation Utilities ====================


def validate_user_gallery(gallery: UserGallery) -> tuple[bool, list[str]]:
    """
    Validate UserGallery business rules and constraints.

    Business Rules:
    1. User must have at least one variant cluster
    2. Gallery must have a valid status
    3. User ID must be non-empty
    4. Updated timestamp must be >= created timestamp

    Args:
        gallery: UserGallery instance to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Rule 1: User ID must be non-empty
    if not gallery.user_id or not gallery.user_id.strip():
        errors.append("User ID must be non-empty")

    # Rule 2: Must have at least one variant cluster (for operational galleries)
    if gallery.status == GalleryStatus.ACTIVE and len(gallery.variant_clusters) == 0:
        errors.append("Active gallery must have at least one variant cluster")

    # Rule 3: Valid status
    if gallery.status not in [
        GalleryStatus.ACTIVE,
        GalleryStatus.INACTIVE,
        GalleryStatus.PENDING_VERIFICATION,
    ]:
        errors.append(f"Invalid gallery status: {gallery.status}")

    # Rule 4: Timestamps must be consistent
    if gallery.updated_at < gallery.created_at:
        errors.append("Updated timestamp cannot be earlier than created timestamp")

    is_valid = len(errors) == 0
    if is_valid:
        logger.debug(f"Gallery '{gallery.user_id}' validation passed")
    else:
        logger.warning(f"Gallery '{gallery.user_id}' validation failed: {errors}")

    return is_valid, errors


def validate_variant_cluster(cluster: VariantCluster) -> tuple[bool, list[str]]:
    """
    Validate VariantCluster business rules.

    Business Rules:
    1. Cluster must have at least one image
    2. Cluster ID must be unique within user context
    3. Primary image must reference an image in the cluster

    Args:
        cluster: VariantCluster instance to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Rule 1: Must have at least one image
    if cluster.image_count < 1:
        errors.append(f"Cluster '{cluster.cluster_id}' must have at least one image")

    # Rule 2: Cluster ID must be non-empty
    if not cluster.cluster_id or not cluster.cluster_id.strip():
        errors.append("Cluster ID must be non-empty")

    # Rule 3: User ID must be non-empty
    if not cluster.user_id or not cluster.user_id.strip():
        errors.append("User ID must be non-empty")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_image_asset(image: ImageAsset) -> tuple[bool, list[str]]:
    """
    Validate ImageAsset business rules.

    Business Rules:
    1. Image path must be non-empty and valid
    2. Face detection confidence must be in [0.0, 1.0] if provided
    3. Processing status must be valid
    4. Modality must be valid

    Args:
        image: ImageAsset instance to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Rule 1: Image path must be non-empty
    if not image.image_path or not image.image_path.strip():
        errors.append(f"Image '{image.image_id}' must have a non-empty path")

    # Rule 2: Face detection confidence validation
    if image.face_detection_confidence is not None:
        if not 0.0 <= image.face_detection_confidence <= 1.0:
            errors.append(
                f"Face detection confidence for '{image.image_id}' must be in [0.0, 1.0]"
            )

    # Rule 3: Processing status must be valid
    if image.processing_status not in [
        ProcessingStatus.PENDING,
        ProcessingStatus.PROCESSING,
        ProcessingStatus.COMPLETED,
        ProcessingStatus.FAILED,
    ]:
        errors.append(
            f"Invalid processing status for '{image.image_id}': {image.processing_status}"
        )

    # Rule 4: Modality must be valid
    if image.modality not in [Modality.BODY, Modality.FACE, Modality.UNKNOWN]:
        errors.append(f"Invalid modality for '{image.image_id}': {image.modality}")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_embedding_set(embedding: EmbeddingSet) -> tuple[bool, list[str]]:
    """
    Validate EmbeddingSet business rules.

    Business Rules:
    1. Body embedding must be present and have valid dimensions
    2. Quality score must be in [0.0, 1.0]
    3. At least one embedding type (body or face) must be present
    4. Face embedding dimensions must be valid if present

    Args:
        embedding: EmbeddingSet instance to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Rule 1: Body embedding validation
    if embedding.body_embedding is None or len(embedding.body_embedding) == 0:
        errors.append(
            f"Embedding '{embedding.embedding_id}' must have a body embedding"
        )
    elif len(embedding.body_embedding) < 512:
        errors.append(
            f"Body embedding dimension for '{embedding.embedding_id}' is too small: {len(embedding.body_embedding)}"
        )

    # Rule 2: Quality score validation
    if not 0.0 <= embedding.quality_score <= 1.0:
        errors.append(
            f"Quality score for '{embedding.embedding_id}' must be in [0.0, 1.0]"
        )

    # Rule 3: At least one embedding type
    if embedding.body_embedding is None and embedding.face_embedding is None:
        errors.append(
            f"Embedding '{embedding.embedding_id}' must have at least one embedding type"
        )

    # Rule 4: Face embedding validation if present
    if embedding.face_embedding is not None and len(embedding.face_embedding) < 128:
        errors.append(
            f"Face embedding dimension for '{embedding.embedding_id}' is too small: {len(embedding.face_embedding)}"
        )

    is_valid = len(errors) == 0
    return is_valid, errors


def enforce_gallery_business_rules(
    gallery: UserGallery, max_clusters: int = 50, max_images: int = 1000
) -> tuple[bool, list[str]]:
    """
    Enforce comprehensive business rules for user galleries.

    Additional Rules:
    5. Maximum number of clusters per gallery
    6. Maximum number of images per gallery
    7. All clusters must belong to the same user
    8. No duplicate cluster IDs

    Args:
        gallery: UserGallery instance to validate
        max_clusters: Maximum allowed clusters per gallery
        max_images: Maximum allowed images per gallery

    Returns:
        Tuple of (is_valid, error_messages)
    """
    # Run basic validation first
    is_valid, errors = validate_user_gallery(gallery)

    # Rule 5: Maximum clusters
    if len(gallery.variant_clusters) > max_clusters:
        errors.append(
            f"Gallery '{gallery.user_id}' exceeds maximum clusters: {len(gallery.variant_clusters)} > {max_clusters}"
        )

    # Rule 6: Maximum images
    if gallery.total_images > max_images:
        errors.append(
            f"Gallery '{gallery.user_id}' exceeds maximum images: {gallery.total_images} > {max_images}"
        )

    # Rule 7: All clusters belong to same user
    for cluster in gallery.variant_clusters:
        if cluster.user_id != gallery.user_id:
            errors.append(
                f"Cluster '{cluster.cluster_id}' user_id mismatch: expected '{gallery.user_id}', got '{cluster.user_id}'"
            )

    # Rule 8: No duplicate cluster IDs
    cluster_ids = [c.cluster_id for c in gallery.variant_clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        duplicates = [cid for cid in cluster_ids if cluster_ids.count(cid) > 1]
        errors.append(f"Duplicate cluster IDs found: {set(duplicates)}")

    is_valid = len(errors) == 0
    return is_valid, errors


# ==================== Storage Management ====================


def create_gallery_storage_structure(base_dir: Path, user_id: str) -> dict[str, Path]:
    """
    Create standard storage directory structure for a user gallery.

    Structure:
    {base_dir}/{user_id}/
        ├── metadata/          # Gallery and cluster metadata
        ├── embeddings/        # Embedding storage
        ├── images/            # Image metadata and references
        └── indexes/           # FAISS indexes for search

    Args:
        base_dir: Base directory for gallery storage
        user_id: User identifier

    Returns:
        Dictionary mapping component names to their paths

    Raises:
        OSError: If directory creation fails
    """
    base_dir = Path(base_dir)
    user_gallery_dir = base_dir / user_id

    structure = {
        "root": user_gallery_dir,
        "metadata": user_gallery_dir / "metadata",
        "embeddings": user_gallery_dir / "embeddings",
        "images": user_gallery_dir / "images",
        "indexes": user_gallery_dir / "indexes",
    }

    for _name, path in structure.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {path}")

    logger.info(f"Created storage structure for user '{user_id}' at {user_gallery_dir}")
    return structure


def cleanup_gallery_storage(gallery_dir: Path, remove_embeddings: bool = False) -> None:
    """
    Clean up temporary and cached files from gallery storage.

    Args:
        gallery_dir: Root directory of the gallery
        remove_embeddings: If True, also remove cached embeddings

    Raises:
        OSError: If cleanup fails
    """
    import shutil

    gallery_dir = Path(gallery_dir)

    if not gallery_dir.exists():
        logger.warning(f"Gallery directory does not exist: {gallery_dir}")
        return

    # Clean up temporary files
    temp_patterns = ["*.tmp", "*.temp", ".DS_Store", "Thumbs.db"]
    for pattern in temp_patterns:
        for temp_file in gallery_dir.rglob(pattern):
            temp_file.unlink()
            logger.debug(f"Removed temporary file: {temp_file}")

    # Optionally remove embeddings
    if remove_embeddings:
        embeddings_dir = gallery_dir / "embeddings"
        if embeddings_dir.exists():
            shutil.rmtree(embeddings_dir)
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cleared embeddings cache for {gallery_dir}")

    logger.info(f"Cleaned up gallery storage: {gallery_dir}")


def get_gallery_size_info(gallery_dir: Path) -> dict[str, Any]:
    """
    Calculate storage size information for a gallery.

    Args:
        gallery_dir: Root directory of the gallery

    Returns:
        Dictionary with size information:
        - total_size: Total size in bytes
        - embeddings_size: Size of embeddings directory
        - metadata_size: Size of metadata files
        - indexes_size: Size of FAISS indexes

    Raises:
        FileNotFoundError: If gallery directory doesn't exist
    """
    gallery_dir = Path(gallery_dir)

    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

    def get_dir_size(path: Path) -> int:
        """Recursively calculate directory size."""
        total = 0
        if path.is_file():
            return path.stat().st_size
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    info = {
        "total_size": get_dir_size(gallery_dir),
        "embeddings_size": get_dir_size(gallery_dir / "embeddings")
        if (gallery_dir / "embeddings").exists()
        else 0,
        "metadata_size": get_dir_size(gallery_dir / "metadata")
        if (gallery_dir / "metadata").exists()
        else 0,
        "indexes_size": get_dir_size(gallery_dir / "indexes")
        if (gallery_dir / "indexes").exists()
        else 0,
    }

    logger.debug(f"Gallery size info: {info}")
    return info

# ==================== Image Validation and Preprocessing ====================


def validate_image_path(image_path: str | Path) -> tuple[bool, list[str]]:
    """
    Validate that an image path exists and is accessible.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    image_path = Path(image_path)

    # Check if path exists
    if not image_path.exists():
        errors.append(f"Image path does not exist: {image_path}")
        return False, errors

    # Check if it's a file
    if not image_path.is_file():
        errors.append(f"Image path is not a file: {image_path}")
        return False, errors

    # Check file extension
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if image_path.suffix.lower() not in valid_extensions:
        errors.append(
            f"Invalid image extension '{image_path.suffix}'. "
            f"Supported: {valid_extensions}"
        )
        return False, errors

    # Check file size (not empty, not too large)
    file_size = image_path.stat().st_size
    if file_size == 0:
        errors.append(f"Image file is empty: {image_path}")
        return False, errors

    max_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_size:
        errors.append(
            f"Image file too large: {file_size / (1024*1024):.1f}MB "
            f"(max: {max_size / (1024*1024):.1f}MB)"
        )
        return False, errors

    return True, []


def validate_image_readable(image_path: str | Path) -> tuple[bool, list[str]]:
    """
    Validate that an image can be read and decoded.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    try:
        from PIL import Image

        image_path = Path(image_path)

        # Try to open and verify the image
        with Image.open(image_path) as img:
            # Verify image can be loaded
            img.verify()

        # Re-open to get dimensions (verify() closes the file)
        with Image.open(image_path) as img:
            width, height = img.size

            # Check minimum dimensions
            min_dimension = 32
            if width < min_dimension or height < min_dimension:
                errors.append(
                    f"Image dimensions too small: {width}x{height} "
                    f"(minimum: {min_dimension}x{min_dimension})"
                )
                return False, errors

            # Check maximum dimensions
            max_dimension = 8192
            if width > max_dimension or height > max_dimension:
                errors.append(
                    f"Image dimensions too large: {width}x{height} "
                    f"(maximum: {max_dimension}x{max_dimension})"
                )
                return False, errors

        return True, []

    except ImportError:
        errors.append("PIL/Pillow is required for image validation")
        return False, errors
    except Exception as e:
        errors.append(f"Failed to read image: {str(e)}")
        return False, errors


def preprocess_image_asset(
    image_path: str | Path,
    user_id: str,
    modality: Modality = Modality.UNKNOWN,
    source_camera: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[ImageAsset | None, list[str]]:
    """
    Preprocess and create an ImageAsset from an image path.

    Performs validation and creates a properly initialized ImageAsset.

    Args:
        image_path: Path to image file
        user_id: User identifier
        modality: Image modality (BODY, FACE, UNKNOWN)
        source_camera: Optional camera identifier
        metadata: Optional additional metadata

    Returns:
        Tuple of (ImageAsset or None, error_messages)
    """
    errors = []

    # Validate path
    is_valid, path_errors = validate_image_path(image_path)
    if not is_valid:
        errors.extend(path_errors)
        return None, errors

    # Validate readability
    is_valid, read_errors = validate_image_readable(image_path)
    if not is_valid:
        errors.extend(read_errors)
        return None, errors

    # Generate unique image ID
    import uuid

    image_id = f"{user_id}_{uuid.uuid4().hex[:12]}"

    # Create ImageAsset
    try:
        image_asset = ImageAsset(
            image_id=image_id,
            user_id=user_id,
            image_path=str(Path(image_path).absolute()),
            modality=modality,
            source_camera=source_camera,
            metadata=metadata or {},
            processing_status=ProcessingStatus.PENDING,
            created_at=datetime.now(),
        )

        logger.debug(f"Created ImageAsset: {image_id} for user '{user_id}'")
        return image_asset, []

    except Exception as e:
        errors.append(f"Failed to create ImageAsset: {str(e)}")
        return None, errors


def batch_validate_images(
    image_paths: list[str | Path],
) -> tuple[list[str | Path], list[tuple[str | Path, list[str]]]]:
    """
    Validate a batch of image paths.

    Args:
        image_paths: List of image paths to validate

    Returns:
        Tuple of (valid_paths, invalid_paths_with_errors)
    """
    valid_paths = []
    invalid_paths = []

    for image_path in image_paths:
        is_valid, errors = validate_image_path(image_path)
        if is_valid:
            is_valid, errors = validate_image_readable(image_path)

        if is_valid:
            valid_paths.append(image_path)
        else:
            invalid_paths.append((image_path, errors))

    logger.info(
        f"Batch validation: {len(valid_paths)} valid, {len(invalid_paths)} invalid"
    )
    return valid_paths, invalid_paths


def infer_modality_from_path(image_path: str | Path) -> Modality:
    """
    Infer image modality from file path or name.

    Looks for keywords like 'face', 'body', 'full' in the path.

    Args:
        image_path: Path to image file

    Returns:
        Inferred Modality (BODY, FACE, or UNKNOWN)
    """
    path_str = str(image_path).lower()

    # Check for face indicators
    face_keywords = ["face", "facial", "portrait", "headshot"]
    if any(keyword in path_str for keyword in face_keywords):
        return Modality.FACE

    # Check for body indicators
    body_keywords = ["body", "full", "fullbody", "person", "whole"]
    if any(keyword in path_str for keyword in body_keywords):
        return Modality.BODY

    # Default to unknown
    return Modality.UNKNOWN


def group_images_by_modality(
    images: list[ImageAsset],
) -> dict[Modality, list[ImageAsset]]:
    """
    Group images by their modality.

    Args:
        images: List of image assets

    Returns:
        Dictionary mapping modality to list of images
    """
    grouped: dict[Modality, list[ImageAsset]] = {
        Modality.BODY: [],
        Modality.FACE: [],
        Modality.UNKNOWN: [],
    }

    for image in images:
        grouped[image.modality].append(image)

    return grouped


# ==================== Face Detection Fallback and Error Handling ====================


def handle_face_detection_failure(
    image_path: str | Path,
    error: Exception,
    fallback_strategy: str = "skip",
) -> tuple[bool, Optional[str]]:
    """
    Handle face detection failures with configurable fallback strategies.

    Args:
        image_path: Path to image where face detection failed
        error: Exception that occurred during face detection
        fallback_strategy: Strategy to use ('skip', 'retry', 'use_body_only')

    Returns:
        Tuple of (should_continue, fallback_message)
        - should_continue: Whether to continue processing
        - fallback_message: Message describing the fallback action

    Examples:
        >>> try:
        ...     embedding = generate_face_embedding("image.jpg")
        ... except Exception as e:
        ...     should_continue, msg = handle_face_detection_failure(
        ...         "image.jpg", e, fallback_strategy="skip"
        ...     )
        ...     if should_continue:
        ...         print(f"Continuing with fallback: {msg}")
    """
    image_name = Path(image_path).name
    error_msg = str(error)

    logger.warning(
        f"Face detection failed for {image_name}: {error_msg} "
        f"(fallback_strategy={fallback_strategy})"
    )

    if fallback_strategy == "skip":
        # Skip this image and continue with others
        fallback_message = (
            f"Skipped face embedding for {image_name} due to detection failure"
        )
        return True, fallback_message

    elif fallback_strategy == "retry":
        # Indicate that retry should be attempted
        fallback_message = f"Will retry face detection for {image_name}"
        return True, fallback_message

    elif fallback_strategy == "use_body_only":
        # Continue with body embeddings only
        fallback_message = (
            f"Using body embeddings only for {image_name} (face detection failed)"
        )
        return True, fallback_message

    else:
        # Unknown strategy, default to skip
        logger.warning(f"Unknown fallback strategy '{fallback_strategy}', defaulting to 'skip'")
        fallback_message = f"Skipped {image_name} (unknown fallback strategy)"
        return True, fallback_message


def validate_face_embedding_requirements(
    image_path: str | Path,
    min_face_size: tuple[int, int] = (80, 80),
    check_image_quality: bool = True,
) -> tuple[bool, list[str]]:
    """
    Validate that an image meets requirements for face embedding generation.

    Args:
        image_path: Path to image file
        min_face_size: Minimum face size (width, height) in pixels
        check_image_quality: Whether to perform quality checks

    Returns:
        Tuple of (is_valid, error_messages)

    Examples:
        >>> is_valid, errors = validate_face_embedding_requirements("face.jpg")
        >>> if not is_valid:
        ...     print(f"Image validation failed: {errors}")
    """
    errors = []

    # Basic path validation
    is_valid, path_errors = validate_image_path(image_path)
    if not is_valid:
        errors.extend(path_errors)
        return False, errors

    # Check image readability
    is_valid, read_errors = validate_image_readable(image_path)
    if not is_valid:
        errors.extend(read_errors)
        return False, errors

    # Additional quality checks
    if check_image_quality:
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size

                # Check if image is large enough for face detection
                if width < min_face_size[0] or height < min_face_size[1]:
                    errors.append(
                        f"Image too small for reliable face detection: {width}x{height} "
                        f"(minimum: {min_face_size[0]}x{min_face_size[1]})"
                    )

                # Check aspect ratio (faces are typically not extremely wide or tall)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                    logger.warning(
                        f"Unusual aspect ratio {aspect_ratio:.2f} for {Path(image_path).name}, "
                        "face detection may be unreliable"
                    )

        except Exception as e:
            errors.append(f"Failed to check image quality: {str(e)}")
            return False, errors

    is_valid = len(errors) == 0
    return is_valid, errors


def retry_with_fallback(
    func: callable,
    *args,
    max_retries: int = 3,
    fallback_value: Any = None,
    exceptions: tuple = (Exception,),
    **kwargs,
) -> tuple[Any, bool, Optional[str]]:
    """
    Retry a function with exponential backoff and fallback on failure.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        fallback_value: Value to return if all retries fail
        exceptions: Tuple of exceptions to catch and retry
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (result, success, error_message)
        - result: Function result or fallback_value
        - success: Whether function succeeded
        - error_message: Error message if failed, None if succeeded

    Examples:
        >>> def unstable_function(x):
        ...     if random.random() < 0.5:
        ...         raise ValueError("Random failure")
        ...     return x * 2
        >>>
        >>> result, success, error = retry_with_fallback(
        ...     unstable_function, 5, max_retries=3, fallback_value=0
        ... )
        >>> if success:
        ...     print(f"Result: {result}")
        ... else:
        ...     print(f"Failed: {error}, using fallback: {result}")
    """
    import time

    last_error = None

    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result, True, None

        except exceptions as e:
            last_error = str(e)
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {last_error}"
            )

            if attempt < max_retries - 1:
                # Exponential backoff: 0.5s, 1s, 2s, ...
                sleep_time = 0.5 * (2 ** attempt)
                logger.debug(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

    # All retries failed
    error_message = f"All {max_retries} attempts failed. Last error: {last_error}"
    logger.error(error_message)
    return fallback_value, False, error_message


def batch_process_with_error_handling(
    items: list[Any],
    process_func: callable,
    error_strategy: str = "skip",
    max_failures: Optional[int] = None,
) -> tuple[list[Any], list[tuple[Any, str]]]:
    """
    Process a batch of items with error handling and failure tracking.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        error_strategy: How to handle errors ('skip', 'stop', 'collect')
        max_failures: Maximum allowed failures before stopping (None for unlimited)

    Returns:
        Tuple of (successful_results, failed_items_with_errors)

    Examples:
        >>> def process_image(img_path):
        ...     # Process image and return result
        ...     return generate_embedding(img_path)
        >>>
        >>> results, failures = batch_process_with_error_handling(
        ...     image_paths, process_image, error_strategy="skip", max_failures=5
        ... )
        >>> print(f"Processed {len(results)} successfully, {len(failures)} failed")
    """
    successful_results = []
    failed_items = []
    failure_count = 0

    for item in items:
        try:
            result = process_func(item)
            successful_results.append(result)

        except Exception as e:
            error_msg = f"Failed to process {item}: {str(e)}"
            logger.warning(error_msg)

            failed_items.append((item, str(e)))
            failure_count += 1

            if error_strategy == "stop":
                logger.error(f"Stopping batch processing due to error: {error_msg}")
                break

            if max_failures is not None and failure_count >= max_failures:
                logger.error(
                    f"Stopping batch processing: reached max failures ({max_failures})"
                )
                break

    logger.info(
        f"Batch processing complete: {len(successful_results)} successful, "
        f"{len(failed_items)} failed"
    )

    return successful_results, failed_items


def log_embedding_generation_error(
    image_id: str,
    user_id: str,
    error: Exception,
    modality: str = "face",
    context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Log and structure embedding generation errors for audit trail.

    Args:
        image_id: Image identifier
        user_id: User identifier
        error: Exception that occurred
        modality: Embedding modality ('face' or 'body')
        context: Optional additional context

    Returns:
        Structured error record dictionary

    Examples:
        >>> try:
        ...     embedding = generate_face_embedding(image_path)
        ... except Exception as e:
        ...     error_record = log_embedding_generation_error(
        ...         image_id="img_123",
        ...         user_id="user_001",
        ...         error=e,
        ...         modality="face",
        ...         context={"model": "Facenet", "detector": "opencv"}
        ...     )
        ...     # Store error_record in database or log file
    """
    error_record = {
        "timestamp": datetime.now().isoformat(),
        "image_id": image_id,
        "user_id": user_id,
        "modality": modality,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {},
    }

    logger.error(
        f"Embedding generation error for {modality} modality: "
        f"user={user_id}, image={image_id}, error={error_record['error_type']}: "
        f"{error_record['error_message']}"
    )

    return error_record


# ==================== Evidence Tracking and Explanation Generation ====================


def generate_retrieval_explanation(
    result,
    include_technical_details: bool = False,
) -> dict[str, Any]:
    """
    Generate human-readable explanation for a retrieval result.

    Creates a structured explanation of why a particular user was matched,
    including modality contributions, confidence levels, and evidence images.

    Args:
        result: RetrievalResult instance
        include_technical_details: Whether to include technical scoring details

    Returns:
        Dictionary with explanation components:
        - summary: High-level explanation text
        - modality_breakdown: Contribution of each modality
        - confidence_explanation: Why this confidence level was assigned
        - evidence_summary: Summary of evidence images
        - technical_details: Optional technical scoring information

    Examples:
        >>> explanation = generate_retrieval_explanation(result)
        >>> print(explanation['summary'])
        "High confidence match based on strong face similarity (0.92) and body similarity (0.85)"
    """
    from .models import ConfidenceLevel

    # Determine primary modality contributor
    primary_modality = None
    if result.face_score is not None and result.body_score is not None:
        if result.face_score * result.face_weight > result.body_score * result.body_weight:
            primary_modality = "face"
        else:
            primary_modality = "body"
    elif result.face_score is not None:
        primary_modality = "face"
    elif result.body_score is not None:
        primary_modality = "body"

    # Generate summary text
    confidence_text = result.confidence_level.value.lower()
    if result.confidence_level == ConfidenceLevel.HIGH:
        confidence_desc = "strong"
    elif result.confidence_level == ConfidenceLevel.MEDIUM:
        confidence_desc = "moderate"
    else:
        confidence_desc = "weak"

    summary_parts = [f"{confidence_text.capitalize()} confidence match"]

    if primary_modality == "face" and result.face_score is not None:
        summary_parts.append(f"primarily based on face similarity ({result.face_score:.2f})")
    elif primary_modality == "body" and result.body_score is not None:
        summary_parts.append(f"primarily based on body similarity ({result.body_score:.2f})")

    if result.face_score is not None and result.body_score is not None:
        summary_parts.append(
            f"with supporting evidence from both face ({result.face_score:.2f}) "
            f"and body ({result.body_score:.2f}) modalities"
        )

    summary = " ".join(summary_parts)

    # Modality breakdown
    modality_breakdown = {
        "face": {
            "score": result.face_score,
            "weight": result.face_weight,
            "contribution": result.face_score * result.face_weight if result.face_score else 0.0,
            "used": result.has_face_contribution,
        },
        "body": {
            "score": result.body_score,
            "weight": result.body_weight,
            "contribution": result.body_score * result.body_weight if result.body_score else 0.0,
            "used": result.has_body_contribution,
        },
    }

    # Confidence explanation
    if result.confidence_level == ConfidenceLevel.HIGH:
        confidence_explanation = (
            f"High confidence assigned due to overall score of {result.overall_score:.2f} "
            "(threshold: 0.70+)"
        )
    elif result.confidence_level == ConfidenceLevel.MEDIUM:
        confidence_explanation = (
            f"Medium confidence assigned due to overall score of {result.overall_score:.2f} "
            "(threshold: 0.40-0.70)"
        )
    else:
        confidence_explanation = (
            f"Low confidence assigned due to overall score of {result.overall_score:.2f} "
            "(threshold: below 0.40)"
        )

    # Evidence summary
    evidence_summary = {
        "total_evidence_images": len(result.evidence_images),
        "evidence_image_ids": result.evidence_images,
    }

    explanation = {
        "summary": summary,
        "modality_breakdown": modality_breakdown,
        "confidence_explanation": confidence_explanation,
        "evidence_summary": evidence_summary,
        "overall_score": result.overall_score,
        "confidence_level": result.confidence_level.value,
    }

    # Add technical details if requested
    if include_technical_details:
        explanation["technical_details"] = {
            "result_id": result.result_id,
            "probe_id": result.probe_id,
            "user_id": result.user_id,
            "retrieved_at": result.retrieved_at.isoformat(),
            "fusion_formula": (
                f"overall_score = (face_score * face_weight) + (body_score * body_weight) = "
                f"({result.face_score if result.face_score else 0.0:.3f} * {result.face_weight:.2f}) + "
                f"({result.body_score if result.body_score else 0.0:.3f} * {result.body_weight:.2f}) = "
                f"{result.overall_score:.3f}"
            ),
            "metadata": result.metadata,
        }

    return explanation


def track_evidence_images(
    user_id: str,
    probe_embedding: np.ndarray,
    gallery_embeddings: list[tuple[str, np.ndarray]],
    top_k: int = 3,
    similarity_metric: str = "cosine",
) -> list[tuple[str, float]]:
    """
    Track which gallery images contributed most to a match.

    Identifies the top-k most similar gallery images for a given probe,
    providing evidence for the retrieval result.

    Args:
        user_id: User identifier
        probe_embedding: Probe embedding vector
        gallery_embeddings: List of (image_id, embedding) tuples for the user
        top_k: Number of top evidence images to return
        similarity_metric: Similarity metric to use ('cosine', 'euclidean')

    Returns:
        List of (image_id, similarity_score) tuples, sorted by similarity

    Examples:
        >>> evidence = track_evidence_images(
        ...     "user_001",
        ...     probe_emb,
        ...     [("img1", emb1), ("img2", emb2), ("img3", emb3)],
        ...     top_k=2
        ... )
        >>> for img_id, score in evidence:
        ...     print(f"Evidence image {img_id}: similarity={score:.3f}")
    """
    if len(gallery_embeddings) == 0:
        logger.warning(f"No gallery embeddings provided for user '{user_id}'")
        return []

    # Compute similarities
    similarities = []
    for image_id, gallery_emb in gallery_embeddings:
        if similarity_metric == "cosine":
            # Cosine similarity: dot product of normalized vectors
            probe_norm = probe_embedding / (np.linalg.norm(probe_embedding) + 1e-8)
            gallery_norm = gallery_emb / (np.linalg.norm(gallery_emb) + 1e-8)
            similarity = float(np.dot(probe_norm, gallery_norm))
        elif similarity_metric == "euclidean":
            # Euclidean distance converted to similarity
            distance = np.linalg.norm(probe_embedding - gallery_emb)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

        similarities.append((image_id, similarity))

    # Sort by similarity (descending) and take top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_evidence = similarities[:top_k]

    logger.debug(
        f"Tracked {len(top_evidence)} evidence images for user '{user_id}' "
        f"(top similarity: {top_evidence[0][1]:.3f})"
    )

    return top_evidence


def aggregate_evidence_across_modalities(
    face_evidence: list[tuple[str, float]],
    body_evidence: list[tuple[str, float]],
    fusion_weights: dict[str, float],
) -> list[tuple[str, float, dict[str, Any]]]:
    """
    Aggregate evidence from multiple modalities with fusion weighting.

    Combines evidence from face and body modalities, applying fusion weights
    to determine overall evidence strength.

    Args:
        face_evidence: List of (image_id, similarity) from face modality
        body_evidence: List of (image_id, similarity) from body modality
        fusion_weights: Fusion weights dict with 'face' and 'body' keys

    Returns:
        List of (image_id, aggregated_score, modality_details) tuples
        sorted by aggregated score

    Examples:
        >>> evidence = aggregate_evidence_across_modalities(
        ...     face_evidence=[("img1", 0.9), ("img2", 0.8)],
        ...     body_evidence=[("img1", 0.85), ("img3", 0.75)],
        ...     fusion_weights={"face": 0.6, "body": 0.4}
        ... )
        >>> for img_id, score, details in evidence:
        ...     print(f"{img_id}: {score:.3f} (face={details['face_score']}, body={details['body_score']})")
    """
    # Create mapping of image_id to scores
    evidence_map: dict[str, dict[str, Any]] = {}

    # Add face evidence
    for image_id, face_sim in face_evidence:
        if image_id not in evidence_map:
            evidence_map[image_id] = {
                "face_score": None,
                "body_score": None,
                "modalities": [],
            }
        evidence_map[image_id]["face_score"] = face_sim
        evidence_map[image_id]["modalities"].append("face")

    # Add body evidence
    for image_id, body_sim in body_evidence:
        if image_id not in evidence_map:
            evidence_map[image_id] = {
                "face_score": None,
                "body_score": None,
                "modalities": [],
            }
        evidence_map[image_id]["body_score"] = body_sim
        evidence_map[image_id]["modalities"].append("body")

    # Compute aggregated scores
    aggregated_evidence = []
    for image_id, scores in evidence_map.items():
        face_score = scores["face_score"] if scores["face_score"] is not None else 0.0
        body_score = scores["body_score"] if scores["body_score"] is not None else 0.0

        # Apply fusion weights
        aggregated_score = (
            face_score * fusion_weights.get("face", 0.5)
            + body_score * fusion_weights.get("body", 0.5)
        )

        modality_details = {
            "face_score": scores["face_score"],
            "body_score": scores["body_score"],
            "modalities_present": scores["modalities"],
            "face_weight": fusion_weights.get("face", 0.5),
            "body_weight": fusion_weights.get("body", 0.5),
        }

        aggregated_evidence.append((image_id, aggregated_score, modality_details))

    # Sort by aggregated score (descending)
    aggregated_evidence.sort(key=lambda x: x[1], reverse=True)

    logger.debug(
        f"Aggregated evidence from {len(face_evidence)} face and {len(body_evidence)} body images "
        f"into {len(aggregated_evidence)} total evidence items"
    )

    return aggregated_evidence


def generate_confidence_explanation(
    overall_score: float,
    face_score: Optional[float],
    body_score: Optional[float],
    face_weight: float,
    body_weight: float,
) -> dict[str, Any]:
    """
    Generate detailed explanation for confidence level assignment.

    Args:
        overall_score: Final fusion score
        face_score: Face modality score (or None)
        body_score: Body modality score (or None)
        face_weight: Weight applied to face score
        body_weight: Weight applied to body score

    Returns:
        Dictionary with confidence explanation details

    Examples:
        >>> explanation = generate_confidence_explanation(
        ...     overall_score=0.85,
        ...     face_score=0.92,
        ...     body_score=0.75,
        ...     face_weight=0.6,
        ...     body_weight=0.4
        ... )
        >>> print(explanation['level'])
        "HIGH"
        >>> print(explanation['reasoning'])
    """
    from .models import ConfidenceLevel

    # Determine confidence level
    if overall_score >= 0.7:
        level = ConfidenceLevel.HIGH
        level_reasoning = "Overall score exceeds high confidence threshold (0.70)"
    elif overall_score >= 0.4:
        level = ConfidenceLevel.MEDIUM
        level_reasoning = "Overall score is in medium confidence range (0.40-0.70)"
    else:
        level = ConfidenceLevel.LOW
        level_reasoning = "Overall score is below medium confidence threshold (0.40)"

    # Analyze modality contributions
    modality_analysis = []
    if face_score is not None:
        face_contribution = face_score * face_weight
        if face_score >= 0.8:
            modality_analysis.append(f"Strong face match ({face_score:.2f})")
        elif face_score >= 0.6:
            modality_analysis.append(f"Moderate face match ({face_score:.2f})")
        else:
            modality_analysis.append(f"Weak face match ({face_score:.2f})")

    if body_score is not None:
        body_contribution = body_score * body_weight
        if body_score >= 0.8:
            modality_analysis.append(f"Strong body match ({body_score:.2f})")
        elif body_score >= 0.6:
            modality_analysis.append(f"Moderate body match ({body_score:.2f})")
        else:
            modality_analysis.append(f"Weak body match ({body_score:.2f})")

    # Generate reasoning text
    reasoning_parts = [level_reasoning]
    if modality_analysis:
        reasoning_parts.append(". ".join(modality_analysis))

    explanation = {
        "level": level.value,
        "overall_score": overall_score,
        "reasoning": ". ".join(reasoning_parts),
        "modality_scores": {
            "face": {"score": face_score, "weight": face_weight},
            "body": {"score": body_score, "weight": body_weight},
        },
        "thresholds": {"high": 0.7, "medium": 0.4, "low": 0.0},
    }

    return explanation


def format_retrieval_results_for_display(
    results: list,
    include_explanations: bool = True,
    max_evidence_images: int = 3,
) -> list[dict[str, Any]]:
    """
    Format retrieval results for user-friendly display.

    Args:
        results: List of RetrievalResult instances
        include_explanations: Whether to include detailed explanations
        max_evidence_images: Maximum evidence images to include per result

    Returns:
        List of formatted result dictionaries ready for display

    Examples:
        >>> formatted = format_retrieval_results_for_display(results)
        >>> for result in formatted:
        ...     print(f"Rank {result['rank']}: {result['user_id']}")
        ...     print(f"  Score: {result['score']}")
        ...     print(f"  Explanation: {result['explanation']}")
    """
    formatted_results = []

    for rank, result in enumerate(results, start=1):
        formatted = {
            "rank": rank,
            "user_id": result.user_id,
            "score": round(result.overall_score, 3),
            "confidence": result.confidence_level.value,
            "modality_scores": {
                "face": round(result.face_score, 3) if result.face_score else None,
                "body": round(result.body_score, 3) if result.body_score else None,
            },
            "fusion_weights": {
                "face": round(result.face_weight, 2),
                "body": round(result.body_weight, 2),
            },
            "evidence_images": result.evidence_images[:max_evidence_images],
            "retrieved_at": result.retrieved_at.isoformat(),
        }

        # Add explanation if requested
        if include_explanations:
            explanation = generate_retrieval_explanation(result, include_technical_details=False)
            formatted["explanation"] = explanation["summary"]
            formatted["confidence_explanation"] = explanation["confidence_explanation"]

        formatted_results.append(formatted)

    return formatted_results
