"""
Utility functions for DeepPerson library.

Provides device selection, normalization helpers, and serialization utilities
for embeddings and gallery management.
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def select_device(prefer_cuda: bool = True, device_id: int = 0) -> torch.device:
    """
    Select execution device (CUDA GPU or CPU) with fallback logic.

    Args:
        prefer_cuda: Whether to prefer CUDA if available
        device_id: CUDA device index (default: 0)

    Returns:
        torch.device: Selected device (cuda:N or cpu)

    Examples:
        >>> device = select_device(prefer_cuda=True)
        >>> print(device)  # cuda:0 if GPU available, else cpu
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Selected device: {device} (CUDA available)")
    else:
        device = torch.device("cpu")
        if prefer_cuda and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
        else:
            logger.info("Selected device: cpu")

    return device


def get_device_name(device: torch.device) -> Literal["cuda", "cpu"]:
    """
    Convert torch.device to canonical device name for entity metadata.

    Args:
        device: torch.device instance

    Returns:
        Device name as 'cuda' or 'cpu'
    """
    return "cuda" if device.type == "cuda" else "cpu"


def normalize_embedding(
    embedding: np.ndarray,
    method: Literal["base", "resnet", "circle"] = "resnet"
) -> np.ndarray:
    """
    Normalize embedding vector according to specified strategy.

    Args:
        embedding: Raw embedding vector (shape: (feature_dim,))
        method: Normalization method
            - 'base': L2 normalization only
            - 'resnet': Standard ResNet normalization (L2)
            - 'circle': Circle loss normalization (L2 + scaling)

    Returns:
        Normalized embedding vector (same shape as input)

    Raises:
        ValueError: If embedding has zero norm
    """
    if embedding.ndim != 1:
        raise ValueError(f"Expected 1D embedding, got shape {embedding.shape}")

    # Compute L2 norm
    norm = np.linalg.norm(embedding)

    if norm < 1e-12:
        raise ValueError("Cannot normalize zero-norm embedding")

    # Base L2 normalization
    normalized = embedding / norm

    if method == "circle":
        # Circle loss typically applies additional scaling
        # For inference, L2 normalization is sufficient
        pass
    elif method == "resnet":
        # Standard ResNet uses L2 normalization
        pass
    elif method == "base":
        # Base normalization is just L2
        pass
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32)


def serialize_embedding(
    embedding_vector: np.ndarray,
    output_path: Path,
    metadata: Optional[dict] = None
) -> None:
    """
    Serialize embedding vector and metadata to disk.

    Saves embedding as .npy file and optional metadata as .json sidecar.

    Args:
        embedding_vector: Embedding to serialize (shape: (feature_dim,))
        output_path: Path for .npy file (without extension)
        metadata: Optional metadata dict to save alongside embedding

    Examples:
        >>> vector = np.random.randn(2048).astype(np.float32)
        >>> serialize_embedding(vector, Path("./embeddings/person_001"))
        # Creates: person_001.npy and person_001.json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embedding vector
    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, embedding_vector)
    logger.debug(f"Saved embedding to {npy_path}")

    # Save metadata if provided
    if metadata is not None:
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.debug(f"Saved metadata to {json_path}")


def deserialize_embedding(input_path: Path) -> tuple[np.ndarray, Optional[dict]]:
    """
    Load embedding vector and metadata from disk.

    Args:
        input_path: Path to .npy file (with or without extension)

    Returns:
        Tuple of (embedding_vector, metadata_dict)

    Raises:
        FileNotFoundError: If .npy file does not exist
    """
    input_path = Path(input_path)

    # Load embedding vector
    npy_path = input_path.with_suffix(".npy")
    if not npy_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {npy_path}")

    embedding_vector = np.load(npy_path)
    logger.debug(f"Loaded embedding from {npy_path}, shape: {embedding_vector.shape}")

    # Load metadata if available
    json_path = input_path.with_suffix(".json")
    metadata = None
    if json_path.exists():
        with open(json_path, "r") as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata from {json_path}")

    return embedding_vector, metadata


def serialize_gallery(
    embeddings: np.ndarray,
    subject_ids: list[str],
    output_dir: Path,
    metadata_list: Optional[list[dict]] = None,
    gallery_name: str = "gallery"
) -> Path:
    """
    Serialize a gallery of embeddings with manifest.

    Creates:
        - {gallery_name}_embeddings.npy: Stacked embedding matrix
        - {gallery_name}_manifest.json: Subject IDs and metadata

    Args:
        embeddings: Stacked embeddings (shape: (n_subjects, feature_dim))
        subject_ids: List of subject identifiers
        output_dir: Directory to save gallery files
        metadata_list: Optional list of metadata dicts per subject
        gallery_name: Base name for gallery files

    Returns:
        Path to the gallery directory

    Raises:
        ValueError: If dimensions mismatch
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if embeddings.shape[0] != len(subject_ids):
        raise ValueError(
            f"Embeddings count ({embeddings.shape[0]}) must match "
            f"subject_ids count ({len(subject_ids)})"
        )

    # Save embedding matrix
    embeddings_path = output_dir / f"{gallery_name}_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved gallery embeddings to {embeddings_path}, shape: {embeddings.shape}")

    # Create manifest
    manifest = {
        "gallery_name": gallery_name,
        "n_subjects": len(subject_ids),
        "feature_dim": embeddings.shape[1],
        "subject_ids": subject_ids,
        "metadata": metadata_list or [{} for _ in subject_ids]
    }

    manifest_path = output_dir / f"{gallery_name}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Saved gallery manifest to {manifest_path}")

    return output_dir


def deserialize_gallery(
    gallery_dir: Path,
    gallery_name: str = "gallery"
) -> tuple[np.ndarray, list[str], list[dict]]:
    """
    Load a serialized gallery from disk.

    Args:
        gallery_dir: Directory containing gallery files
        gallery_name: Base name for gallery files

    Returns:
        Tuple of (embeddings, subject_ids, metadata_list)

    Raises:
        FileNotFoundError: If gallery files do not exist
        ValueError: If manifest and embeddings are inconsistent
    """
    gallery_dir = Path(gallery_dir)

    # Load embeddings
    embeddings_path = gallery_dir / f"{gallery_name}_embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Gallery embeddings not found: {embeddings_path}")

    embeddings = np.load(embeddings_path)
    logger.info(f"Loaded gallery embeddings from {embeddings_path}, shape: {embeddings.shape}")

    # Load manifest
    manifest_path = gallery_dir / f"{gallery_name}_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Gallery manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    subject_ids = manifest["subject_ids"]
    metadata_list = manifest.get("metadata", [{} for _ in subject_ids])

    # Validate consistency
    if embeddings.shape[0] != len(subject_ids):
        raise ValueError(
            f"Embeddings count ({embeddings.shape[0]}) does not match "
            f"manifest subject_ids count ({len(subject_ids)})"
        )

    logger.info(f"Loaded gallery manifest from {manifest_path}, {len(subject_ids)} subjects")

    return embeddings, subject_ids, metadata_list


def validate_embedding_dimension(
    embedding: np.ndarray,
    expected_dim: int
) -> None:
    """
    Validate that embedding has expected dimensionality.

    Args:
        embedding: Embedding vector to validate
        expected_dim: Expected feature dimension

    Raises:
        ValueError: If dimension mismatch
    """
    if embedding.shape[-1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, "
            f"got {embedding.shape[-1]}"
        )


def validate_gallery_compatibility(
    gallery_path: Path,
    model_profile_id: str,
    feature_dimension: int
) -> bool:
    """
    Validate that a gallery is compatible with a model profile.

    Args:
        gallery_path: Path to gallery directory
        model_profile_id: Expected model profile identifier
        feature_dimension: Expected feature dimension

    Returns:
        True if compatible, False otherwise
    """
    manifest_path = gallery_path / "gallery_manifest.json"

    if not manifest_path.exists():
        logger.warning(f"Gallery manifest not found: {manifest_path}")
        return False

    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Check model profile
        if manifest.get("model_profile_id") != model_profile_id:
            logger.warning(
                f"Model profile mismatch: expected {model_profile_id}, "
                f"got {manifest.get('model_profile_id')}"
            )
            return False

        # Check feature dimension
        if manifest.get("feature_dimension") != feature_dimension:
            logger.warning(
                f"Feature dimension mismatch: expected {feature_dimension}, "
                f"got {manifest.get('feature_dimension')}"
            )
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating gallery compatibility: {e}")
        return False
