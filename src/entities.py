"""
Data entities for DeepPerson minimal embedding library.

Defines dataclasses for embeddings, model profiles, gallery entries, and similarity results.
Based on data-model.md specifications.

This module now supports both single-modal (body-only) and multi-modal (body+face) embeddings,
with backward compatibility for existing code.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import numpy as np


class Modality(str, Enum):
    """
    Image modality types for person re-identification.

    Attributes:
        BODY: Full body or person detection
        FACE: Face-only detection
        BODY_FACE: Combined body and face embeddings
        UNKNOWN: Modality not determined
    """
    BODY = "BODY"
    FACE = "FACE"
    BODY_FACE = "BODY_FACE"
    UNKNOWN = "UNKNOWN"


@dataclass
class PersonEmbedding:
    """
    Enhanced embedding representation for a detected person.

    Supports both single-modal (body-only) and multi-modal (body+face) embeddings
    with backward compatibility for existing code.

    Core Attributes (required for basic usage):
        embedding_vector: Body feature vector produced by backbone (shape: feature_dim,)
        subject_confidence: Detector confidence for the associated bounding box
        bbox: Bounding box coordinates (x1, y1, x2, y2) in original image coordinates
        normalization: Normalization strategy applied to the embedding
        model_profile_id: Reference to ModelProfile.identifier
        hardware: Device used for embedding generation

    Optional Attributes:
        timestamp: Capture or processing timestamp
        source_image_id: Identifier linking to original image asset

    Multi-modal Attributes (for body+face fusion):
        modality: Type of embedding (BODY, FACE, BODY_FACE)
        face_embedding: Face feature vector (optional, for multi-modal)
        face_confidence: Face detection confidence (optional, for multi-modal)
        face_bbox: Face bounding box (optional, for multi-modal)

    User Gallery Attributes (for user-centric galleries):
        user_id: User identifier for gallery management
        embedding_id: Unique identifier for this embedding
        cluster_id: Variant cluster identifier (e.g., different outfits)
        quality_score: Embedding quality metric (0.0-1.0)
        embedding_provider: Provider identifier (e.g., "resnet50_circle_dg")
        embedding_version: Provider version for compatibility tracking

    Additional:
        metadata: Flexible dictionary for custom attributes

    Examples:
        >>> # Basic body-only embedding (backward compatible)
        >>> emb = PersonEmbedding(
        ...     embedding_vector=np.random.rand(2048).astype(np.float32),
        ...     subject_confidence=0.95,
        ...     bbox=(100, 100, 200, 400),
        ...     normalization="resnet",
        ...     model_profile_id="resnet50_circle_dg",
        ...     hardware="cuda"
        ... )
        >>> emb.modality
        Modality.BODY

        >>> # Multi-modal body+face embedding
        >>> emb_multi = PersonEmbedding(
        ...     embedding_vector=body_emb,
        ...     face_embedding=face_emb,
        ...     subject_confidence=0.95,
        ...     face_confidence=0.98,
        ...     bbox=(100, 100, 200, 400),
        ...     face_bbox=(120, 110, 180, 170),
        ...     modality=Modality.BODY_FACE,
        ...     normalization="resnet",
        ...     model_profile_id="resnet50_circle_dg",
        ...     hardware="cuda"
        ... )
    """
    # Core fields (required)
    embedding_vector: np.ndarray  # Body embedding, shape: (feature_dim,), dtype: float32
    subject_confidence: float
    bbox: tuple[int, int, int, int]
    normalization: Literal["base", "resnet", "circle"]
    model_profile_id: str
    hardware: Literal["cuda", "cpu"]

    # Optional core fields
    timestamp: Optional[datetime] = None
    source_image_id: Optional[str] = None

    # Multi-modal support (NEW)
    modality: Modality = Modality.BODY
    face_embedding: Optional[np.ndarray] = None
    face_confidence: Optional[float] = None
    face_bbox: Optional[tuple[int, int, int, int]] = None

    # User gallery support (NEW)
    user_id: Optional[str] = None
    embedding_id: Optional[str] = None
    cluster_id: Optional[str] = None
    quality_score: Optional[float] = None
    embedding_provider: Optional[str] = None
    embedding_version: Optional[str] = None

    # Additional metadata
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate embedding vector properties and multi-modal constraints."""
        # Validate body embedding
        if not isinstance(self.embedding_vector, np.ndarray):
            raise TypeError(
                f"embedding_vector must be numpy.ndarray, got {type(self.embedding_vector)}"
            )
        if self.embedding_vector.ndim != 1:
            raise ValueError(
                f"embedding_vector must be 1-dimensional, got shape {self.embedding_vector.shape}"
            )
        if self.embedding_vector.dtype != np.float32:
            self.embedding_vector = self.embedding_vector.astype(np.float32)

        # Validate face embedding if present
        if self.face_embedding is not None:
            if not isinstance(self.face_embedding, np.ndarray):
                raise TypeError(
                    f"face_embedding must be numpy.ndarray, got {type(self.face_embedding)}"
                )
            if self.face_embedding.ndim != 1:
                raise ValueError(
                    f"face_embedding must be 1-dimensional, got shape {self.face_embedding.shape}"
                )
            if self.face_embedding.dtype != np.float32:
                self.face_embedding = self.face_embedding.astype(np.float32)

            # Auto-detect modality if face embedding is provided
            if self.modality == Modality.BODY:
                self.modality = Modality.BODY_FACE

        # Validate quality score
        if self.quality_score is not None:
            if not 0.0 <= self.quality_score <= 1.0:
                raise ValueError(
                    f"quality_score must be between 0.0 and 1.0, got {self.quality_score}"
                )

        # Validate confidence values
        if not 0.0 <= self.subject_confidence <= 1.0:
            raise ValueError(
                f"subject_confidence must be between 0.0 and 1.0, got {self.subject_confidence}"
            )
        if self.face_confidence is not None:
            if not 0.0 <= self.face_confidence <= 1.0:
                raise ValueError(
                    f"face_confidence must be between 0.0 and 1.0, got {self.face_confidence}"
                )

    @property
    def has_face_embedding(self) -> bool:
        """Check if this embedding includes face features."""
        return self.face_embedding is not None

    @property
    def is_multi_modal(self) -> bool:
        """Check if this is a multi-modal (body+face) embedding."""
        return self.modality == Modality.BODY_FACE and self.has_face_embedding

    @property
    def body_dimension(self) -> int:
        """Get the dimensionality of the body embedding."""
        return len(self.embedding_vector)

    @property
    def face_dimension(self) -> Optional[int]:
        """Get the dimensionality of the face embedding (if present)."""
        return len(self.face_embedding) if self.face_embedding is not None else None

    def normalize_body_embedding(self) -> None:
        """Normalize body embedding to unit vector (L2 normalization)."""
        norm = np.linalg.norm(self.embedding_vector)
        if norm > 0:
            self.embedding_vector = self.embedding_vector / norm

    def normalize_face_embedding(self) -> None:
        """Normalize face embedding to unit vector (L2 normalization)."""
        if self.face_embedding is not None:
            norm = np.linalg.norm(self.face_embedding)
            if norm > 0:
                self.face_embedding = self.face_embedding / norm


@dataclass
class ModelProfile:
    """
    Configuration profile for an embedding backbone model.

    Attributes:
        identifier: Unique model identifier (e.g., 'resnet50_circle_dg')
        backbone_path: Path to model weights file
        feature_dim: Dimensionality of embedding vectors produced by this model
        requires_cuda: Whether GPU is recommended for optimal performance
        preprocess_config: Preprocessing configuration (mean, std, resize, crop settings)
    """
    identifier: str
    backbone_path: Path
    feature_dim: int
    requires_cuda: bool = False
    preprocess_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Ensure backbone_path is a Path object."""
        if not isinstance(self.backbone_path, Path):
            self.backbone_path = Path(self.backbone_path)


@dataclass
class GalleryEntry:
    """
    Entry in a person re-identification gallery.

    Attributes:
        embedding: PersonEmbedding instance or serialized reference
        subject_id: User-defined identifier for this subject
        metadata: Additional attributes (role, camera, notes, etc.)
        created_at: Timestamp when entry was added to gallery
    """
    embedding: PersonEmbedding
    subject_id: str
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SimilarityMatch:
    """
    Single match result from gallery search.

    Attributes:
        gallery_subject_id: Subject ID from the matching GalleryEntry
        distance: Computed distance metric between query and gallery embedding
        score: Normalized similarity score (higher is more similar)
        metadata: Gallery entry metadata passthrough
    """
    gallery_subject_id: str
    distance: float
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class SimilarityResult:
    """
    Complete similarity search result.

    Attributes:
        query_id: Optional identifier for the query embedding
        matches: Ranked list of similarity matches
        distance_metric: Metric used for distance computation
        threshold: Decision boundary used for verification (if applicable)
    """
    matches: list[SimilarityMatch]
    distance_metric: Literal["cosine", "euclidean", "euclidean_l2"]
    threshold: Optional[float] = None
    query_id: Optional[str] = None


# ==================== Conversion Utilities ====================


def person_embedding_from_legacy_embedding_set(
    embedding_set,  # Type: EmbeddingSet from user_gallery.models
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0),
    normalization: Literal["base", "resnet", "circle"] = "resnet",
    model_profile_id: str = "resnet50_circle_dg",
    hardware: Literal["cuda", "cpu"] = "cuda",
) -> PersonEmbedding:
    """
    Convert a user_gallery EmbeddingSet to a unified PersonEmbedding.

    This utility helps migrate from the legacy user_gallery data model to the
    unified core PersonEmbedding model.

    Args:
        embedding_set: Legacy EmbeddingSet instance from user_gallery.models
        bbox: Bounding box (required for PersonEmbedding, defaults to (0, 0, 0, 0))
        normalization: Normalization method (defaults to "resnet")
        model_profile_id: Model identifier (defaults to "resnet50_circle_dg")
        hardware: Hardware used (defaults to "cuda")

    Returns:
        PersonEmbedding: Unified embedding with all fields mapped

    Example:
        >>> from user_gallery.models import EmbeddingSet
        >>> legacy_emb = EmbeddingSet(...)
        >>> unified_emb = person_embedding_from_legacy_embedding_set(legacy_emb)
    """
    # Determine modality
    if embedding_set.face_embedding is not None:
        modality = Modality.BODY_FACE
    else:
        modality = Modality.BODY

    return PersonEmbedding(
        # Core fields
        embedding_vector=embedding_set.body_embedding,
        subject_confidence=embedding_set.quality_score,  # Map quality to confidence
        bbox=bbox,
        normalization=normalization,
        model_profile_id=model_profile_id,
        hardware=hardware,
        timestamp=embedding_set.generated_at,
        source_image_id=embedding_set.image_id,
        # Multi-modal fields
        modality=modality,
        face_embedding=embedding_set.face_embedding,
        face_confidence=None,  # Not available in legacy model
        face_bbox=None,  # Not available in legacy model
        # User gallery fields
        user_id=embedding_set.user_id,
        embedding_id=embedding_set.embedding_id,
        cluster_id=embedding_set.cluster_id,
        quality_score=embedding_set.quality_score,
        embedding_provider=embedding_set.embedding_provider,
        embedding_version=embedding_set.embedding_version,
        # Metadata
        metadata=embedding_set.metadata,
    )


def person_embedding_to_legacy_embedding_set(
    person_embedding: PersonEmbedding,
    embedding_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Convert a unified PersonEmbedding to legacy EmbeddingSet format (as dict).

    This utility helps maintain compatibility with existing user_gallery code
    during the migration period.

    Args:
        person_embedding: Unified PersonEmbedding instance
        embedding_id: Override embedding_id (uses person_embedding.embedding_id if None)
        user_id: Override user_id (uses person_embedding.user_id if None)

    Returns:
        dict: Dictionary compatible with EmbeddingSet initialization

    Example:
        >>> from user_gallery.models import EmbeddingSet
        >>> unified_emb = PersonEmbedding(...)
        >>> legacy_dict = person_embedding_to_legacy_embedding_set(unified_emb)
        >>> legacy_emb = EmbeddingSet(**legacy_dict)
    """
    import uuid

    return {
        "embedding_id": embedding_id or person_embedding.embedding_id or str(uuid.uuid4()),
        "user_id": user_id or person_embedding.user_id or "unknown",
        "body_embedding": person_embedding.embedding_vector,
        "face_embedding": person_embedding.face_embedding,
        "embedding_provider": person_embedding.embedding_provider or person_embedding.model_profile_id,
        "embedding_version": person_embedding.embedding_version or "1.0",
        "quality_score": person_embedding.quality_score or person_embedding.subject_confidence,
        "generated_at": person_embedding.timestamp or datetime.now(),
        "image_id": person_embedding.source_image_id,
        "cluster_id": person_embedding.cluster_id,
        "metadata": person_embedding.metadata,
    }
