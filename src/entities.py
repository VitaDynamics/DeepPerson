"""
Data entities for DeepPerson minimal embedding library.

Defines dataclasses for embeddings, model profiles, gallery entries, and similarity results.
Based on data-model.md specifications.

This module supports both single-modal (body-only) and multi-modal (body+face) embeddings.
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

    Supports both single-modal (body-only) and multi-modal (body+face) embeddings.

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

    def compute_quality_score(
        self,
        detection_confidence: Optional[float] = None,
        normalization_check: bool = True,
    ) -> float:
        """
        Compute quality score for this embedding.

        Quality is based on:
        - Detection confidence (uses subject_confidence if not provided)
        - Embedding norm (should be close to 1.0 if normalized)
        - Embedding variance (should not be too low)

        Args:
            detection_confidence: Override detection confidence (uses subject_confidence if None)
            normalization_check: Whether to check if embedding is normalized

        Returns:
            Quality score between 0.0 and 1.0

        Examples:
            >>> emb = PersonEmbedding(...)
            >>> quality = emb.compute_quality_score()
            >>> print(f"Embedding quality: {quality:.3f}")
        """
        if self.embedding_vector is None or len(self.embedding_vector) == 0:
            return 0.0

        # Start with detection confidence
        confidence = detection_confidence if detection_confidence is not None else self.subject_confidence
        quality = confidence

        # Check normalization
        if normalization_check:
            norm = np.linalg.norm(self.embedding_vector)
            # Penalize if not normalized (should be close to 1.0)
            norm_score = 1.0 - min(abs(norm - 1.0), 1.0)
            quality *= norm_score

        # Check variance (embeddings with very low variance are suspicious)
        variance = np.var(self.embedding_vector)
        if variance < 1e-6:
            quality *= 0.5  # Penalize low variance

        # Ensure quality is in valid range
        quality = max(0.0, min(1.0, quality))

        return quality

    def with_face_embedding(
        self,
        face_embedding: np.ndarray,
        face_confidence: float,
        face_bbox: Optional[tuple[int, int, int, int]] = None,
        **additional_fields
    ) -> "PersonEmbedding":
        """
        Create a new PersonEmbedding with face embedding merged in.

        Returns a new instance with BODY_FACE modality and face fields populated.
        Useful for adding face embeddings to existing body-only embeddings without
        verbose field-by-field copying.

        Args:
            face_embedding: Face embedding vector
            face_confidence: Face detection confidence (0.0-1.0)
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
            **additional_fields: Additional fields to override (user_id, cluster_id, etc.)

        Returns:
            New PersonEmbedding with face embedding included and modality set to BODY_FACE

        Examples:
            >>> body_emb = PersonEmbedding(...)
            >>> # Add face embedding and user gallery fields
            >>> multi_modal_emb = body_emb.with_face_embedding(
            ...     face_embedding=face_vec,
            ...     face_confidence=0.95,
            ...     face_bbox=(120, 110, 180, 170),
            ...     user_id="user_001",
            ...     cluster_id="cluster_a"
            ... )
            >>> assert multi_modal_emb.modality == Modality.BODY_FACE
            >>> assert multi_modal_emb.has_face_embedding
        """
        # Create a copy of current embedding data
        new_data = {
            # Core fields
            "embedding_vector": self.embedding_vector.copy(),
            "subject_confidence": self.subject_confidence,
            "bbox": self.bbox,
            "normalization": self.normalization,
            "model_profile_id": self.model_profile_id,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
            "source_image_id": self.source_image_id,
            # Multi-modal fields (update with face data)
            "modality": Modality.BODY_FACE,
            "face_embedding": face_embedding,
            "face_confidence": face_confidence,
            "face_bbox": face_bbox or self.face_bbox,
            # User gallery fields
            "user_id": self.user_id,
            "embedding_id": self.embedding_id,
            "cluster_id": self.cluster_id,
            "quality_score": self.quality_score,
            "embedding_provider": self.embedding_provider,
            "embedding_version": self.embedding_version,
            # Metadata
            "metadata": self.metadata.copy(),
        }

        # Override with additional fields
        new_data.update(additional_fields)

        return PersonEmbedding(**new_data)


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
