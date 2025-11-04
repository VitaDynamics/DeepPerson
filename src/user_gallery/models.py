"""
User gallery data models for multi-modal person re-identification.

This module defines the core data structures for managing user galleries
with both body and face embeddings, including variant clustering and
fusion scoring capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from pathlib import Path


class Modality(str, Enum):
    """Image modality types."""
    BODY = "BODY"
    FACE = "FACE"
    UNKNOWN = "UNKNOWN"


class ProcessingStatus(str, Enum):
    """Image processing status states."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class GalleryStatus(str, Enum):
    """User gallery status states."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"


class ConfidenceLevel(str, Enum):
    """Confidence levels for retrieval results."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ImageAsset:
    """Stores media inputs (body or face) with source descriptors and processing status."""

    image_id: str
    user_id: str
    image_path: str
    modality: Modality = Modality.UNKNOWN
    source_camera: Optional[str] = None
    capture_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    face_detection_confidence: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    cluster_id: Optional[str] = None

    def __post_init__(self):
        """Validate image asset constraints."""
        if self.face_detection_confidence is not None:
            if not 0.0 <= self.face_detection_confidence <= 1.0:
                raise ValueError("Face detection confidence must be between 0.0 and 1.0")

    @property
    def is_face_detected(self) -> bool:
        """Check if face detection was successful."""
        return self.modality == Modality.FACE or (
            self.face_detection_confidence is not None and
            self.face_detection_confidence > 0.5
        )


@dataclass
class EmbeddingSet:
    """Encapsulates body and face embedding vectors with provider metadata."""

    embedding_id: str
    user_id: str
    body_embedding: np.ndarray
    embedding_provider: str
    embedding_version: str
    quality_score: float
    generated_at: datetime = field(default_factory=datetime.now)
    face_embedding: Optional[np.ndarray] = None
    image_id: Optional[str] = None
    cluster_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate embedding constraints."""
        # Validate body embedding dimensions
        if self.body_embedding is None or len(self.body_embedding) == 0:
            raise ValueError("Body embedding must be provided and non-empty")

        # Validate embedding dimensions (typically 2048 for ResNet-50 Circle DG)
        if len(self.body_embedding) < 512:
            raise ValueError(f"Body embedding dimension {len(self.body_embedding)} is too small")

        # Validate face embedding if present
        if self.face_embedding is not None and len(self.face_embedding) < 128:
            raise ValueError(f"Face embedding dimension {len(self.face_embedding)} is too small")

        # Validate quality score
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")

        # Ensure at least one embedding type is present
        if self.body_embedding is None and self.face_embedding is None:
            raise ValueError("At least one embedding type (body or face) must be present")

    @property
    def has_face_embedding(self) -> bool:
        """Check if face embedding is available."""
        return self.face_embedding is not None

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of body embeddings."""
        return len(self.body_embedding) if self.body_embedding is not None else 0

    def normalize_body_embedding(self) -> None:
        """Normalize body embedding to unit vector."""
        if self.body_embedding is not None:
            norm = np.linalg.norm(self.body_embedding)
            if norm > 0:
                self.body_embedding = self.body_embedding / norm

    def normalize_face_embedding(self) -> None:
        """Normalize face embedding to unit vector."""
        if self.face_embedding is not None:
            norm = np.linalg.norm(self.face_embedding)
            if norm > 0:
                self.face_embedding = self.face_embedding / norm


@dataclass
class VariantCluster:
    """Captures a specific appearance cluster for a user (e.g., outfit, time period)."""

    cluster_id: str
    user_id: str
    cluster_name: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    image_count: int = 0
    primary_image_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate cluster constraints."""
        if self.image_count < 0:
            raise ValueError("Image count cannot be negative")

    def add_image(self, image_id: str) -> None:
        """Add an image to this cluster."""
        self.image_count += 1
        if self.primary_image_id is None:
            self.primary_image_id = image_id

    def remove_image(self) -> None:
        """Remove an image from this cluster."""
        self.image_count = max(0, self.image_count - 1)
        if self.image_count == 0:
            self.primary_image_id = None


@dataclass
class UserGallery:
    """Represents a unique person within the system with aggregated media and embeddings."""

    user_id: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: GalleryStatus = GalleryStatus.ACTIVE
    variant_clusters: List[VariantCluster] = field(default_factory=list)

    def __post_init__(self):
        """Validate gallery constraints."""
        if not self.user_id or not self.user_id.strip():
            raise ValueError("User ID must be non-empty")

    @property
    def total_images(self) -> int:
        """Get total number of images across all clusters."""
        return sum(cluster.image_count for cluster in self.variant_clusters)

    @property
    def total_clusters(self) -> int:
        """Get total number of variant clusters."""
        return len(self.variant_clusters)

    @property
    def has_face_images(self) -> bool:
        """Check if gallery contains any face images."""
        # This would be populated when images are added
        return False

    @property
    def has_body_images(self) -> bool:
        """Check if gallery contains any body images."""
        # This would be populated when images are added
        return False

    def add_cluster(self, cluster: VariantCluster) -> None:
        """Add a variant cluster to the gallery."""
        if cluster.user_id != self.user_id:
            raise ValueError("Cluster user_id must match gallery user_id")
        self.variant_clusters.append(cluster)
        self.updated_at = datetime.now()

    def get_cluster(self, cluster_id: str) -> Optional[VariantCluster]:
        """Get a cluster by ID."""
        for cluster in self.variant_clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None

    def update_status(self, new_status: GalleryStatus) -> None:
        """Update gallery status."""
        self.status = new_status
        self.updated_at = datetime.now()


@dataclass
class RetrievalProbe:
    """Represents a query instance containing image input and derived embeddings."""

    probe_id: str
    probe_image_path: str
    probe_modality: Modality = Modality.AUTO_DETECT
    generated_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    fusion_weights: Dict[str, float] = field(default_factory=dict)
    retrieval_config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None

    def __post_init__(self):
        """Validate probe constraints."""
        if not self.probe_id or not self.probe_image_path:
            raise ValueError("Probe ID and image path are required")

    @property
    def has_body_embedding(self) -> bool:
        """Check if body embedding is available."""
        return 'body' in self.generated_embeddings

    @property
    def has_face_embedding(self) -> bool:
        """Check if face embedding is available."""
        return 'face' in self.generated_embeddings

    def add_embedding(self, modality: str, embedding: np.ndarray) -> None:
        """Add an embedding for a specific modality."""
        self.generated_embeddings[modality] = embedding

    def set_fusion_weight(self, modality: str, weight: float) -> None:
        """Set fusion weight for a modality."""
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Fusion weight must be between 0.0 and 1.0")
        self.fusion_weights[modality] = weight


@dataclass
class RetrievalResult:
    """Stores retrieval results with user-level scores and evidence."""

    result_id: str
    probe_id: str
    user_id: str
    overall_score: float
    retrieved_at: datetime = field(default_factory=datetime.now)

    # Optional modality-specific scores
    face_score: Optional[float] = None
    body_score: Optional[float] = None

    # Applied weights
    face_weight: float = 0.5
    body_weight: float = 0.5

    # Evidence and metadata
    evidence_images: List[str] = field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result constraints."""
        if not 0.0 <= self.overall_score <= 1.0:
            raise ValueError("Overall score must be between 0.0 and 1.0")

        if not 0.0 <= self.face_weight <= 1.0 or not 0.0 <= self.body_weight <= 1.0:
            raise ValueError("Weights must be between 0.0 and 1.0")

        if abs(self.face_weight + self.body_weight - 1.0) > 1e-6:
            raise ValueError("Face and body weights must sum to 1.0")

    @property
    def has_face_contribution(self) -> bool:
        """Check if face modality contributed to this result."""
        return self.face_score is not None and self.face_weight > 0

    @property
    def has_body_contribution(self) -> bool:
        """Check if body modality contributed to this result."""
        return self.body_score is not None and self.body_weight > 0

    def add_evidence_image(self, image_id: str) -> None:
        """Add an evidence image to this result."""
        if image_id not in self.evidence_images:
            self.evidence_images.append(image_id)

    def update_scores(self, face_score: Optional[float] = None,
                     body_score: Optional[float] = None) -> None:
        """Update modality-specific scores."""
        if face_score is not None:
            self.face_score = face_score
        if body_score is not None:
            self.body_score = body_score

        # Recalculate overall score if both components are available
        if self.face_score is not None and self.body_score is not None:
            self.overall_score = (self.face_score * self.face_weight +
                                self.body_score * self.body_weight)


# Type aliases for convenience
UserGalleryDict = Dict[str, Any]
ImageAssetDict = Dict[str, Any]
EmbeddingSetDict = Dict[str, Any]
RetrievalResultDict = Dict[str, Any]