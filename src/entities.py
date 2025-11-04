"""
Data entities for DeepPerson minimal embedding library.

Defines dataclasses for embeddings, model profiles, gallery entries, and similarity results.
Based on data-model.md specifications.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np


@dataclass
class PersonEmbedding:
    """
    Embedding representation for a detected person.

    Attributes:
        embedding_vector: Feature vector produced by backbone (shape: feature_dim,)
        subject_confidence: Detector confidence for the associated bounding box
        bbox: Bounding box coordinates (x1, y1, x2, y2) in original image coordinates
        normalization: Normalization strategy applied to the embedding
        model_profile_id: Reference to ModelProfile.identifier
        hardware: Device used for embedding generation
        timestamp: Optional capture or processing timestamp
        source_image_id: Optional identifier linking to original image asset
    """
    embedding_vector: np.ndarray  # shape: (feature_dim,), dtype: float32
    subject_confidence: float
    bbox: tuple[int, int, int, int]
    normalization: Literal["base", "resnet", "circle"]
    model_profile_id: str
    hardware: Literal["cuda", "cpu"]
    timestamp: Optional[datetime] = None
    source_image_id: Optional[str] = None

    def __post_init__(self):
        """Validate embedding vector properties."""
        if not isinstance(self.embedding_vector, np.ndarray):
            raise TypeError(f"embedding_vector must be numpy.ndarray, got {type(self.embedding_vector)}")
        if self.embedding_vector.ndim != 1:
            raise ValueError(f"embedding_vector must be 1-dimensional, got shape {self.embedding_vector.shape}")
        if self.embedding_vector.dtype != np.float32:
            self.embedding_vector = self.embedding_vector.astype(np.float32)


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
