"""
Abstract interfaces for DeepPerson components.

This module defines abstract base classes (ABCs) that provide common interfaces
for embedding generators and other components. This enables polymorphism and
makes the codebase more extensible and testable.
"""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import numpy as np
from PIL import Image

from .entities import PersonEmbedding


class EmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.

    This interface defines the contract for all embedding generation components,
    whether they generate body embeddings, face embeddings, or multi-modal embeddings.

    Implementations:
        - BodyEmbeddingGenerator: Body embeddings using ReID models
        - FaceEmbeddingGenerator: Face embeddings using face recognition models
        - MultiModalEmbeddingGenerator: Combined body+face embeddings
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model identifier for this generator."""
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Get the dimensionality of generated embeddings."""
        pass

    @property
    @abstractmethod
    def modality(self) -> str:
        """Get the modality type (BODY, FACE, or BODY_FACE)."""
        pass

    @abstractmethod
    def generate_embedding(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        confidence: float,
        normalize_method: Literal["base", "resnet", "circle"] = "resnet",
        source_image_id: Optional[str] = None,
        **kwargs
    ) -> PersonEmbedding:
        """
        Generate embedding for a single person image.

        Args:
            image: Cropped person or face image (PIL Image)
            bbox: Bounding box in original image (x1, y1, x2, y2)
            confidence: Detection confidence
            normalize_method: Normalization method for embedding
            source_image_id: Optional source image identifier
            **kwargs: Additional generator-specific parameters

        Returns:
            PersonEmbedding: Embedding with metadata

        Raises:
            ValueError: If image is invalid or processing fails
        """
        pass

    @abstractmethod
    def generate_embeddings_batch(
        self,
        images: List[Image.Image],
        bboxes: List[tuple[int, int, int, int]],
        confidences: List[float],
        normalize_method: Literal["base", "resnet", "circle"] = "resnet",
        source_image_ids: Optional[List[str]] = None,
        batch_size: int = 16,
        show_progress: bool = False,
        **kwargs
    ) -> List[PersonEmbedding]:
        """
        Generate embeddings for multiple images with batching.

        Args:
            images: List of cropped person/face images
            bboxes: List of bounding boxes
            confidences: List of detection confidences
            normalize_method: Normalization method
            source_image_ids: Optional list of source image identifiers
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            **kwargs: Additional generator-specific parameters

        Returns:
            List[PersonEmbedding]: List of embeddings with metadata

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        pass
