"""
Abstract interfaces for DeepPerson components.

This module defines abstract base classes (ABCs) that provide common interfaces
for embedding generators, searchers, and other components. This enables
polymorphism and makes the codebase more extensible and testable.
"""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import numpy as np
from PIL import Image

from .entities import PersonEmbedding


class EmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.

    This interface defines the contract for all embedding generation components,
    whether they generate body embeddings, face embeddings, or multi-modal embeddings.

    Implementations:
        - EmbeddingPipeline: Body embeddings using ReID models
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


class Searcher(ABC):
    """
    Abstract base class for similarity search engines.

    This interface defines the contract for all searcher implementations,
    whether they use FAISS, sklearn, or other backends.

    Implementations:
        - FAISSSearcher: GPU/CPU-accelerated similarity search
        - SklearnSearcher: CPU-based similarity search
        - MultiModalSearcher: Coordinates multiple modality searchers
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the embedding dimension for this searcher."""
        pass

    @property
    @abstractmethod
    def metric(self) -> str:
        """Get the distance metric used (cosine, euclidean, etc.)."""
        pass

    @property
    @abstractmethod
    def subject_ids(self) -> List[str]:
        """Get the list of all subject IDs in the index."""
        pass

    @abstractmethod
    def add_embedding(
        self,
        embedding: np.ndarray,
        subject_id: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Add a single embedding to the search index.

        Args:
            embedding: Feature vector (1D numpy array)
            subject_id: Unique identifier for this subject
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If embedding dimension doesn't match searcher dimension
        """
        pass

    @abstractmethod
    def add_embeddings_batch(
        self,
        embeddings: np.ndarray,
        subject_ids: List[str],
        metadata: Optional[List[dict]] = None
    ) -> None:
        """
        Add multiple embeddings to the search index (batch operation).

        Args:
            embeddings: Feature matrix (N, dimension)
            subject_ids: List of subject IDs (length N)
            metadata: Optional list of metadata dicts (length N)

        Raises:
            ValueError: If shapes don't match or embeddings already exist
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[dict]:
        """
        Search for k nearest neighbors to the query embedding.

        Args:
            query_embedding: Query feature vector (1D numpy array)
            k: Number of nearest neighbors to return
            threshold: Optional distance threshold for filtering results

        Returns:
            List of dicts containing:
                - subject_id: Subject identifier
                - distance: Distance to query
                - metadata: Associated metadata (if any)

        Raises:
            ValueError: If query dimension doesn't match searcher dimension
            RuntimeError: If index is empty
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the search index to disk.

        Args:
            path: Directory path to save index files

        Raises:
            IOError: If save operation fails
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the search index from disk.

        Args:
            path: Directory path containing index files

        Raises:
            FileNotFoundError: If index files don't exist
            IOError: If load operation fails
        """
        pass


class StorageBackend(ABC):
    """
    Abstract base class for gallery storage backends.

    This interface defines the contract for storage systems that persist
    galleries, embeddings, and metadata.

    Implementations:
        - FileStorageBackend: File-based storage (NPY, NPZ, JSON)
        - DatabaseStorageBackend: SQL database storage (future)
        - CloudStorageBackend: Cloud object storage (future)
    """

    @abstractmethod
    def save_embeddings(
        self,
        embeddings: List[PersonEmbedding],
        gallery_id: str,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Save embeddings for a gallery.

        Args:
            embeddings: List of PersonEmbedding objects
            gallery_id: Unique gallery identifier
            metadata: Optional gallery metadata

        Returns:
            str: Path or identifier where embeddings were saved

        Raises:
            IOError: If save operation fails
        """
        pass

    @abstractmethod
    def load_embeddings(
        self,
        gallery_id: str
    ) -> tuple[List[PersonEmbedding], Optional[dict]]:
        """
        Load embeddings for a gallery.

        Args:
            gallery_id: Unique gallery identifier

        Returns:
            Tuple of (embeddings, metadata)

        Raises:
            FileNotFoundError: If gallery doesn't exist
            IOError: If load operation fails
        """
        pass

    @abstractmethod
    def delete_embeddings(
        self,
        gallery_id: str,
        permanent: bool = False
    ) -> bool:
        """
        Delete embeddings for a gallery.

        Args:
            gallery_id: Unique gallery identifier
            permanent: If True, permanently delete; if False, soft delete

        Returns:
            bool: True if deletion successful

        Raises:
            FileNotFoundError: If gallery doesn't exist
        """
        pass

    @abstractmethod
    def list_galleries(self) -> List[str]:
        """
        List all available gallery identifiers.

        Returns:
            List of gallery IDs

        Raises:
            IOError: If listing operation fails
        """
        pass
