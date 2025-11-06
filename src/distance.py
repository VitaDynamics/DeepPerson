"""
Distance Metrics Utility

Provides utility functions for computing various distance metrics between embeddings.
Implements cosine, euclidean, and L2-normalized euclidean distances.
"""

from typing import Any, Union
import numpy as np


class DistanceMetrics:
    """
    Utility class for computing various distance metrics between embeddings.

    Implements cosine, euclidean, and L2-normalized euclidean distances
    following DeepFace verification patterns.
    """

    @staticmethod
    def cosine_distance(embedding1: Union[np.ndarray, Any], embedding2: Union[np.ndarray, Any]) -> float:
        """
        Compute cosine distance between two embeddings.

        Cosine distance = 1 - cosine_similarity
        where cosine_similarity = dot(a, b) / (||a|| * ||b||)

        Args:
            embedding1: First embedding vector (numpy array)
            embedding2: Second embedding vector (numpy array)

        Returns:
            Cosine distance in range [0, 2], where 0 = identical, 2 = opposite

        Examples:
            >>> emb1 = np.array([1.0, 0.0, 0.0])
            >>> emb2 = np.array([1.0, 0.0, 0.0])
            >>> DistanceMetrics.cosine_distance(emb1, emb2)
            0.0
        """
        # Ensure numpy arrays
        emb1 = np.asarray(embedding1, dtype=np.float32)
        emb2 = np.asarray(embedding2, dtype=np.float32)

        # Flatten to 1D if needed
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        # Validate same dimensionality
        if emb1.shape != emb2.shape:
            raise ValueError(
                f"Embedding dimensions must match: {emb1.shape} vs {emb2.shape}"
            )

        # Compute cosine similarity: dot product / (norm1 * norm2)
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        # Prevent division by zero
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance for zero vectors

        cosine_similarity = dot_product / (norm1 * norm2)

        # Clamp to [-1, 1] to handle numerical precision issues
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

        # Cosine distance = 1 - cosine_similarity
        # Range: [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite
        distance = 1.0 - cosine_similarity

        return float(distance)

    @staticmethod
    def euclidean_distance(embedding1: Union[np.ndarray, Any], embedding2: Union[np.ndarray, Any]) -> float:
        """
        Compute Euclidean (L2) distance between two embeddings.

        Euclidean distance = sqrt(sum((a - b)^2))

        Args:
            embedding1: First embedding vector (numpy array)
            embedding2: Second embedding vector (numpy array)

        Returns:
            Euclidean distance (non-negative float)

        Examples:
            >>> emb1 = np.array([1.0, 0.0, 0.0])
            >>> emb2 = np.array([0.0, 1.0, 0.0])
            >>> DistanceMetrics.euclidean_distance(emb1, emb2)
            1.4142135...
        """
        # Ensure numpy arrays
        emb1 = np.asarray(embedding1, dtype=np.float32)
        emb2 = np.asarray(embedding2, dtype=np.float32)

        # Flatten to 1D if needed
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        # Validate same dimensionality
        if emb1.shape != emb2.shape:
            raise ValueError(
                f"Embedding dimensions must match: {emb1.shape} vs {emb2.shape}"
            )

        # Compute Euclidean distance
        distance = np.linalg.norm(emb1 - emb2)

        return float(distance)

    @staticmethod
    def euclidean_l2_distance(embedding1: Union[np.ndarray, Any], embedding2: Union[np.ndarray, Any]) -> float:
        """
        Compute Euclidean distance between L2-normalized embeddings.

        This metric first normalizes both embeddings to unit length, then
        computes the Euclidean distance. This is equivalent to:
        sqrt(2 - 2 * cosine_similarity) but numerically more stable.

        Args:
            embedding1: First embedding vector (numpy array)
            embedding2: Second embedding vector (numpy array)

        Returns:
            L2-normalized Euclidean distance in range [0, 2]

        Examples:
            >>> emb1 = np.array([1.0, 0.0, 0.0])
            >>> emb2 = np.array([2.0, 0.0, 0.0])  # Same direction, different magnitude
            >>> DistanceMetrics.euclidean_l2_distance(emb1, emb2)
            0.0
        """
        # Ensure numpy arrays
        emb1 = np.asarray(embedding1, dtype=np.float32)
        emb2 = np.asarray(embedding2, dtype=np.float32)

        # Flatten to 1D if needed
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        # Validate same dimensionality
        if emb1.shape != emb2.shape:
            raise ValueError(
                f"Embedding dimensions must match: {emb1.shape} vs {emb2.shape}"
            )

        # L2-normalize both embeddings
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        # Handle zero vectors
        if norm1 == 0 or norm2 == 0:
            return 2.0  # Maximum distance for zero vectors

        emb1_normalized = emb1 / norm1
        emb2_normalized = emb2 / norm2

        # Compute Euclidean distance on normalized embeddings
        distance = np.linalg.norm(emb1_normalized - emb2_normalized)

        return float(distance)


def compute_distance(embedding1: Any, embedding2: Any, metric: str = "cosine") -> float:
    """
    Compute distance between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding
        metric: Distance metric to use

    Returns:
        Distance value
    """
    if metric == "cosine":
        return DistanceMetrics.cosine_distance(embedding1, embedding2)
    elif metric == "euclidean":
        return DistanceMetrics.euclidean_distance(embedding1, embedding2)
    elif metric == "euclidean_l2":
        return DistanceMetrics.euclidean_l2_distance(embedding1, embedding2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
