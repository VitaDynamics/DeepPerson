"""
Similarity Search and Gallery Management

Handles similarity search operations, gallery management, and distance metrics.
Provides both FAISS-accelerated and scikit-learn fallback implementations.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import warnings
from abc import ABC, abstractmethod
import logging
import threading
from pathlib import Path
import json

import numpy as np

# FAISS import with availability detection
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Sklearn imports
try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .entities import PersonEmbedding, GalleryEntry, SimilarityResult, SimilarityMatch

logger = logging.getLogger(__name__)


class SimilaritySearcher(ABC):
    """
    Abstract base class for similarity search implementations.
    """

    @abstractmethod
    def add_embedding(self, embedding: Any, subject_id: str, metadata: Dict[str, Any] = None):
        """Add an embedding to the search index."""
        pass

    @abstractmethod
    def search(self, query_embedding: Any, k: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the search index to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the search index from disk."""
        pass


class FAISSSearcher(SimilaritySearcher):
    """
    FAISS-based similarity search implementation.

    Features:
    - Thread-safe gallery operations with RLock
    - GPU/CPU auto-detection and fallback
    - Supports cosine (IndexFlatIP) and euclidean (IndexFlatL2) metrics
    - Batch search optimization
    """

    def __init__(self, dimension: int, metric: str = "cosine", device: str = "cpu"):
        """
        Initialize FAISS searcher.

        Args:
            dimension: Embedding dimension
            metric: Distance metric to use ('cosine', 'euclidean', 'euclidean_l2')
            device: Device to run on ('cpu', 'cuda', or 'cuda:0', etc.)

        Raises:
            ImportError: If FAISS is not available
            ValueError: If unsupported metric or device
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not available. Install with: pip install faiss-cpu or faiss-gpu"
            )

        self.dimension = dimension
        self.metric = metric
        self.device = device
        self._lock = threading.RLock()

        # Initialize storage for metadata
        self.embeddings_list = []  # Store embeddings for metadata retrieval
        self.subject_ids = []
        self.metadata_list = []

        # Create FAISS index
        self.index = self._create_faiss_index(dimension, metric, device)

        logger.info(f"Initialized FAISSSearcher: dim={dimension}, metric={metric}, device={device}")

    def _create_faiss_index(self, dimension: int, metric: str, device: str):
        """
        Create appropriate FAISS index based on metric and device.

        Args:
            dimension: Embedding dimension
            metric: Distance metric
            device: Target device

        Returns:
            FAISS index instance
        """
        # Create CPU index based on metric
        if metric == "cosine":
            # IndexFlatIP for inner product (cosine similarity on normalized embeddings)
            cpu_index = faiss.IndexFlatIP(dimension)
            logger.debug("Created IndexFlatIP for cosine similarity")
        elif metric in ["euclidean", "euclidean_l2"]:
            # IndexFlatL2 for L2/Euclidean distance
            cpu_index = faiss.IndexFlatL2(dimension)
            logger.debug("Created IndexFlatL2 for euclidean distance")
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'cosine', 'euclidean', or 'euclidean_l2'")

        # Try to move to GPU if requested
        if device.startswith("cuda"):
            try:
                # Parse GPU device ID
                if ":" in device:
                    gpu_id = int(device.split(":")[1])
                else:
                    gpu_id = 0

                # Move index to GPU
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                logger.info(f"Successfully moved FAISS index to GPU {gpu_id}")
                return gpu_index
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU, falling back to CPU: {e}")
                return cpu_index
        else:
            return cpu_index

    def add_embedding(self, embedding: Any, subject_id: str, metadata: Dict[str, Any] = None):
        """
        Add an embedding to the FAISS index.

        Args:
            embedding: Embedding vector (numpy array or PersonEmbedding)
            subject_id: Subject identifier
            metadata: Optional metadata dictionary
        """
        with self._lock:
            # Extract numpy array from PersonEmbedding if needed
            if isinstance(embedding, PersonEmbedding):
                emb_array = embedding.embedding_vector
                if metadata is None:
                    metadata = {"model_profile": embedding.model_profile}
            else:
                emb_array = np.asarray(embedding, dtype=np.float32)

            # Validate dimension
            if emb_array.shape[-1] != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.dimension}, got {emb_array.shape[-1]}"
                )

            # Reshape to 2D for FAISS (batch dimension)
            emb_array = emb_array.reshape(1, -1)

            # Normalize for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(emb_array)

            # Add to index
            self.index.add(emb_array)

            # Store metadata
            self.embeddings_list.append(emb_array[0])
            self.subject_ids.append(subject_id)
            self.metadata_list.append(metadata or {})

            logger.debug(f"Added embedding for subject: {subject_id}")

    def add_batch(self, embeddings: np.ndarray, subject_ids: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None):
        """
        Add multiple embeddings in batch (more efficient than individual adds).

        Args:
            embeddings: Embedding matrix (shape: [N, dimension])
            subject_ids: List of subject identifiers
            metadata_list: Optional list of metadata dictionaries
        """
        with self._lock:
            embeddings = np.asarray(embeddings, dtype=np.float32)

            # Validate inputs
            if embeddings.ndim != 2:
                raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
            if embeddings.shape[0] != len(subject_ids):
                raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) must match subject_ids ({len(subject_ids)})")
            if embeddings.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")

            # Normalize for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(embeddings)

            # Add to index
            self.index.add(embeddings)

            # Store metadata
            for i, (emb, subject_id) in enumerate(zip(embeddings, subject_ids)):
                self.embeddings_list.append(emb)
                self.subject_ids.append(subject_id)
                self.metadata_list.append(metadata_list[i] if metadata_list else {})

            logger.info(f"Added {len(subject_ids)} embeddings in batch")

    def search(self, query_embedding: Any, k: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using FAISS.

        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            threshold: Optional distance threshold for filtering results

        Returns:
            List of match dictionaries with keys: subject_id, distance, metadata
        """
        with self._lock:
            # Check if index is empty
            if self.index.ntotal == 0:
                logger.warning("Search called on empty index")
                return []

            # Extract numpy array
            if isinstance(query_embedding, PersonEmbedding):
                query_array = query_embedding.embedding_vector
            else:
                query_array = np.asarray(query_embedding, dtype=np.float32)

            # Reshape to 2D
            query_array = query_array.reshape(1, -1)

            # Normalize for cosine similarity
            if self.metric == "cosine":
                faiss.normalize_L2(query_array)

            # Clamp k to index size
            k = min(k, self.index.ntotal)

            # Perform search
            distances, indices = self.index.search(query_array, k)

            # Convert to results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                # Skip invalid indices
                if idx < 0 or idx >= len(self.subject_ids):
                    continue

                # Apply threshold if provided
                if threshold is not None and dist > threshold:
                    continue

                results.append({
                    "subject_id": self.subject_ids[idx],
                    "distance": float(dist),
                    "metadata": self.metadata_list[idx]
                })

            logger.debug(f"Search returned {len(results)} results (k={k}, threshold={threshold})")
            return results

    def save(self, path: str):
        """
        Save FAISS index and metadata to disk.

        Args:
            path: Directory path to save index and metadata
        """
        with self._lock:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            index_path = path / "index.faiss"
            faiss.write_index(faiss.index_gpu_to_cpu(self.index) if str(self.device).startswith("cuda") else self.index, str(index_path))

            # Save embeddings
            embeddings_path = path / "embeddings.npy"
            np.save(embeddings_path, np.array(self.embeddings_list))

            # Save metadata
            metadata_path = path / "metadata.json"
            metadata = {
                "dimension": self.dimension,
                "metric": self.metric,
                "device": self.device,
                "total_entries": len(self.subject_ids),
                "subject_ids": self.subject_ids,
                "metadata_list": self.metadata_list
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str):
        """
        Load FAISS index and metadata from disk.

        Args:
            path: Directory path containing saved index and metadata
        """
        with self._lock:
            path = Path(path)

            # Load metadata
            metadata_path = path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Validate compatibility
            if metadata["dimension"] != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: current={self.dimension}, saved={metadata['dimension']}"
                )

            # Load FAISS index
            index_path = path / "index.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")

            cpu_index = faiss.read_index(str(index_path))

            # Move to GPU if needed
            if self.device.startswith("cuda"):
                try:
                    gpu_id = int(self.device.split(":")[1]) if ":" in self.device else 0
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
                except Exception as e:
                    logger.warning(f"Failed to move index to GPU, using CPU: {e}")
                    self.index = cpu_index
            else:
                self.index = cpu_index

            # Load embeddings
            embeddings_path = path / "embeddings.npy"
            if embeddings_path.exists():
                embeddings = np.load(embeddings_path)
                self.embeddings_list = list(embeddings)

            # Restore metadata
            self.subject_ids = metadata["subject_ids"]
            self.metadata_list = metadata["metadata_list"]

            logger.info(f"Loaded FAISS index from {path}: {metadata['total_entries']} entries}")


class SklearnSearcher(SimilaritySearcher):
    """
    Scikit-learn based similarity search implementation (fallback).

    Uses NearestNeighbors for k-NN search when FAISS is not available.
    Thread-safe and supports same interface as FAISSSearcher.
    """

    def __init__(self, metric: str = "cosine"):
        """
        Initialize sklearn searcher.

        Args:
            metric: Distance metric to use ('cosine', 'euclidean', 'euclidean_l2')

        Raises:
            ImportError: If sklearn is not available
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn is not available. Install with: pip install scikit-learn"
            )

        self.metric = metric
        self._lock = threading.RLock()

        # Storage for embeddings and metadata
        self.embeddings_list = []
        self.subject_ids = []
        self.metadata_list = []

        # NearestNeighbors model (created on first search)
        self.nn_model = None
        self._dirty = True  # Flag to indicate model needs rebuild

        # Map metric names to sklearn metric names
        self.sklearn_metric = self._get_sklearn_metric(metric)

        logger.info(f"Initialized SklearnSearcher: metric={metric}")

    def _get_sklearn_metric(self, metric: str) -> str:
        """Map our metric names to sklearn metric names."""
        metric_map = {
            "cosine": "cosine",
            "euclidean": "euclidean",
            "euclidean_l2": "euclidean"  # Will normalize before search
        }
        if metric not in metric_map:
            raise ValueError(f"Unsupported metric: {metric}")
        return metric_map[metric]

    def _rebuild_model(self):
        """Rebuild NearestNeighbors model from current embeddings."""
        if len(self.embeddings_list) == 0:
            self.nn_model = None
            return

        embeddings = np.array(self.embeddings_list, dtype=np.float32)

        # Normalize for euclidean_l2 metric
        if self.metric == "euclidean_l2":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        # Create and fit NearestNeighbors model
        self.nn_model = NearestNeighbors(
            n_neighbors=min(10, len(self.embeddings_list)),
            metric=self.sklearn_metric,
            algorithm='brute'  # Use brute force for small datasets
        )
        self.nn_model.fit(embeddings)
        self._dirty = False

        logger.debug(f"Rebuilt sklearn NearestNeighbors model with {len(self.embeddings_list)} embeddings")

    def add_embedding(self, embedding: Any, subject_id: str, metadata: Dict[str, Any] = None):
        """
        Add an embedding to the storage.

        Args:
            embedding: Embedding vector (numpy array or PersonEmbedding)
            subject_id: Subject identifier
            metadata: Optional metadata dictionary
        """
        with self._lock:
            # Extract numpy array from PersonEmbedding if needed
            if isinstance(embedding, PersonEmbedding):
                emb_array = embedding.embedding_vector
                if metadata is None:
                    metadata = {"model_profile": embedding.model_profile}
            else:
                emb_array = np.asarray(embedding, dtype=np.float32).flatten()

            # Store
            self.embeddings_list.append(emb_array)
            self.subject_ids.append(subject_id)
            self.metadata_list.append(metadata or {})

            # Mark model as dirty
            self._dirty = True

            logger.debug(f"Added embedding for subject: {subject_id}")

    def search(self, query_embedding: Any, k: int = 10, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using sklearn.

        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            threshold: Optional distance threshold for filtering results

        Returns:
            List of match dictionaries with keys: subject_id, distance, metadata
        """
        with self._lock:
            # Check if we have any embeddings
            if len(self.embeddings_list) == 0:
                logger.warning("Search called on empty index")
                return []

            # Rebuild model if needed
            if self._dirty or self.nn_model is None:
                self._rebuild_model()

            # Extract numpy array
            if isinstance(query_embedding, PersonEmbedding):
                query_array = query_embedding.embedding_vector
            else:
                query_array = np.asarray(query_embedding, dtype=np.float32).flatten()

            # Normalize for euclidean_l2 metric
            if self.metric == "euclidean_l2":
                norm = np.linalg.norm(query_array)
                query_array = query_array / max(norm, 1e-12)

            # Reshape for sklearn
            query_array = query_array.reshape(1, -1)

            # Clamp k
            k = min(k, len(self.embeddings_list))

            # Perform search
            distances, indices = self.nn_model.kneighbors(query_array, n_neighbors=k)

            # Convert to results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                # Apply threshold if provided
                if threshold is not None and dist > threshold:
                    continue

                results.append({
                    "subject_id": self.subject_ids[idx],
                    "distance": float(dist),
                    "metadata": self.metadata_list[idx]
                })

            logger.debug(f"Search returned {len(results)} results (k={k}, threshold={threshold})")
            return results

    def save(self, path: str):
        """
        Save embeddings and metadata to disk.

        Args:
            path: Directory path to save data
        """
        with self._lock:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save embeddings
            embeddings_path = path / "embeddings.npy"
            np.save(embeddings_path, np.array(self.embeddings_list))

            # Save metadata
            metadata_path = path / "metadata.json"
            metadata = {
                "metric": self.metric,
                "total_entries": len(self.subject_ids),
                "subject_ids": self.subject_ids,
                "metadata_list": self.metadata_list
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved sklearn index to {path}")

    def load(self, path: str):
        """
        Load embeddings and metadata from disk.

        Args:
            path: Directory path containing saved data
        """
        with self._lock:
            path = Path(path)

            # Load metadata
            metadata_path = path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Load embeddings
            embeddings_path = path / "embeddings.npy"
            if not embeddings_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

            embeddings = np.load(embeddings_path)
            self.embeddings_list = list(embeddings)

            # Restore metadata
            self.subject_ids = metadata["subject_ids"]
            self.metadata_list = metadata["metadata_list"]

            # Mark model as dirty to trigger rebuild on next search
            self._dirty = True

            logger.info(f"Loaded sklearn index from {path}: {metadata['total_entries']} entries}")


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


class SearcherFactory:
    """
    Factory for creating similarity searchers.

    Automatically selects best available backend (FAISS if available, sklearn otherwise).
    """

    @staticmethod
    def create_searcher(backend: str = "auto", dimension: int = 2048,
                       metric: str = "cosine", device: str = "cpu") -> SimilaritySearcher:
        """
        Create a similarity searcher instance.

        Args:
            backend: Search backend ("faiss", "sklearn", "auto")
            dimension: Embedding dimension
            metric: Distance metric ('cosine', 'euclidean', 'euclidean_l2')
            device: Device to run on ('cpu' or 'cuda')

        Returns:
            SimilaritySearcher instance

        Raises:
            ValueError: If unsupported backend specified
            ImportError: If requested backend is not available
        """
        if backend == "auto":
            # Auto-detect best available backend
            if FAISS_AVAILABLE:
                logger.info("Auto-selected FAISS backend (available)")
                return FAISSSearcher(dimension, metric, device)
            elif SKLEARN_AVAILABLE:
                logger.info("Auto-selected sklearn backend (FAISS not available)")
                return SklearnSearcher(metric)
            else:
                raise ImportError(
                    "No search backend available. Install faiss-cpu or scikit-learn"
                )
        elif backend == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError(
                    "FAISS backend requested but not available. "
                    "Install with: pip install faiss-cpu or faiss-gpu"
                )
            return FAISSSearcher(dimension, metric, device)
        elif backend == "sklearn":
            if not SKLEARN_AVAILABLE:
                raise ImportError(
                    "sklearn backend requested but not available. "
                    "Install with: pip install scikit-learn"
                )
            return SklearnSearcher(metric)
        else:
            raise ValueError(
                f"Unsupported search backend: {backend}. "
                f"Use 'auto', 'faiss', or 'sklearn'"
            )


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