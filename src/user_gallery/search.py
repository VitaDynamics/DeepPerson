"""
Multi-modal similarity search for user gallery fusion.

This module provides FAISS-based multi-modal search capabilities for user galleries,
supporting separate indexes for body and face embeddings with fusion scoring.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

# Import from parent module
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from search import SearcherFactory  # noqa: E402

from .config import FusionConfig, SearchConfig  # noqa: E402
from .models import ConfidenceLevel, RetrievalProbe, RetrievalResult  # noqa: E402

logger = logging.getLogger(__name__)


class MultiModalSearcher:
    """
    Multi-modal similarity searcher for user galleries.

    Manages separate FAISS indexes for body and face embeddings, performs
    independent searches, and combines results using fusion scoring.
    """

    def __init__(
        self,
        body_dimension: int = 2048,
        face_dimension: int | None = 512,
        metric: str = "cosine",
        device: str = "cpu",
        search_config: SearchConfig | None = None,
        fusion_config: FusionConfig | None = None,
        backend: str = "auto",
    ):
        """
        Initialize multi-modal searcher with separate indexes.

        Args:
            body_dimension: Dimension of body embeddings
            face_dimension: Dimension of face embeddings (None if not using faces)
            metric: Distance metric ('cosine', 'euclidean', 'euclidean_l2')
            device: Device to run on ('cpu' or 'cuda')
            search_config: Optional search configuration
            fusion_config: Optional fusion configuration
            backend: Search backend ("auto", "faiss", "sklearn")

        Raises:
            ImportError: If no search backend is available
        """
        self.body_dimension = body_dimension
        self.face_dimension = face_dimension
        self.metric = metric
        self.device = device
        self.backend = backend
        self._lock = threading.RLock()

        # Configuration
        self.search_config = search_config if search_config else SearchConfig()
        self.fusion_config = fusion_config if fusion_config else FusionConfig()

        # Create separate search indexes using SearcherFactory
        self.body_index = SearcherFactory.create_searcher(
            backend=backend, dimension=body_dimension, metric=metric, device=device
        )

        self.face_index = None
        if face_dimension is not None:
            self.face_index = SearcherFactory.create_searcher(
                backend=backend, dimension=face_dimension, metric=metric, device=device
            )

        # User-level metadata mapping
        self.user_metadata: dict[str, dict[str, Any]] = {}

        logger.info(
            f"Initialized MultiModalSearcher: body_dim={body_dimension}, "
            f"face_dim={face_dimension}, metric={metric}, device={device}, backend={backend}"
        )

    def add_user_gallery(
        self,
        user_id: str,
        body_embeddings: np.ndarray,
        face_embeddings: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a user gallery with body and optional face embeddings.

        Args:
            user_id: Unique user identifier
            body_embeddings: Body embedding matrix (shape: [n_images, body_dim])
            face_embeddings: Optional face embedding matrix (shape: [n_images, face_dim])
            metadata: Optional user metadata

        Raises:
            ValueError: If dimensions mismatch or invalid inputs
        """
        with self._lock:
            if body_embeddings is None or len(body_embeddings) == 0:
                raise ValueError(f"Body embeddings must be provided for user {user_id}")

            # Validate body embeddings
            if body_embeddings.ndim == 1:
                body_embeddings = body_embeddings.reshape(1, -1)
            if body_embeddings.shape[1] != self.body_dimension:
                raise ValueError(
                    f"Body embedding dimension mismatch: expected {self.body_dimension}, "
                    f"got {body_embeddings.shape[1]}"
                )

            # Add body embeddings to body index using batch operation if available
            n_body_images = body_embeddings.shape[0]
            body_subject_ids = [user_id] * n_body_images
            body_metadata = [
                {"modality": "body", "embedding_index": i} for i in range(n_body_images)
            ]
            
            # Use add_batch if available (FAISS), otherwise fall back to individual adds (sklearn)
            if hasattr(self.body_index, 'add_batch'):
                self.body_index.add_batch(body_embeddings, body_subject_ids, body_metadata)
            else:
                for i, body_emb in enumerate(body_embeddings):
                    self.body_index.add_embedding(
                        body_emb,
                        subject_id=user_id,
                        metadata=body_metadata[i],
                    )

            # Add face embeddings if provided
            if face_embeddings is not None and self.face_index is not None:
                if face_embeddings.ndim == 1:
                    face_embeddings = face_embeddings.reshape(1, -1)

                if face_embeddings.shape[1] != self.face_dimension:
                    raise ValueError(
                        f"Face embedding dimension mismatch: expected {self.face_dimension}, "
                        f"got {face_embeddings.shape[1]}"
                    )

                n_face_images = face_embeddings.shape[0]
                face_subject_ids = [user_id] * n_face_images
                face_metadata = [
                    {"modality": "face", "embedding_index": i} for i in range(n_face_images)
                ]
                
                # Use add_batch if available (FAISS), otherwise fall back to individual adds (sklearn)
                if hasattr(self.face_index, 'add_batch'):
                    self.face_index.add_batch(face_embeddings, face_subject_ids, face_metadata)
                else:
                    for i, face_emb in enumerate(face_embeddings):
                        self.face_index.add_embedding(
                            face_emb,
                            subject_id=user_id,
                            metadata=face_metadata[i],
                        )

            # Store user metadata
            self.user_metadata[user_id] = metadata or {}
            self.user_metadata[user_id].update(
                {
                    "n_body_embeddings": n_body_images,
                    "n_face_embeddings": len(face_embeddings)
                    if face_embeddings is not None
                    else 0,
                }
            )

            logger.debug(
                f"Added user gallery '{user_id}': {n_body_images} body, "
                f"{len(face_embeddings) if face_embeddings is not None else 0} face embeddings"
            )

    def search_multi_modal(
        self,
        probe: RetrievalProbe,
        k: int = 10,
        min_score: float = 0.0,
        fusion_weights: dict[str, float] | None = None,
    ) -> list[RetrievalResult]:
        """
        Perform multi-modal search with fusion scoring.

        Args:
            probe: RetrievalProbe containing query embeddings
            k: Number of top results to return
            min_score: Minimum fusion score threshold
            fusion_weights: Optional custom fusion weights

        Returns:
            List of RetrievalResult objects sorted by fusion score

        Raises:
            ValueError: If probe doesn't have required embeddings
        """
        with self._lock:
            # Validate probe has body embedding at minimum
            if not probe.has_body_embedding:
                raise ValueError("Probe must have body embedding")

            # Determine fusion weights
            if fusion_weights is None:
                fusion_weights = {
                    "face": self.fusion_config.default_face_weight,
                    "body": self.fusion_config.default_body_weight,
                }

            # Search body index
            body_query = probe.generated_embeddings.get("body")
            body_results = self.body_index.search(
                body_query,
                k=k * 2,  # Get more results for fusion
                threshold=None,
            )

            # Search face index if available
            face_results = []
            if probe.has_face_embedding and self.face_index is not None:
                face_query = probe.generated_embeddings.get("face")
                face_results = self.face_index.search(
                    face_query, k=k * 2, threshold=None
                )

            # Aggregate results by user and compute fusion scores
            user_scores = self._compute_fusion_scores(
                body_results, face_results, fusion_weights
            )

            # Convert to RetrievalResult objects
            results = []
            for user_id, score_info in sorted(
                user_scores.items(), key=lambda x: x[1]["overall_score"], reverse=True
            )[:k]:
                if score_info["overall_score"] < min_score:
                    continue

                # Determine confidence level
                confidence = self._compute_confidence_level(score_info["overall_score"])

                result = RetrievalResult(
                    result_id=f"{probe.probe_id}_{user_id}",
                    probe_id=probe.probe_id,
                    user_id=user_id,
                    overall_score=score_info["overall_score"],
                    face_score=score_info.get("face_score"),
                    body_score=score_info.get("body_score"),
                    face_weight=fusion_weights["face"],
                    body_weight=fusion_weights["body"],
                    evidence_images=score_info.get("evidence_images", []),
                    confidence_level=confidence,
                    metadata=self.user_metadata.get(user_id, {}),
                )
                results.append(result)

            logger.info(
                f"Multi-modal search completed: {len(results)} results for probe '{probe.probe_id}'"
            )
            return results

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert distance to similarity score based on the current metric.
        
        Args:
            distance: Distance value from search
            
        Returns:
            Similarity score in range [0, 1] where higher is better
        """
        if self.metric == "cosine":
            # For cosine distance: similarity = 1 - distance
            return max(0.0, 1.0 - distance)
        elif self.metric in ["euclidean", "euclidean_l2"]:
            # For euclidean distances, use a simple inverse relationship
            # This is a simplified conversion - could be made more sophisticated
            return max(0.0, 1.0 / (1.0 + distance))
        else:
            # Fallback: assume lower distance is better
            return max(0.0, 1.0 - distance)

    def _compute_fusion_scores(
        self,
        body_results: list[dict[str, Any]],
        face_results: list[dict[str, Any]],
        fusion_weights: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        """
        Compute fusion scores by aggregating body and face search results.

        Args:
            body_results: Results from body index search
            face_results: Results from face index search
            fusion_weights: Fusion weights for body and face modalities

        Returns:
            Dictionary mapping user_id to score information
        """
        user_scores: dict[str, dict[str, Any]] = {}

        # Process body results
        for result in body_results:
            user_id = result["subject_id"]
            # Convert distance to similarity score based on metric
            body_similarity = self._distance_to_similarity(result["distance"])

            if user_id not in user_scores:
                user_scores[user_id] = {
                    "body_score": body_similarity,
                    "face_score": None,
                    "evidence_images": [],
                }
            else:
                # Take maximum similarity for multiple images
                user_scores[user_id]["body_score"] = max(
                    user_scores[user_id]["body_score"], body_similarity
                )

        # Process face results
        for result in face_results:
            user_id = result["subject_id"]
            # Convert distance to similarity score based on metric
            face_similarity = self._distance_to_similarity(result["distance"])

            if user_id not in user_scores:
                user_scores[user_id] = {
                    "body_score": None,
                    "face_score": face_similarity,
                    "evidence_images": [],
                }
            else:
                # Take maximum similarity for multiple images
                if user_scores[user_id]["face_score"] is None:
                    user_scores[user_id]["face_score"] = face_similarity
                else:
                    user_scores[user_id]["face_score"] = max(
                        user_scores[user_id]["face_score"], face_similarity
                    )

        # Compute overall fusion scores
        for user_id, scores in user_scores.items():
            body_score = (
                scores["body_score"] if scores["body_score"] is not None else 0.0
            )
            face_score = (
                scores["face_score"] if scores["face_score"] is not None else 0.0
            )

            # Compute weighted fusion score
            if face_score > 0 and body_score > 0:
                # Both modalities available
                overall_score = (
                    fusion_weights["body"] * body_score
                    + fusion_weights["face"] * face_score
                )
            elif body_score > 0:
                # Only body available
                overall_score = body_score
            elif face_score > 0:
                # Only face available
                overall_score = face_score
            else:
                overall_score = 0.0

            scores["overall_score"] = overall_score

        return user_scores

    def _compute_confidence_level(self, score: float) -> ConfidenceLevel:
        """
        Determine confidence level based on fusion score.

        Args:
            score: Fusion score (0.0 - 1.0)

        Returns:
            ConfidenceLevel enum value
        """
        if score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def save(self, output_dir: Path, gallery_name: str = "user_gallery") -> Path:
        """
        Save multi-modal indexes to disk.

        Creates:
            {output_dir}/{gallery_name}/
                ├── body_index/          # Body FAISS index
                ├── face_index/          # Face FAISS index (if available)
                ├── user_metadata.json   # User metadata
                └── config.json          # Searcher configuration

        Args:
            output_dir: Base directory for saving
            gallery_name: Name for this gallery

        Returns:
            Path to the gallery directory

        Raises:
            OSError: If save fails
        """
        with self._lock:
            output_dir = Path(output_dir)
            gallery_dir = output_dir / gallery_name
            gallery_dir.mkdir(parents=True, exist_ok=True)

            # Save body index
            body_index_dir = gallery_dir / "body_index"
            self.body_index.save(str(body_index_dir))
            logger.debug(f"Saved body index to {body_index_dir}")

            # Save face index if available
            if self.face_index is not None:
                face_index_dir = gallery_dir / "face_index"
                self.face_index.save(str(face_index_dir))
                logger.debug(f"Saved face index to {face_index_dir}")

            # Save user metadata
            metadata_path = gallery_dir / "user_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(self.user_metadata, f, indent=2, default=str)

            # Save configuration
            config_path = gallery_dir / "config.json"
            config = {
                "body_dimension": self.body_dimension,
                "face_dimension": self.face_dimension,
                "metric": self.metric,
                "device": self.device,
                "backend": self.backend,
                "has_face_index": self.face_index is not None,
                "total_users": len(self.user_metadata),
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Saved multi-modal gallery '{gallery_name}' to {gallery_dir}")
            return gallery_dir

    def load(self, gallery_dir: Path) -> None:
        """
        Load multi-modal indexes from disk.

        Args:
            gallery_dir: Directory containing saved gallery

        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If configuration mismatch
        """
        with self._lock:
            gallery_dir = Path(gallery_dir)

            if not gallery_dir.exists():
                raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

            # Load configuration
            config_path = gallery_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                config = json.load(f)

            # Validate configuration
            if config["body_dimension"] != self.body_dimension:
                raise ValueError(
                    f"Body dimension mismatch: expected {self.body_dimension}, "
                    f"got {config['body_dimension']}"
                )

            if config["face_dimension"] != self.face_dimension:
                logger.warning(
                    f"Face dimension mismatch: expected {self.face_dimension}, "
                    f"got {config['face_dimension']}"
                )

            # Load body index
            body_index_dir = gallery_dir / "body_index"
            if not body_index_dir.exists():
                raise FileNotFoundError(f"Body index not found: {body_index_dir}")
            self.body_index.load(str(body_index_dir))
            logger.debug(f"Loaded body index from {body_index_dir}")

            # Load face index if available
            if config.get("has_face_index", False):
                face_index_dir = gallery_dir / "face_index"
                if face_index_dir.exists() and self.face_index is not None:
                    self.face_index.load(str(face_index_dir))
                    logger.debug(f"Loaded face index from {face_index_dir}")

            # Load user metadata
            metadata_path = gallery_dir / "user_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    self.user_metadata = json.load(f)

            logger.info(
                f"Loaded multi-modal gallery from {gallery_dir}: {len(self.user_metadata)} users"
            )

    def get_user_count(self) -> int:
        """Get the number of users in the gallery."""
        return len(self.user_metadata)

    def get_gallery_stats(self) -> dict[str, Any]:
        """
        Get statistics about the gallery.

        Returns:
            Dictionary with gallery statistics
        """
        with self._lock:
            total_body = sum(
                meta.get("n_body_embeddings", 0) for meta in self.user_metadata.values()
            )
            total_face = sum(
                meta.get("n_face_embeddings", 0) for meta in self.user_metadata.values()
            )

            return {
                "total_users": len(self.user_metadata),
                "total_body_embeddings": total_body,
                "total_face_embeddings": total_face,
                "body_dimension": self.body_dimension,
                "face_dimension": self.face_dimension,
                "metric": self.metric,
                "device": self.device,
                "has_face_index": self.face_index is not None,
            }

    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user from the gallery.

        Note: This requires rebuilding the indexes, which is expensive.
        For production use, consider marking users as inactive instead.

        Args:
            user_id: User identifier to remove

        Returns:
            True if user was removed, False if not found
        """
        with self._lock:
            if user_id not in self.user_metadata:
                logger.warning(f"User '{user_id}' not found in gallery")
                return False

            # Remove from metadata
            del self.user_metadata[user_id]

            # Note: Removing from FAISS index requires rebuilding
            # This is a limitation of FAISS - we'd need to rebuild the index
            # For now, just log a warning
            logger.warning(
                f"User '{user_id}' metadata removed, but FAISS index not updated. "
                f"Consider rebuilding the index for optimal performance."
            )

            return True


class MultiModalIndexManager:
    """
    Manager for multiple user gallery indexes.

    Provides high-level interface for managing multiple galleries,
    switching between them, and performing searches across galleries.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize index manager.

        Args:
            base_dir: Base directory for storing gallery indexes
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._galleries: dict[str, MultiModalSearcher] = {}
        self._lock = threading.RLock()

        logger.info(f"Initialized MultiModalIndexManager at {base_dir}")

    def create_gallery(
        self,
        gallery_name: str,
        body_dimension: int = 2048,
        face_dimension: int | None = 512,
        metric: str = "cosine",
        device: str = "cpu",
        backend: str = "auto",
    ) -> MultiModalSearcher:
        """
        Create a new gallery index.

        Args:
            gallery_name: Unique name for the gallery
            body_dimension: Body embedding dimension
            face_dimension: Face embedding dimension
            metric: Distance metric
            device: Compute device
            backend: Search backend ("auto", "faiss", "sklearn")

        Returns:
            MultiModalSearcher instance

        Raises:
            ValueError: If gallery already exists
        """
        with self._lock:
            if gallery_name in self._galleries:
                raise ValueError(f"Gallery '{gallery_name}' already exists")

            searcher = MultiModalSearcher(
                body_dimension=body_dimension,
                face_dimension=face_dimension,
                metric=metric,
                device=device,
                backend=backend,
            )

            self._galleries[gallery_name] = searcher
            logger.info(f"Created new gallery '{gallery_name}' with backend '{backend}'")
            return searcher

    def load_gallery(self, gallery_name: str) -> MultiModalSearcher:
        """
        Load an existing gallery from disk.

        Args:
            gallery_name: Name of the gallery to load

        Returns:
            Loaded MultiModalSearcher instance

        Raises:
            FileNotFoundError: If gallery doesn't exist
        """
        with self._lock:
            gallery_dir = self.base_dir / gallery_name

            # Load configuration to create searcher
            config_path = gallery_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Gallery '{gallery_name}' not found")

            with open(config_path) as f:
                config = json.load(f)

            # Create searcher with same configuration
            searcher = MultiModalSearcher(
                body_dimension=config["body_dimension"],
                face_dimension=config.get("face_dimension"),
                metric=config["metric"],
                device=config["device"],
                backend=config.get("backend", "auto"),
            )

            # Load data
            searcher.load(gallery_dir)

            self._galleries[gallery_name] = searcher
            logger.info(f"Loaded gallery '{gallery_name}' from {gallery_dir}")
            return searcher

    def save_gallery(self, gallery_name: str) -> Path:
        """
        Save a gallery to disk.

        Args:
            gallery_name: Name of the gallery to save

        Returns:
            Path to saved gallery directory

        Raises:
            ValueError: If gallery doesn't exist
        """
        with self._lock:
            if gallery_name not in self._galleries:
                raise ValueError(f"Gallery '{gallery_name}' not found")

            searcher = self._galleries[gallery_name]
            gallery_path = searcher.save(self.base_dir, gallery_name)
            logger.info(f"Saved gallery '{gallery_name}' to {gallery_path}")
            return gallery_path

    def get_gallery(self, gallery_name: str) -> MultiModalSearcher | None:
        """Get a gallery by name."""
        return self._galleries.get(gallery_name)

    def list_galleries(self) -> list[str]:
        """List all loaded galleries."""
        return list(self._galleries.keys())

    def list_saved_galleries(self) -> list[str]:
        """List all galleries saved in the base directory."""
        galleries = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                galleries.append(item.name)
        return galleries
