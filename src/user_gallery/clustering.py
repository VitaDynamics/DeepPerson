"""
Automatic variant clustering for user gallery images.

This module provides clustering algorithms to automatically group images
into appearance variants (e.g., different outfits, time periods) based on
embedding similarity.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import numpy as np

from .config import ClusteringAlgorithm, ClusteringConfig
from .models import EmbeddingSet, ImageAsset, VariantCluster

logger = logging.getLogger(__name__)


class VariantClusterer:
    """
    Clusters images into appearance variants using embedding similarity.

    Supports multiple clustering algorithms (DBSCAN, KMeans, HDBSCAN, Agglomerative)
    with configurable parameters for different use cases.
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize variant clusterer.

        Args:
            config: Optional clustering configuration
        """
        self.config = config if config else ClusteringConfig()
        logger.info(
            f"Initialized VariantClusterer with algorithm: {self.config.algorithm.value}"
        )

    def cluster_images(
        self,
        embeddings: list[EmbeddingSet],
        images: list[ImageAsset],
        user_id: str,
    ) -> list[VariantCluster]:
        """
        Cluster images into appearance variants.

        Args:
            embeddings: List of embedding sets for the images
            images: List of image assets
            user_id: User identifier for the clusters

        Returns:
            List of VariantCluster objects

        Raises:
            ValueError: If embeddings and images don't match or are empty
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings provided for clustering")
            return []

        if len(embeddings) != len(images):
            raise ValueError(
                f"Embeddings count ({len(embeddings)}) must match images count ({len(images)})"
            )

        # Extract body embeddings for clustering
        embedding_matrix = np.array([emb.body_embedding for emb in embeddings])

        # Perform clustering based on configured algorithm
        if self.config.algorithm == ClusteringAlgorithm.DBSCAN:
            cluster_labels = self._cluster_dbscan(embedding_matrix)
        elif self.config.algorithm == ClusteringAlgorithm.KMEANS:
            cluster_labels = self._cluster_kmeans(embedding_matrix)
        elif self.config.algorithm == ClusteringAlgorithm.HDBSCAN:
            cluster_labels = self._cluster_hdbscan(embedding_matrix)
        elif self.config.algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            cluster_labels = self._cluster_agglomerative(embedding_matrix)
        else:
            raise ValueError(
                f"Unsupported clustering algorithm: {self.config.algorithm}"
            )

        # Create VariantCluster objects
        clusters = self._create_clusters_from_labels(
            cluster_labels, images, embeddings, user_id
        )

        logger.info(
            f"Clustered {len(images)} images into {len(clusters)} variant clusters"
        )
        return clusters

    def _cluster_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster using DBSCAN algorithm.

        Args:
            embeddings: Embedding matrix (shape: [n_samples, n_features])

        Returns:
            Cluster labels array
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            raise ImportError(
                "scikit-learn is required for DBSCAN clustering. "
                "Install with: pip install scikit-learn"
            )

        # Normalize embeddings for cosine distance
        if self.config.dbscan_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)

        clusterer = DBSCAN(
            eps=self.config.dbscan_eps,
            min_samples=self.config.dbscan_min_samples,
            metric=self.config.dbscan_metric,
        )

        labels = clusterer.fit_predict(embeddings)
        logger.debug(
            f"DBSCAN clustering: {len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
            f"{list(labels).count(-1)} noise points"
        )

        return labels

    def _cluster_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster using KMeans algorithm.

        Args:
            embeddings: Embedding matrix (shape: [n_samples, n_features])

        Returns:
            Cluster labels array
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for KMeans clustering. "
                "Install with: pip install scikit-learn"
            )

        # Determine number of clusters (min of configured value and number of samples)
        n_clusters = min(self.config.kmeans_n_clusters, len(embeddings))

        if n_clusters < 2:
            # If only one sample or cluster, assign all to cluster 0
            return np.zeros(len(embeddings), dtype=int)

        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=self.config.kmeans_random_state,
            max_iter=self.config.kmeans_max_iter,
            n_init=10,
        )

        labels = clusterer.fit_predict(embeddings)
        logger.debug(f"KMeans clustering: {n_clusters} clusters")

        return labels

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster using HDBSCAN algorithm.

        Args:
            embeddings: Embedding matrix (shape: [n_samples, n_features])

        Returns:
            Cluster labels array
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError(
                "hdbscan is required for HDBSCAN clustering. "
                "Install with: pip install hdbscan"
            )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            metric=self.config.hdbscan_metric,
        )

        labels = clusterer.fit_predict(embeddings)
        logger.debug(
            f"HDBSCAN clustering: {len(set(labels)) - (1 if -1 in labels else 0)} clusters, "
            f"{list(labels).count(-1)} noise points"
        )

        return labels

    def _cluster_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster using Agglomerative Hierarchical clustering.

        Args:
            embeddings: Embedding matrix (shape: [n_samples, n_features])

        Returns:
            Cluster labels array
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            raise ImportError(
                "scikit-learn is required for Agglomerative clustering. "
                "Install with: pip install scikit-learn"
            )

        # Determine number of clusters
        if self.config.agglomerative_distance_threshold is not None:
            # Use distance threshold (n_clusters must be None)
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.config.agglomerative_distance_threshold,
                linkage=self.config.agglomerative_linkage,
            )
        else:
            # Use fixed number of clusters
            n_clusters = min(self.config.agglomerative_n_clusters, len(embeddings))
            if n_clusters < 2:
                return np.zeros(len(embeddings), dtype=int)

            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=self.config.agglomerative_linkage
            )

        labels = clusterer.fit_predict(embeddings)
        logger.debug(f"Agglomerative clustering: {len(set(labels))} clusters")

        return labels

    def _create_clusters_from_labels(
        self,
        labels: np.ndarray,
        images: list[ImageAsset],
        embeddings: list[EmbeddingSet],
        user_id: str,
    ) -> list[VariantCluster]:
        """
        Create VariantCluster objects from cluster labels.

        Args:
            labels: Cluster label array
            images: List of image assets
            embeddings: List of embedding sets
            user_id: User identifier

        Returns:
            List of VariantCluster objects
        """
        clusters_dict: dict[int, list[tuple[ImageAsset, EmbeddingSet]]] = {}

        # Group images by cluster label
        for label, image, embedding in zip(labels, images, embeddings):
            # Handle noise points (label -1) by creating individual clusters
            if label == -1:
                label = len(clusters_dict) + 1000  # Offset to avoid conflicts

            if label not in clusters_dict:
                clusters_dict[label] = []

            clusters_dict[label].append((image, embedding))

        # Create VariantCluster objects
        variant_clusters = []

        for cluster_id, items in clusters_dict.items():
            # Generate unique cluster ID
            cluster_uuid = str(uuid.uuid4())[:8]
            cluster_name = f"variant_{cluster_id}_{cluster_uuid}"

            # Select primary image (first image in cluster)
            primary_image_id = items[0][0].image_id

            # Update image cluster assignments
            for image, _embedding in items:
                image.cluster_id = cluster_name

            # Create cluster
            cluster = VariantCluster(
                cluster_id=cluster_name,
                user_id=user_id,
                cluster_name=f"Variant {cluster_id}",
                description=f"Automatically clustered variant with {len(items)} images",
                created_at=datetime.now(),
                image_count=len(items),
                primary_image_id=primary_image_id,
                metadata={
                    "algorithm": self.config.algorithm.value,
                    "auto_generated": True,
                    "cluster_label": int(cluster_id),
                },
            )

            variant_clusters.append(cluster)

        # Sort clusters by size (largest first)
        variant_clusters.sort(key=lambda c: c.image_count, reverse=True)

        return variant_clusters

    def merge_clusters(
        self,
        clusters: list[VariantCluster],
        similarity_threshold: float = 0.8,
    ) -> list[VariantCluster]:
        """
        Merge similar clusters based on similarity threshold.

        Args:
            clusters: List of variant clusters to potentially merge
            similarity_threshold: Minimum similarity to merge clusters

        Returns:
            List of merged variant clusters

        Note:
            This is a placeholder for future implementation.
            Currently returns clusters unchanged.
        """
        # TODO: Implement cluster merging based on centroid similarity
        logger.debug(
            f"Merge clusters called with {len(clusters)} clusters "
            f"(threshold: {similarity_threshold})"
        )
        return clusters

    def refine_clusters(
        self,
        clusters: list[VariantCluster],
        embeddings: list[EmbeddingSet],
        images: list[ImageAsset],
    ) -> list[VariantCluster]:
        """
        Refine cluster assignments by reassigning outliers.

        Args:
            clusters: Current variant clusters
            embeddings: List of embedding sets
            images: List of image assets

        Returns:
            Refined list of variant clusters

        Note:
            This is a placeholder for future implementation.
            Currently returns clusters unchanged.
        """
        # TODO: Implement cluster refinement
        logger.debug(f"Refine clusters called with {len(clusters)} clusters")
        return clusters


def create_single_cluster(
    images: list[ImageAsset], user_id: str, cluster_name: str = "default"
) -> VariantCluster:
    """
    Create a single cluster containing all images.

    Useful when clustering is disabled or when all images should be in one group.

    Args:
        images: List of image assets
        user_id: User identifier
        cluster_name: Name for the cluster

    Returns:
        VariantCluster containing all images
    """
    if len(images) == 0:
        raise ValueError("Cannot create cluster from empty image list")

    # Generate unique cluster ID
    cluster_uuid = str(uuid.uuid4())[:8]
    cluster_id = f"{cluster_name}_{cluster_uuid}"

    # Update image cluster assignments
    for image in images:
        image.cluster_id = cluster_id

    # Create cluster
    cluster = VariantCluster(
        cluster_id=cluster_id,
        user_id=user_id,
        cluster_name=cluster_name.replace("_", " ").title(),
        description=f"Single cluster containing all {len(images)} images",
        created_at=datetime.now(),
        image_count=len(images),
        primary_image_id=images[0].image_id,
        metadata={"auto_generated": False, "single_cluster": True},
    )

    logger.debug(f"Created single cluster '{cluster_id}' with {len(images)} images")
    return cluster
