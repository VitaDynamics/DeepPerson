"""
Configuration schema and default values for User Gallery Fusion component.

This module defines the configuration structure for the user gallery fusion
functionality, including fusion weights, clustering parameters, and performance
settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class FusionStrategy(str, Enum):
    """Available fusion strategies for multi-modal retrieval."""
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EQUAL_WEIGHT = "equal_weight"
    FACE_BIASED = "face_biased"
    BODY_BIASED = "body_biased"


class ClusteringAlgorithm(str, Enum):
    """Available clustering algorithms for variant detection."""
    DBSCAN = "dbscan"
    KMEANS = "kmeans"
    HDBSCAN = "hdbscan"
    AGGLOMERATIVE = "agglomerative"


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion scoring."""

    # Fusion strategy
    strategy: FusionStrategy = FusionStrategy.CONFIDENCE_WEIGHTED

    # Default weights (sum must equal 1.0)
    default_face_weight: float = 0.5
    default_body_weight: float = 0.5

    # Confidence-based weighting
    enable_confidence_weighting: bool = True
    min_confidence_threshold: float = 0.1
    max_confidence_threshold: float = 0.9

    # Face-specific parameters
    face_confidence_factor: float = 1.5
    face_quality_factor: float = 1.2

    # Body-specific parameters
    body_confidence_factor: float = 1.0
    body_quality_factor: float = 1.0

    def __post_init__(self):
        """Validate fusion configuration."""
        if not 0.0 <= self.default_face_weight <= 1.0:
            raise ValueError("Default face weight must be between 0.0 and 1.0")
        if not 0.0 <= self.default_body_weight <= 1.0:
            raise ValueError("Default body weight must be between 0.0 and 1.0")
        if abs(self.default_face_weight + self.default_body_weight - 1.0) > 1e-6:
            raise ValueError("Default weights must sum to 1.0")


@dataclass
class ClusteringConfig:
    """Configuration for variant clustering."""

    # Algorithm selection
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.DBSCAN

    # DBSCAN parameters
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_metric: str = "cosine"

    # KMeans parameters
    kmeans_n_clusters: int = 5
    kmeans_random_state: int = 42
    kmeans_max_iter: int = 300

    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: Optional[int] = None
    hdbscan_metric: str = "euclidean"

    # Agglomerative parameters
    agglomerative_n_clusters: int = 5
    agglomerative_linkage: str = "ward"
    agglomerative_distance_threshold: Optional[float] = None

    # General parameters
    similarity_threshold: float = 0.8
    max_images_per_cluster: int = 50
    enable_auto_cluster: bool = True

    def __post_init__(self):
        """Validate clustering configuration."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        if self.max_images_per_cluster < 1:
            raise ValueError("Max images per cluster must be positive")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    # Model configuration
    body_model_name: str = "resnet50_circle_dg"
    face_model_name: str = "Facenet"
    face_detector_backend: str = "opencv"

    # Performance parameters
    batch_size: int = 32
    enable_gpu: bool = True
    force_regenerate: bool = False
    force_download: bool = False

    # Quality control
    min_quality_score: float = 0.1
    max_quality_score: float = 1.0
    quality_threshold: float = 0.5

    # Caching
    cache_embeddings: bool = True
    cache_directory: str = ".cache/embeddings"

    def __post_init__(self):
        """Validate embedding configuration."""
        if not 1 <= self.batch_size <= 128:
            raise ValueError("Batch size must be between 1 and 128")
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("Min quality score must be between 0.0 and 1.0")
        if not 0.0 <= self.max_quality_score <= 1.0:
            raise ValueError("Max quality score must be between 0.0 and 1.0")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")


@dataclass
class SearchConfig:
    """Configuration for similarity search."""

    # Search parameters
    default_top_k: int = 10
    default_min_score: float = 0.0
    max_top_k: int = 100

    # FAISS configuration
    use_gpu: bool = True
    index_type: str = "IVF_SQ8"
    nlist: int = 100
    nprobe: int = 10

    # Distance metrics
    distance_metric: str = "cosine"
    normalize_embeddings: bool = True

    # Performance
    enable_approximate_search: bool = True
    search_timeout: int = 30000  # milliseconds

    def __post_init__(self):
        """Validate search configuration."""
        if self.default_top_k < 1:
            raise ValueError("Default top_k must be positive")
        if not 0.0 <= self.default_min_score <= 1.0:
            raise ValueError("Default min_score must be between 0.0 and 1.0")
        if self.max_top_k < self.default_top_k:
            raise ValueError("Max top_k must be >= default top_k")
        if self.nlist < 1:
            raise ValueError("FAISS nlist must be positive")
        if self.nprobe < 1:
            raise ValueError("FAISS nprobe must be positive")


@dataclass
class UserGalleryConfig:
    """Main configuration for User Gallery Fusion component."""

    # Component enablement
    enabled: bool = True
    debug_mode: bool = False

    # Sub-component configurations
    fusion: FusionConfig = field(default_factory=FusionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    # Gallery management
    max_images_per_gallery: int = 1000
    max_clusters_per_gallery: int = 50
    auto_cleanup_enabled: bool = True
    backup_enabled: bool = True

    # Storage configuration
    gallery_storage_path: str = "galleries/"
    model_cache_path: str = ".cache/models/"
    temp_path: str = "temp/"

    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 8001

    def __post_init__(self):
        """Validate overall configuration."""
        if self.max_images_per_gallery < 1:
            raise ValueError("Max images per gallery must be positive")
        if self.max_clusters_per_gallery < 1:
            raise ValueError("Max clusters per gallery must be positive")
        if not self.gallery_storage_path.endswith('/'):
            raise ValueError("Gallery storage path must end with '/'")
        if not self.model_cache_path.endswith('/'):
            raise ValueError("Model cache path must end with '/'")
        if not self.temp_path.endswith('/'):
            raise ValueError("Temp path must end with '/'")

    @property
    def face_embedding_model(self) -> str:
        """Get face embedding model name."""
        return self.embedding.face_model_name

    @property
    def face_detector_backend(self) -> str:
        """Get face detector backend."""
        return self.embedding.face_detector_backend

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'enabled': self.enabled,
            'debug_mode': self.debug_mode,
            'fusion': {
                'strategy': self.fusion.strategy.value,
                'default_face_weight': self.fusion.default_face_weight,
                'default_body_weight': self.fusion.default_body_weight,
                'enable_confidence_weighting': self.fusion.enable_confidence_weighting,
                'min_confidence_threshold': self.fusion.min_confidence_threshold,
                'max_confidence_threshold': self.fusion.max_confidence_threshold,
                'face_confidence_factor': self.fusion.face_confidence_factor,
                'face_quality_factor': self.fusion.face_quality_factor,
                'body_confidence_factor': self.fusion.body_confidence_factor,
                'body_quality_factor': self.fusion.body_quality_factor,
            },
            'clustering': {
                'algorithm': self.clustering.algorithm.value,
                'dbscan_eps': self.clustering.dbscan_eps,
                'dbscan_min_samples': self.clustering.dbscan_min_samples,
                'dbscan_metric': self.clustering.dbscan_metric,
                'kmeans_n_clusters': self.clustering.kmeans_n_clusters,
                'kmeans_random_state': self.clustering.kmeans_random_state,
                'kmeans_max_iter': self.clustering.kmeans_max_iter,
                'hdbscan_min_cluster_size': self.clustering.hdbscan_min_cluster_size,
                'hdbscan_min_samples': self.clustering.hdbscan_min_samples,
                'hdbscan_metric': self.clustering.hdbscan_metric,
                'agglomerative_n_clusters': self.clustering.agglomerative_n_clusters,
                'agglomerative_linkage': self.clustering.agglomerative_linkage,
                'agglomerative_distance_threshold': self.clustering.agglomerative_distance_threshold,
                'similarity_threshold': self.clustering.similarity_threshold,
                'max_images_per_cluster': self.clustering.max_images_per_cluster,
                'enable_auto_cluster': self.clustering.enable_auto_cluster,
            },
            'embedding': {
                'body_model_name': self.embedding.body_model_name,
                'face_model_name': self.embedding.face_model_name,
                'face_detector_backend': self.embedding.face_detector_backend,
                'batch_size': self.embedding.batch_size,
                'enable_gpu': self.embedding.enable_gpu,
                'force_regenerate': self.embedding.force_regenerate,
                'force_download': self.embedding.force_download,
                'min_quality_score': self.embedding.min_quality_score,
                'max_quality_score': self.embedding.max_quality_score,
                'quality_threshold': self.embedding.quality_threshold,
                'cache_embeddings': self.embedding.cache_embeddings,
                'cache_directory': self.embedding.cache_directory,
            },
            'search': {
                'default_top_k': self.search.default_top_k,
                'default_min_score': self.search.default_min_score,
                'max_top_k': self.search.max_top_k,
                'use_gpu': self.search.use_gpu,
                'index_type': self.search.index_type,
                'nlist': self.search.nlist,
                'nprobe': self.search.nprobe,
                'distance_metric': self.search.distance_metric,
                'normalize_embeddings': self.search.normalize_embeddings,
                'enable_approximate_search': self.search.enable_approximate_search,
                'search_timeout': self.search.search_timeout,
            },
            'max_images_per_gallery': self.max_images_per_gallery,
            'max_clusters_per_gallery': self.max_clusters_per_gallery,
            'auto_cleanup_enabled': self.auto_cleanup_enabled,
            'backup_enabled': self.backup_enabled,
            'gallery_storage_path': self.gallery_storage_path,
            'model_cache_path': self.model_cache_path,
            'temp_path': self.temp_path,
            'log_level': self.log_level,
            'enable_metrics': self.enable_metrics,
            'metrics_port': self.metrics_port,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UserGalleryConfig':
        """Create configuration from dictionary."""
        # Reconstruct sub-configurations
        fusion_config = FusionConfig(
            strategy=FusionStrategy(config_dict['fusion']['strategy']),
            default_face_weight=config_dict['fusion']['default_face_weight'],
            default_body_weight=config_dict['fusion']['default_body_weight'],
            enable_confidence_weighting=config_dict['fusion']['enable_confidence_weighting'],
            min_confidence_threshold=config_dict['fusion']['min_confidence_threshold'],
            max_confidence_threshold=config_dict['fusion']['max_confidence_threshold'],
            face_confidence_factor=config_dict['fusion']['face_confidence_factor'],
            face_quality_factor=config_dict['fusion']['face_quality_factor'],
            body_confidence_factor=config_dict['fusion']['body_confidence_factor'],
            body_quality_factor=config_dict['fusion']['body_quality_factor'],
        )

        clustering_config = ClusteringConfig(
            algorithm=ClusteringAlgorithm(config_dict['clustering']['algorithm']),
            dbscan_eps=config_dict['clustering']['dbscan_eps'],
            dbscan_min_samples=config_dict['clustering']['dbscan_min_samples'],
            dbscan_metric=config_dict['clustering']['dbscan_metric'],
            kmeans_n_clusters=config_dict['clustering']['kmeans_n_clusters'],
            kmeans_random_state=config_dict['clustering']['kmeans_random_state'],
            kmeans_max_iter=config_dict['clustering']['kmeans_max_iter'],
            hdbscan_min_cluster_size=config_dict['clustering']['hdbscan_min_cluster_size'],
            hdbscan_min_samples=config_dict['clustering']['hdbscan_min_samples'],
            hdbscan_metric=config_dict['clustering']['hdbscan_metric'],
            agglomerative_n_clusters=config_dict['clustering']['agglomerative_n_clusters'],
            agglomerative_linkage=config_dict['clustering']['agglomerative_linkage'],
            agglomerative_distance_threshold=config_dict['clustering']['agglomerative_distance_threshold'],
            similarity_threshold=config_dict['clustering']['similarity_threshold'],
            max_images_per_cluster=config_dict['clustering']['max_images_per_cluster'],
            enable_auto_cluster=config_dict['clustering']['enable_auto_cluster'],
        )

        embedding_config = EmbeddingConfig(
            body_model_name=config_dict['embedding']['body_model_name'],
            face_model_name=config_dict['embedding']['face_model_name'],
            face_detector_backend=config_dict['embedding']['face_detector_backend'],
            batch_size=config_dict['embedding']['batch_size'],
            enable_gpu=config_dict['embedding']['enable_gpu'],
            force_regenerate=config_dict['embedding']['force_regenerate'],
            force_download=config_dict['embedding']['force_download'],
            min_quality_score=config_dict['embedding']['min_quality_score'],
            max_quality_score=config_dict['embedding']['max_quality_score'],
            quality_threshold=config_dict['embedding']['quality_threshold'],
            cache_embeddings=config_dict['embedding']['cache_embeddings'],
            cache_directory=config_dict['embedding']['cache_directory'],
        )

        search_config = SearchConfig(
            default_top_k=config_dict['search']['default_top_k'],
            default_min_score=config_dict['search']['default_min_score'],
            max_top_k=config_dict['search']['max_top_k'],
            use_gpu=config_dict['search']['use_gpu'],
            index_type=config_dict['search']['index_type'],
            nlist=config_dict['search']['nlist'],
            nprobe=config_dict['search']['nprobe'],
            distance_metric=config_dict['search']['distance_metric'],
            normalize_embeddings=config_dict['search']['normalize_embeddings'],
            enable_approximate_search=config_dict['search']['enable_approximate_search'],
            search_timeout=config_dict['search']['search_timeout'],
        )

        return cls(
            enabled=config_dict.get('enabled', True),
            debug_mode=config_dict.get('debug_mode', False),
            fusion=fusion_config,
            clustering=clustering_config,
            embedding=embedding_config,
            search=search_config,
            max_images_per_gallery=config_dict.get('max_images_per_gallery', 1000),
            max_clusters_per_gallery=config_dict.get('max_clusters_per_gallery', 50),
            auto_cleanup_enabled=config_dict.get('auto_cleanup_enabled', True),
            backup_enabled=config_dict.get('backup_enabled', True),
            gallery_storage_path=config_dict.get('gallery_storage_path', 'galleries/'),
            model_cache_path=config_dict.get('model_cache_path', '.cache/models/'),
            temp_path=config_dict.get('temp_path', 'temp/'),
            log_level=config_dict.get('log_level', 'INFO'),
            enable_metrics=config_dict.get('enable_metrics', True),
            metrics_port=config_dict.get('metrics_port', 8001),
        )

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'UserGalleryConfig':
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Default configuration instance
DEFAULT_CONFIG = UserGalleryConfig()


def get_default_config() -> UserGalleryConfig:
    """Get the default configuration for User Gallery Fusion."""
    return DEFAULT_CONFIG


def create_config_from_env() -> UserGalleryConfig:
    """Create configuration from environment variables."""
    import os

    # This would be expanded to read from environment variables
    # For now, return default configuration
    return get_default_config()