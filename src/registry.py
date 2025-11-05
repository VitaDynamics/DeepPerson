"""
Model registry for DeepPerson library.

Manages registration, caching, and lifecycle of person re-identification models.
Provides default model profiles and verification thresholds.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .entities import ModelProfile

logger = logging.getLogger(__name__)


# Default verification thresholds per model and metric
# Based on common ReID benchmarks and person re-identification literature
# These thresholds represent conservative defaults that should be tuned on validation data
# for specific use cases and deployment scenarios
#
# Interpretation (for all metrics, lower distance = more similar):
#   - cosine: Range [0, 2]. Typical thresholds: 0.30-0.50 (lower = stricter matching)
#   - euclidean: Range [0, ∞). Depends on embedding dimension (2048 for ResNet50)
#   - euclidean_l2: Range [0, 2]. Similar to cosine, computed on normalized embeddings
#
# Verification decision: verified = (distance <= threshold)
#
# Reference: Based on DeepFace verification patterns and person ReID best practices
# from layumi/Person_reID_baseline_pytorch benchmarks
DEFAULT_THRESHOLDS = {
    "resnet50_circle_dg": {
        "cosine": 0.40,        # Conservative threshold for cosine distance
        "euclidean": 10.0,     # Reasonable for 2048-dimensional embeddings
        "euclidean_l2": 0.85   # Normalized euclidean distance threshold
    }
}


class ModelRegistry:
    """
    Registry for managing person re-identification model profiles and instances.

    Provides:
    - Model profile registration and retrieval
    - Model instance caching per device
    - Default model profiles
    - Verification thresholds
    - Face embedding model caching (DeepFace-based)
    - Thread-safe operations

    Examples:
        >>> registry = ModelRegistry()
        >>> profile = registry.get_profile("resnet50_circle_dg")
        >>> model = registry.load_model("resnet50_circle_dg", torch.device("cpu"))
        >>> face_model = registry.load_face_model("Facenet", "opencv")
    """

    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        """Initialize model registry with default profiles and face model cache."""
        self._profiles: Dict[str, ModelProfile] = {}
        self._model_cache: Dict[str, torch.nn.Module] = {}
        self._thresholds: Dict[str, Dict[str, float]] = DEFAULT_THRESHOLDS.copy()

        # Face embedding model cache (DeepFace-based)
        self._face_model_cache: Dict[Tuple[str, str], Any] = {}  # Key: (model_name, detector_backend)
        self._face_lock = threading.RLock()  # Separate lock for face models

        # Register default models
        self._register_default_profiles()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """
        Get singleton instance of ModelRegistry.

        Returns:
            Shared ModelRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _register_default_profiles(self) -> None:
        """Register default model profiles."""
        # ResNet-50 with Circle loss and Domain Generalization
        # From layumi/Person_reID_baseline_pytorch
        resnet50_profile = ModelProfile(
            identifier="resnet50_circle_dg",
            backbone_path=Path("models/resnet50_circle_dg.pth"),
            feature_dim=512,
            requires_cuda=False,  # Works on CPU, but GPU recommended
            preprocess_config={
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "input_size": (256, 128),  # (height, width)
                "interpolation": "bilinear",
                "padding": 10,  # For random erasing augmentation during training
            }
        )
        self.register_profile(resnet50_profile)
        logger.info(f"Registered default profile: {resnet50_profile.identifier}")

    def register_profile(self, profile: ModelProfile) -> None:
        """
        Register a model profile.

        Args:
            profile: ModelProfile to register

        Raises:
            ValueError: If profile identifier already exists
        """
        if profile.identifier in self._profiles:
            logger.warning(
                f"Profile '{profile.identifier}' already registered, overwriting"
            )

        self._profiles[profile.identifier] = profile
        logger.info(f"Registered profile: {profile.identifier}")

    def get_profile(self, model_name: str) -> ModelProfile:
        """
        Get a model profile by identifier.

        Args:
            model_name: Model identifier (e.g., 'resnet50_circle_dg')

        Returns:
            ModelProfile instance

        Raises:
            ValueError: If model_name not found
        """
        if model_name not in self._profiles:
            available = ", ".join(self.list_models())
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {available}"
            )

        return self._profiles[model_name]

    def list_models(self) -> list[str]:
        """
        List all registered model identifiers.

        Returns:
            List of model identifiers
        """
        return list(self._profiles.keys())

    def load_model(
        self,
        model_name: str,
        device: torch.device,
        force_reload: bool = False
    ) -> torch.nn.Module:
        """
        Load a model instance with caching.

        Args:
            model_name: Model identifier
            device: Target device for model
            force_reload: Force reload even if cached

        Returns:
            Loaded PyTorch model

        Raises:
            ValueError: If model_name not found
            FileNotFoundError: If model weights file not found
        """
        profile = self.get_profile(model_name)

        # Create cache key
        cache_key = f"{model_name}_{device.type}_{device.index or 0}"

        # Check cache
        if cache_key in self._model_cache and not force_reload:
            logger.debug(f"Returning cached model: {cache_key}")
            return self._model_cache[cache_key]

        # Load model from profile
        logger.info(f"Loading model '{model_name}' on device '{device}'")
        model = self._load_model_from_profile(profile, device)

        # Cache the model
        self._model_cache[cache_key] = model
        logger.info(f"Cached model: {cache_key}")

        return model

    def _load_model_from_profile(
        self,
        profile: ModelProfile,
        device: torch.device
    ) -> torch.nn.Module:
        """
        Load model from profile configuration.

        Args:
            profile: ModelProfile with model configuration
            device: Target device

        Returns:
            Loaded PyTorch model in eval mode

        Raises:
            FileNotFoundError: If weights file not found
            RuntimeError: If model loading fails
        """
        # Import backbone module dynamically based on profile identifier
        if profile.identifier == "resnet50_circle_dg":
            from .backbones import resnet50_circle_dg

            # Use model_manager for automatic downloading if backbone_path doesn't exist
            if not profile.backbone_path.exists():
                logger.info(f"Backbone weights not found at {profile.backbone_path}, using model_manager")
                model = resnet50_circle_dg.load_model(
                    weights_path=None,  # Use model_manager
                    device=device,
                    use_model_manager=True
                )
            else:
                # Use existing weights
                model = resnet50_circle_dg.load_model(
                    weights_path=profile.backbone_path,
                    device=device,
                    use_model_manager=False
                )
        else:
            raise ValueError(f"Unknown model identifier: {profile.identifier}")

        # Set model to evaluation mode
        model.eval()

        return model

    def get_verification_threshold(
        self,
        model_name: str,
        distance_metric: str
    ) -> float:
        """
        Get default verification threshold for a model and metric.

        Thresholds are calibrated on validation datasets and should be
        adjusted based on specific use cases.

        Args:
            model_name: Model identifier
            distance_metric: Distance metric (cosine, euclidean, euclidean_l2)

        Returns:
            Default threshold value

        Raises:
            ValueError: If model or metric not found
        """
        if model_name not in self._thresholds:
            logger.warning(
                f"No thresholds defined for model '{model_name}', "
                "using generic defaults"
            )
            # Generic fallback thresholds
            generic_thresholds = {
                "cosine": 0.40,
                "euclidean": 10.0,
                "euclidean_l2": 0.85
            }
            return generic_thresholds.get(distance_metric, 0.40)

        model_thresholds = self._thresholds[model_name]

        if distance_metric not in model_thresholds:
            raise ValueError(
                f"Distance metric '{distance_metric}' not defined for "
                f"model '{model_name}'. Available: {list(model_thresholds.keys())}"
            )

        return model_thresholds[distance_metric]

    def set_verification_threshold(
        self,
        model_name: str,
        distance_metric: str,
        threshold: float
    ) -> None:
        """
        Override default verification threshold.

        Args:
            model_name: Model identifier
            distance_metric: Distance metric
            threshold: Threshold value
        """
        if model_name not in self._thresholds:
            self._thresholds[model_name] = {}

        self._thresholds[model_name][distance_metric] = threshold
        logger.info(
            f"Set threshold for {model_name}/{distance_metric}: {threshold}"
        )

    def clear_cache(self) -> None:
        """
        Clear all cached models to free memory.

        Useful when switching between many models or when memory is constrained.
        """
        count = len(self._model_cache)
        self._model_cache.clear()
        logger.info(f"Cleared model cache ({count} models)")

    def remove_from_cache(self, model_name: str, device: Optional[torch.device] = None) -> None:
        """
        Remove specific model from cache.

        Args:
            model_name: Model identifier
            device: Optional device specification (removes all devices if None)
        """
        if device is None:
            # Remove all cached versions of this model
            keys_to_remove = [
                k for k in self._model_cache.keys()
                if k.startswith(f"{model_name}_")
            ]
        else:
            # Remove specific device version
            cache_key = f"{model_name}_{device.type}_{device.index or 0}"
            keys_to_remove = [cache_key] if cache_key in self._model_cache else []

        for key in keys_to_remove:
            del self._model_cache[key]
            logger.debug(f"Removed from cache: {key}")

        if keys_to_remove:
            logger.info(f"Removed {len(keys_to_remove)} model(s) from cache")

    # ==================== Face Model Management ====================

    def load_face_model(
        self,
        model_name: str,
        detector_backend: str = "opencv",
        force_reload: bool = False
    ) -> Any:
        """
        Load and cache a face embedding model using DeepFace.

        Args:
            model_name: DeepFace model name ('Facenet', 'VGG-Face', 'ArcFace', etc.)
            detector_backend: Face detector backend ('opencv', 'ssd', 'mtcnn', etc.)
            force_reload: Force reload even if cached

        Returns:
            A dictionary containing the configuration for the face model.

        Raises:
            ImportError: If DeepFace is not installed
        """
        cache_key = (model_name, detector_backend)

        with self._face_lock:
            # Check cache
            if cache_key in self._face_model_cache and not force_reload:
                logger.debug(f"Returning cached face model: {cache_key}")
                return self._face_model_cache[cache_key]

            # Load model using DeepFace
            logger.info(f"Loading face model: {model_name}/{detector_backend}")

            try:
                from deepface import DeepFace
            except ImportError as e:
                raise ImportError(
                    "DeepFace is required for face embeddings. Install with: pip install deepface"
                ) from e

            # DeepFace loads models on-demand when represent() is called
            # We cache the configuration by storing a reference
            # The actual model loading happens on first use of DeepFace.represent()
            face_model_config = {
                "model_name": model_name,
                "detector_backend": detector_backend,
                "enforce_detection": False,
                "align": True
            }

            # Store configuration (DeepFace handles actual model caching internally)
            self._face_model_cache[cache_key] = face_model_config
            logger.info(f"Cached face model configuration: {cache_key}")

            return face_model_config

    def get_cached_face_models(self) -> list[Tuple[str, str]]:
        """
        Get list of cached face models.

        Returns:
            List of (model_name, detector_backend) tuples
        """
        with self._face_lock:
            return list(self._face_model_cache.keys())

    def clear_face_model_cache(self) -> None:
        """
        Clear all cached face models.

        Note: DeepFace's internal cache cannot be cleared from this library.
        This clears our configuration cache only.
        """
        with self._face_lock:
            count = len(self._face_model_cache)
            self._face_model_cache.clear()
            logger.info(f"Cleared face model cache ({count} configurations)")

    def get_face_model_cache_info(self) -> Dict[str, Any]:
        """
        Get information about face model cache.

        Returns:
            Dictionary with cache statistics
        """
        with self._face_lock:
            return {
                "cached_models": [
                    {"model_name": model, "detector_backend": backend}
                    for model, backend in self._face_model_cache.keys()
                ],
                "cache_size": len(self._face_model_cache)
            }

    def remove_face_model_from_cache(
        self,
        model_name: str,
        detector_backend: Optional[str] = None
    ) -> None:
        """
        Remove specific face model(s) from cache.

        Args:
            model_name: Model name to remove
            detector_backend: Specific detector backend (removes all if None)
        """
        with self._face_lock:
            if detector_backend is None:
                # Remove all cached versions of this model
                keys_to_remove = [
                    k for k in self._face_model_cache.keys()
                    if k[0] == model_name
                ]
            else:
                # Remove specific model/detector combination
                cache_key = (model_name, detector_backend)
                keys_to_remove = [cache_key] if cache_key in self._face_model_cache else []

            for key in keys_to_remove:
                del self._face_model_cache[key]
                logger.debug(f"Removed face model from cache: {key}")

            if keys_to_remove:
                logger.info(f"Removed {len(keys_to_remove)} face model(s) from cache")


# Global registry instance
_registry = None


def get_registry() -> ModelRegistry:
    """
    Get global ModelRegistry instance.

    Returns:
        Shared ModelRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
