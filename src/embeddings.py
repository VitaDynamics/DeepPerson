"""
Person embedding generation pipeline.

Handles preprocessing, feature extraction, and embedding generation from person images.
Supports batch processing for efficiency.
"""

import logging
from datetime import datetime
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from .entities import Modality, PersonEmbedding
from .interfaces import EmbeddingGenerator
from .registry import ModelRegistry
from .utils import get_device_name, normalize_embedding

logger = logging.getLogger(__name__)


class BodyEmbeddingGenerator(EmbeddingGenerator):
    """
    Pipeline for generating person embeddings from cropped person images.

    Handles:
    - Image preprocessing (resize, normalize)
    - Batch processing
    - Feature extraction via backbone model
    - Post-processing (normalization)
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize embedding pipeline.

        Args:
            model_name: Model identifier (e.g., 'resnet50_circle_dg')
            device: torch.device for inference
            registry: ModelRegistry instance (creates one if None)
        """
        self._model_name = model_name
        self.device = device

        # Get or create registry
        if registry is None:
            from .registry import get_registry
            registry = get_registry()
        self.registry = registry

        # Get model profile
        self.profile = registry.get_profile(model_name)

        # Load model
        self.model = registry.load_model(model_name, device)

        # Build preprocessing pipeline
        self.preprocess = self._build_preprocessing()

        logger.info(
            f"BodyEmbeddingGenerator initialized: {model_name} on {device}, "
            f"feature_dim={self.profile.feature_dim}"
        )

    # ==================== EmbeddingGenerator Interface ====================

    @property
    def feature_dim(self) -> int:
        """Get the dimensionality of generated embeddings."""
        return self.profile.feature_dim

    @property
    def modality(self) -> str:
        """Get the modality type (always BODY for BodyEmbeddingGenerator)."""
        return Modality.BODY.value
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name
    
    # ==================== Private Methods ====================

    def _build_preprocessing(self) -> T.Compose:
        """
        Build image preprocessing pipeline from model profile.

        Returns:
            Composed torchvision transforms
        """
        config = self.profile.preprocess_config

        # Extract preprocessing parameters
        mean = config.get("mean", [0.485, 0.456, 0.406])
        std = config.get("std", [0.229, 0.224, 0.225])
        input_size = config.get("input_size", (256, 128))  # (height, width)

        # Build transform pipeline
        transforms = T.Compose([
            T.Resize(input_size),  # Resize to (H, W)
            T.ToTensor(),  # Convert to tensor [0, 1]
            T.Normalize(mean=mean, std=std)  # Normalize
        ])

        logger.debug(
            f"Preprocessing: size={input_size}, mean={mean}, std={std}"
        )

        return transforms

    def preprocess_images(
        self,
        images: List[Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess a batch of PIL images.

        Args:
            images: List of PIL Image objects

        Returns:
            Batched tensor (B, 3, H, W)
        """
        # Apply preprocessing to each image
        tensors = [self.preprocess(img) for img in images]

        # Stack into batch
        batch = torch.stack(tensors, dim=0)

        return batch

    @torch.no_grad()
    def extract_features(
        self,
        image_tensor: torch.Tensor
    ) -> np.ndarray:
        """
        Extract features from preprocessed images.

        Args:
            image_tensor: Preprocessed tensor (B, 3, H, W)

        Returns:
            Feature embeddings (B, feature_dim) as numpy array
        """
        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Forward pass
        self.model.eval()
        features = self.model(image_tensor)

        # Convert to numpy
        features_np = features.cpu().numpy().astype(np.float32)

        return features_np

    def generate_embedding(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        confidence: float,
        normalize_method: Literal["base", "resnet", "circle"] = "resnet",
        source_image_id: Optional[str] = None
    ) -> PersonEmbedding:
        """
        Generate embedding for a single person image.

        Args:
            image: Cropped person image (PIL Image)
            bbox: Bounding box in original image (x1, y1, x2, y2)
            confidence: Detection confidence
            normalize_method: Normalization method for embedding
            source_image_id: Optional source image identifier

        Returns:
            PersonEmbedding object
        """
        # Preprocess
        tensor = self.preprocess(image).unsqueeze(0)  # Add batch dimension

        # Extract features
        features = self.extract_features(tensor)[0]  # Get single result

        # Normalize embedding
        features_normalized = normalize_embedding(features, method=normalize_method)

        # Create PersonEmbedding
        embedding = PersonEmbedding(
            embedding_vector=features_normalized,
            subject_confidence=confidence,
            bbox=bbox,
            normalization=normalize_method,
            model_profile_id=self.model_name,
            hardware=get_device_name(self.device),
            timestamp=datetime.now(),
            source_image_id=source_image_id,
            modality=Modality.BODY,  # Body embeddings only
            embedding_provider=self.model_name,  # Set provider for tracking
        )

        return embedding

    def generate_embeddings_batch(
        self,
        images: List[Image.Image],
        bboxes: List[tuple[int, int, int, int]],
        confidences: List[float],
        normalize_method: Literal["base", "resnet", "circle"] = "resnet",
        source_image_ids: Optional[List[str]] = None,
        batch_size: int = 16,
        show_progress: bool = False
    ) -> List[PersonEmbedding]:
        """
        Generate embeddings for multiple person images with batching.

        Args:
            images: List of cropped person images
            bboxes: List of bounding boxes
            confidences: List of detection confidences
            normalize_method: Normalization method
            source_image_ids: Optional list of source image identifiers
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of PersonEmbedding objects
        """
        num_images = len(images)

        if source_image_ids is None:
            source_image_ids = [None] * num_images

        # Validate inputs
        assert len(bboxes) == num_images, "Mismatch in number of bboxes"
        assert len(confidences) == num_images, "Mismatch in number of confidences"
        assert len(source_image_ids) == num_images, "Mismatch in number of source IDs"

        embeddings = []

        # Process in batches
        num_batches = (num_images + batch_size - 1) // batch_size

        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_images)

            # Get batch
            batch_images = images[start_idx:end_idx]
            batch_bboxes = bboxes[start_idx:end_idx]
            batch_confidences = confidences[start_idx:end_idx]
            batch_source_ids = source_image_ids[start_idx:end_idx]

            # Preprocess batch
            batch_tensor = self.preprocess_images(batch_images)

            # Extract features
            batch_features = self.extract_features(batch_tensor)

            # Create PersonEmbedding objects
            for features, bbox, conf, source_id in zip(
                batch_features,
                batch_bboxes,
                batch_confidences,
                batch_source_ids
            ):
                # Normalize
                features_normalized = normalize_embedding(
                    features,
                    method=normalize_method
                )

                # Create embedding
                embedding = PersonEmbedding(
                    embedding_vector=features_normalized,
                    subject_confidence=conf,
                    bbox=bbox,
                    normalization=normalize_method,
                    model_profile_id=self.model_name,
                    hardware=get_device_name(self.device),
                    timestamp=datetime.now(),
                    source_image_id=source_id,
                    modality=Modality.BODY,  # Body embeddings only
                    embedding_provider=self.model_name,  # Set provider for tracking
                )

                embeddings.append(embedding)

        logger.info(
            f"Generated {len(embeddings)} embeddings in {num_batches} batch(es)"
        )

        return embeddings


def create_embedding_pipeline(
    model_name: str = "resnet50_circle_dg",
    device: Optional[torch.device] = None,
    prefer_cuda: bool = True
) -> BodyEmbeddingGenerator:
    """
    Convenience function to create an BodyEmbeddingGenerator.

    Args:
        model_name: Model identifier
        device: Target device (auto-detected if None)
        prefer_cuda: Prefer CUDA if available

    Returns:
        Initialized BodyEmbeddingGenerator
    """
    # Auto-detect device if not provided
    if device is None:
        from .utils import select_device
        device = select_device(prefer_cuda=prefer_cuda)

    pipeline = BodyEmbeddingGenerator(model_name=model_name, device=device)

    return pipeline
