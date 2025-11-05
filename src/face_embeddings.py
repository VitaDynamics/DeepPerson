"""
Face embedding generation using DeepFace library.

This module provides face detection, alignment, and embedding generation
with automatic fallback handling for failed detections. It implements the
EmbeddingGenerator interface for polymorphic use with body embedding generators.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from PIL import Image

from .entities import Modality, PersonEmbedding
from .interfaces import EmbeddingGenerator

logger = logging.getLogger(__name__)


class FaceEmbeddingGenerator(EmbeddingGenerator):
    """
    Face embedding generator using DeepFace library.

    Implements the EmbeddingGenerator interface to provide polymorphic
    face embedding generation that can be used interchangeably with
    body embedding generators.

    Attributes:
        model_name: DeepFace model identifier ('Facenet', 'VGG-Face', etc.)
        detector_backend: Face detector backend ('opencv', 'ssd', 'mtcnn', etc.)
        enforce_detection: Whether to raise errors on detection failure
        _deepface: Lazy-loaded DeepFace module
        _feature_dim: Cached feature dimension (model-dependent)

    Examples:
        >>> generator = FaceEmbeddingGenerator(model_name="Facenet")
        >>> image = Image.open("face.jpg")
        >>> embedding = generator.generate_embedding(
        ...     image=image,
        ...     bbox=(0, 0, 224, 224),
        ...     confidence=0.95
        ... )
        >>> print(f"Face embedding dimension: {embedding.face_dimension}")
    """

    # Feature dimensions for common DeepFace models
    MODEL_DIMENSIONS = {
        "Facenet": 128,
        "Facenet512": 512,
        "VGG-Face": 4096,
        "OpenFace": 128,
        "DeepFace": 4096,
        "DeepID": 160,
        "ArcFace": 512,
        "Dlib": 128,
        "SFace": 128,
    }

    def __init__(
        self,
        model_name: str = "Facenet512", # Default to Facenet512 for better performance
        detector_backend: str = "opencv",
        enforce_detection: bool = False,
    ):
        """
        Initialize face embedding generator.

        Args:
            model_name: DeepFace model to use ('Facenet', 'VGG-Face', 'OpenFace', etc.)
            detector_backend: Face detector backend ('opencv', 'ssd', 'mtcnn', etc.)
            enforce_detection: If True, raise error when no face detected

        Raises:
            ImportError: If DeepFace is not installed
        """
        self._model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection

        # Cache feature dimension
        self._feature_dim = self.MODEL_DIMENSIONS.get(model_name, 128)

        # Get face model configuration from registry (registers and caches the model)
        from .registry import get_registry
        self._registry = get_registry()
        # Note: load_face_model() registers the model configuration but we don't
        # store the config here since it's only needed by the registry's cache
        self._registry.load_face_model(
            model_name=model_name,
            detector_backend=detector_backend
        )
        logger.info(
            f"Initialized FaceEmbeddingGenerator with registry: model={model_name}, "
            f"detector={detector_backend}, feature_dim={self._feature_dim}"
        )


    def _get_deepface_config(self) -> dict:
        """
        Get DeepFace configuration for represent() call.

        Returns:
            Dictionary with DeepFace parameters
        """
        # Return configuration from registry
        return {
            "model_name": self._model_name,
            "detector_backend": self.detector_backend,
            "enforce_detection": self.enforce_detection,
            "align": True
        }

    # ==================== EmbeddingGenerator Interface ====================

    @property
    def model_name(self) -> str:
        """Get the model identifier for this generator."""
        return self._model_name

    @property
    def feature_dim(self) -> int:
        """Get the dimensionality of generated embeddings."""
        return self._feature_dim

    @property
    def modality(self) -> str:
        """Get the modality type (always FACE for FaceEmbeddingGenerator)."""
        return Modality.FACE.value

    def generate_embedding(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
        confidence: float,
        normalize_method: Literal["base", "resnet", "circle"] = "base",
        source_image_id: Optional[str] = None,
        **kwargs
    ) -> PersonEmbedding:
        """
        Generate face embedding from a PIL Image.

        Args:
            image: Face image (PIL Image) - can be full image or cropped face
            bbox: Bounding box in original image (x1, y1, x2, y2)
            confidence: Detection confidence (from face detector)
            normalize_method: Normalization method (Note: DeepFace handles its own normalization)
            source_image_id: Optional source image identifier
            **kwargs: Additional parameters:
                - align (bool): Whether to align face before embedding (default: True)
                - image_path (str): Optional path to save image temporarily

        Returns:
            PersonEmbedding: Embedding with face_embedding field populated

        Raises:
            ValueError: If face detection fails and enforce_detection=True
            ImportError: If DeepFace is not installed

        Examples:
            >>> from PIL import Image
            >>> generator = FaceEmbeddingGenerator()
            >>> img = Image.open("face.jpg")
            >>> emb = generator.generate_embedding(img, (0, 0, 224, 224), 0.95)
            >>> assert emb.modality == Modality.FACE
        """
        align = kwargs.get("align", True)

        # DeepFace requires a file path, so we need to save temporarily if not provided
        temp_path = None
        image_path = kwargs.get("image_path")

        if image_path is None:
            # Save image temporarily
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "deepperson_faces"
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / f"temp_{id(image)}.jpg"
            image.save(temp_path)
            image_path = str(temp_path)

        try:
            # Generate embedding using DeepFace
            config = self._get_deepface_config()
            from deepface import DeepFace

            result = DeepFace.represent(
                img_path=image_path,
                **config
            )


            # Handle no detection
            if not result or len(result) == 0:
                logger.warning(f"No face detected in image")
                if self.enforce_detection:
                    raise ValueError("No face detected in image")

                # Return embedding with None face_embedding
                return PersonEmbedding(
                    embedding_vector=np.zeros(self._feature_dim, dtype=np.float32),
                    subject_confidence=0.0,
                    bbox=bbox,
                    normalization=normalize_method,
                    model_profile_id=self._model_name,
                    hardware="cpu",
                    timestamp=datetime.now(),
                    source_image_id=source_image_id,
                    modality=Modality.FACE,
                    face_embedding=None,
                    face_confidence=0.0,
                    embedding_provider=f"deepface_{self._model_name}",
                )

            # Use first detected face
            face_data = result[0]
            face_embedding = np.array(face_data["embedding"], dtype=np.float32)

            # Extract face bbox if available
            face_bbox = None
            if "facial_area" in face_data:
                facial_area = face_data["facial_area"]
                face_bbox = (
                    facial_area.get("x", 0),
                    facial_area.get("y", 0),
                    facial_area.get("x", 0) + facial_area.get("w", 0),
                    facial_area.get("y", 0) + facial_area.get("h", 0),
                )

            # DeepFace doesn't provide confidence directly
            face_confidence = 0.9 if len(result) > 0 else 0.0

            logger.debug(
                f"Generated face embedding: dim={len(face_embedding)}, "
                f"confidence={face_confidence:.3f}"
            )

            # Create PersonEmbedding with face as primary embedding
            return PersonEmbedding(
                embedding_vector=face_embedding,  # Face is primary
                subject_confidence=confidence,  # Original detection confidence
                bbox=bbox,  # Original bbox
                normalization=normalize_method,
                model_profile_id=self._model_name,
                hardware="cpu",  # DeepFace typically uses CPU
                timestamp=datetime.now(),
                source_image_id=source_image_id,
                modality=Modality.FACE,
                face_embedding=face_embedding,  # Also store in face field
                face_confidence=face_confidence,
                face_bbox=face_bbox or bbox,
                embedding_provider=f"deepface_{self._model_name}",
            )

        except Exception as e:
            logger.error(f"Error generating face embedding: {e}")
            if self.enforce_detection:
                raise

            # Return empty embedding on failure
            return PersonEmbedding(
                embedding_vector=np.zeros(self._feature_dim, dtype=np.float32),
                subject_confidence=0.0,
                bbox=bbox,
                normalization=normalize_method,
                model_profile_id=self._model_name,
                hardware="cpu",
                timestamp=datetime.now(),
                source_image_id=source_image_id,
                modality=Modality.FACE,
                face_embedding=None,
                face_confidence=0.0,
                embedding_provider=f"deepface_{self._model_name}",
            )

        finally:
            # Clean up temporary file
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def generate_embeddings_batch(
        self,
        images: List[Image.Image],
        bboxes: List[tuple[int, int, int, int]],
        confidences: List[float],
        normalize_method: Literal["base", "resnet", "circle"] = "base",
        source_image_ids: Optional[List[str]] = None,
        batch_size: int = 16,
        show_progress: bool = False,
        **kwargs
    ) -> List[PersonEmbedding]:
        """
        Generate face embeddings for multiple images.

        Note: DeepFace doesn't support true batch processing, so this
        processes images sequentially with optional progress bar.

        Args:
            images: List of face images (PIL Images)
            bboxes: List of bounding boxes
            confidences: List of detection confidences
            normalize_method: Normalization method
            source_image_ids: Optional list of source image identifiers
            batch_size: Ignored (for interface compatibility)
            show_progress: Whether to show progress bar
            **kwargs: Additional parameters passed to generate_embedding

        Returns:
            List[PersonEmbedding]: List of face embeddings

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        num_images = len(images)

        if source_image_ids is None:
            source_image_ids = [None] * num_images

        # Validate inputs
        if len(bboxes) != num_images:
            raise ValueError(f"Mismatch in number of bboxes: {len(bboxes)} != {num_images}")
        if len(confidences) != num_images:
            raise ValueError(f"Mismatch in number of confidences: {len(confidences)} != {num_images}")
        if len(source_image_ids) != num_images:
            raise ValueError(f"Mismatch in number of source IDs: {len(source_image_ids)} != {num_images}")

        embeddings = []

        # Create iterator with optional progress bar
        iterator = enumerate(zip(images, bboxes, confidences, source_image_ids))
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Generating face embeddings")

        # Process images sequentially (DeepFace limitation)
        for idx, (image, bbox, conf, source_id) in iterator:
            try:
                embedding = self.generate_embedding(
                    image=image,
                    bbox=bbox,
                    confidence=conf,
                    normalize_method=normalize_method,
                    source_image_id=source_id,
                    **kwargs
                )
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding for image {idx}: {e}")
                if self.enforce_detection:
                    raise

                # Add empty embedding on failure
                embeddings.append(
                    PersonEmbedding(
                        embedding_vector=np.zeros(self._feature_dim, dtype=np.float32),
                        subject_confidence=0.0,
                        bbox=bbox,
                        normalization=normalize_method,
                        model_profile_id=self._model_name,
                        hardware="cpu",
                        timestamp=datetime.now(),
                        source_image_id=source_id,
                        modality=Modality.FACE,
                        face_embedding=None,
                        face_confidence=0.0,
                        embedding_provider=f"deepface_{self._model_name}",
                    )
                )

        logger.info(f"Generated {len(embeddings)} face embeddings")

        return embeddings
