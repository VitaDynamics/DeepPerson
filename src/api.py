"""
DeepPerson API Facade

Main public interface for the DeepPerson library, providing methods for:
- represent: Generate person embeddings from images
- verify: Compare two images for identity verification
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import torch
from PIL import Image

from .detectors import DetectorFactory, PersonDetector
from .distance import compute_distance
from .embeddings import BodyEmbeddingGenerator
from .entities import PersonEmbedding
from .fusion import FusionScorer
from .registry import get_registry
from .utils import select_device

logger = logging.getLogger(__name__)

# Type alias for all supported image input types
ImageInput = Union[str, Path, Image.Image, np.ndarray]


class DeepPerson:
    """
    Main facade for DeepPerson functionality.

    Provides high-level methods for person re-identification including
    embedding generation and identity verification.

    Examples:
        >>> from src.api import DeepPerson
        >>> dp = DeepPerson(model_name="resnet50_circle_dg")
        >>>
        >>> # Generate embeddings for persons in image
        >>> result = dp.represent("image.jpg")
        >>>
        >>> # Access embeddings
        >>> for subject in result["subjects"]:
        ...     print(f"Embedding shape: {subject['embedding'].shape}")
        ...     print(f"Confidence: {subject['metadata']['confidence']}")
        >>>
        >>> # Verify if two images show the same person
        >>> result = dp.verify("person1.jpg", "person2.jpg")
        >>> print(f"Same person: {result['verified']}")
    """

    def __init__(
        self,
        model_name: str = "resnet50_circle_dg",
        device: str | torch.device | None = None,
        detector_backend: str = "yolo",
    ):
        """
        Initialize DeepPerson with specified model and device.

        Args:
            model_name: Name of the backbone model to use
            device: Device to run on ("cuda", "cpu", torch.device, or None for auto-detection)
            detector_backend: Person detection backend ('yolo', 'ultralytics', 'fasterrcnn', 'torchvision')
        """
        self.model_name = model_name

        # Handle device
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = select_device(prefer_cuda=True)

        self.detector_backend = detector_backend

        # Initialize components
        logger.info(
            f"Initializing DeepPerson: model={model_name}, device={self.device}, detector={detector_backend}"
        )

        # Initialize detector
        self.detector: PersonDetector = DetectorFactory.create_detector(
            backend=detector_backend, device=self.device
        )

        # Initialize embedding pipeline
        self.embedding_pipeline = BodyEmbeddingGenerator(
            model_name=model_name, device=self.device
        )

        # Get registry for threshold lookups
        self.registry = get_registry()

        logger.info("DeepPerson initialized successfully")

    @staticmethod
    def _normalize_image(img_input: ImageInput) -> tuple[Image.Image, str]:
        """
        Convert various image formats to PIL Image and generate source ID.

        Args:
            img_input: Image in any supported format (str, Path, PIL.Image, np.ndarray)

        Returns:
            Tuple of (PIL Image in RGB mode, source_id string)

        Raises:
            FileNotFoundError: If path provided but file doesn't exist
            TypeError: If unsupported image type provided
        """
        if isinstance(img_input, (str, Path)):
            path = Path(img_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return Image.open(path).convert("RGB"), str(path)
        elif isinstance(img_input, Image.Image):
            source_id = f"in_memory_{uuid.uuid4().hex[:8]}"
            return img_input.convert("RGB"), source_id
        elif isinstance(img_input, np.ndarray):
            source_id = f"in_memory_{uuid.uuid4().hex[:8]}"
            return Image.fromarray(img_input).convert("RGB"), source_id
        else:
            raise TypeError(
                f"Unsupported image type: {type(img_input)}. "
                f"Supported types: str, Path, PIL.Image.Image, np.ndarray"
            )

    @staticmethod
    def _validate_batch_type_consistency(images: list[ImageInput]) -> None:
        """
        Ensure all images in batch are the same type.

        Mixed types within a single batch are not supported for consistency
        and predictable behavior.

        Args:
            images: List of images in various formats

        Raises:
            TypeError: If mixed types detected in batch
        """
        if not images:
            return

        # Determine first type (treat str and Path as compatible)
        first_type = type(images[0])
        is_first_path = isinstance(images[0], (str, Path))

        # Check consistency
        for i, img in enumerate(images[1:], start=1):
            if is_first_path:
                if not isinstance(img, (str, Path)):
                    raise TypeError(
                        f"Mixed batch types not supported. "
                        f"Image 0 is {type(images[0]).__name__}, "
                        f"but image {i} is {type(img).__name__}. "
                        f"All items in batch must be the same type "
                        f"(str/Path are considered compatible)."
                    )
            elif not isinstance(img, first_type):
                raise TypeError(
                    f"Mixed batch types not supported. "
                    f"Image 0 is {first_type.__name__}, "
                    f"but image {i} is {type(img).__name__}. "
                    f"All items in batch must be the same type."
                )

    def represent(
        self,
        img_path: ImageInput | list[ImageInput],
        detector_backend: str | None = None,
        normalization: Literal["base", "resnet", "circle"] = "resnet",
        batch_size: int = 16,
        confidence_threshold: float = 0.5,
        generate_face_embeddings: bool = False,
        face_model_name: str = "Facenet",
        face_detector_backend: str = "opencv",
        return_multi_modal: bool = False,
    ) -> dict[str, Any]:
        """
        Generate person embeddings from image(s) with optional multi-modal support.

        Detects persons in each image and generates embeddings for all detected subjects.
        Optionally generates face embeddings for multi-modal fusion capabilities.

        Args:
            img_path: Image or list of images (str path, Path, PIL.Image, or np.ndarray)
            detector_backend: Override detector backend (None to use default)
            normalization: Normalization method for embeddings ('base', 'resnet', 'circle')
            batch_size: Batch size for embedding generation
            confidence_threshold: Minimum detection confidence threshold
            generate_face_embeddings: Whether to generate face embeddings
            face_model_name: Face recognition model to use (default: "Facenet")
            face_detector_backend: Face detection backend (default: "opencv")
            return_multi_modal: Return multi-modal PersonEmbedding objects

        Returns:
            Dictionary containing:
                - subjects: List of detected persons with embeddings and metadata
                - warnings: List of warning messages (e.g., no detection)
                - model_info: Model and hardware information
                - face_model_info: Face model information (if generate_face_embeddings=True)

        Examples:
            >>> # From file path
            >>> result = dp.represent("person.jpg")
            >>> print(f"Detected {len(result['subjects'])} person(s)")
            >>>
            >>> # From PIL Image
            >>> from PIL import Image
            >>> pil_img = Image.open("person.jpg")
            >>> result = dp.represent(pil_img)
            >>>
            >>> # From NumPy array
            >>> import numpy as np
            >>> numpy_img = np.array(Image.open("person.jpg"))
            >>> result = dp.represent(numpy_img)
            >>>
            >>> # Multi-modal body+face embeddings (any input type)
            >>> result = dp.represent(
            ...     pil_img,
            ...     generate_face_embeddings=True,
            ...     face_model_name="Facenet"
            ... )
            >>> for subject in result["subjects"]:
            ...     body_emb = subject["embedding"]  # Body embedding
            ...     face_emb = subject.get("face_embedding")  # Face embedding (if detected)
            ...     print(f"Body: {body_emb.shape}, Face: {face_emb.shape if face_emb is not None else 'None'}")
            >>>
            >>> # Batch processing (all same type)
            >>> result = dp.represent([pil_img1, pil_img2, pil_img3])
            >>>
            >>> # Batch with paths
            >>> result = dp.represent(["img1.jpg", "img2.jpg"], generate_face_embeddings=True)
        """
        # Normalize input to list and validate consistency
        if isinstance(img_path, (str, Path, Image.Image, np.ndarray)):
            input_list = [img_path]
            single_image = True
        else:
            # Handle list/iterable inputs
            try:
                input_list = list(img_path)
            except TypeError:
                raise TypeError(
                    f"Unsupported image type: {type(img_path)}. "
                    f"Expected str, Path, PIL.Image, np.ndarray, or list of these types."
                )
            single_image = False
            # Validate batch type consistency
            self._validate_batch_type_consistency(input_list)

        # Normalize all images to PIL and get source IDs
        normalized_images = []
        source_ids = []
        for img_input in input_list:
            pil_img, source_id = self._normalize_image(img_input)
            normalized_images.append(pil_img)
            source_ids.append(source_id)

        # Use override detector if specified
        if detector_backend is not None:
            detector = DetectorFactory.create_detector(
                backend=detector_backend, device=self.device
            )
        else:
            detector = self.detector

        # Create face embedding generator if needed
        face_generator = None
        warnings_list = []
        if generate_face_embeddings:
            try:
                from .face_embeddings import FaceEmbeddingGenerator

                face_generator = FaceEmbeddingGenerator(
                    model_name=face_model_name,
                    detector_backend=face_detector_backend,
                    enforce_detection=False,
                )
                logger.info(f"Face embedding generator ready: {face_model_name}")
            except ImportError:
                warning_msg = (
                    "DeepFace not available, face embeddings will be skipped. "
                    "Install with: pip install deepface"
                )
                logger.warning(warning_msg)
                warnings_list.append(warning_msg)
                generate_face_embeddings = False

        # Results containers
        all_subjects = []

        # Process each image
        for pil_image, source_id in zip(normalized_images, source_ids):
            # Detect persons in image
            logger.debug(f"Processing image: {source_id}")
            detections = detector.detect(
                image=pil_image, confidence_threshold=confidence_threshold
            )

            # Handle no detections
            if len(detections) == 0:
                # Extract display name from source_id
                display_name = (
                    Path(source_id).name
                    if not source_id.startswith("in_memory_")
                    else source_id
                )
                warning_msg = f"No person detected in {display_name}"
                logger.warning(warning_msg)
                warnings_list.append(warning_msg)
                continue

            logger.debug(f"Detected {len(detections)} person(s) in {source_id}")

            # Crop detected persons
            cropped_persons = detector.crop_persons(
                image=pil_image, detections=detections
            )

            # Prepare for batch embedding generation
            bboxes = [det.bbox for det in detections]
            confidences = [det.confidence for det in detections]
            source_ids_for_batch = [source_id] * len(detections)

            # Generate body embeddings (batch processing within image)
            body_embeddings: list[PersonEmbedding] = (
                self.embedding_pipeline.generate_embeddings_batch(
                    images=cropped_persons,
                    bboxes=bboxes,
                    confidences=confidences,
                    normalize_method=normalization,
                    source_image_ids=source_ids_for_batch,
                    batch_size=batch_size,
                    show_progress=False,
                )
            )

            # Generate face embeddings if requested
            face_embeddings = []
            if generate_face_embeddings and face_generator:
                try:
                    face_embeddings = face_generator.generate_embeddings_batch(
                        images=cropped_persons,
                        bboxes=bboxes,
                        confidences=confidences,
                        normalize_method="base",  # DeepFace handles normalization
                        source_image_ids=source_ids_for_batch,
                        batch_size=batch_size,
                        show_progress=False,
                    )
                    logger.debug(f"Generated {len(face_embeddings)} face embeddings")
                except Exception as e:
                    logger.warning(f"Face embedding generation failed: {e}")
                    face_embeddings = [None] * len(body_embeddings)
            else:
                face_embeddings = [None] * len(body_embeddings)

            # Combine body and face embeddings
            combined_embeddings = []
            for body_emb, face_emb in zip(body_embeddings, face_embeddings):
                if face_emb and face_emb.face_embedding is not None:
                    # Create multi-modal embedding
                    from .entities import Modality

                    combined_emb = PersonEmbedding(
                        embedding_vector=body_emb.embedding_vector,
                        subject_confidence=body_emb.subject_confidence,
                        bbox=body_emb.bbox,
                        normalization=body_emb.normalization,
                        model_profile_id=body_emb.model_profile_id,
                        hardware=body_emb.hardware,
                        timestamp=body_emb.timestamp,
                        source_image_id=body_emb.source_image_id,
                        modality=Modality.BODY_FACE,
                        face_embedding=face_emb.face_embedding,
                        face_confidence=face_emb.face_confidence,
                        face_bbox=face_emb.face_bbox,
                        embedding_provider=body_emb.embedding_provider,
                        metadata=body_emb.metadata,
                    )
                    combined_embeddings.append(combined_emb)
                else:
                    # Body-only embedding
                    combined_embeddings.append(body_emb)

            # Package subjects
            for embedding in combined_embeddings:
                subject = {
                    "embedding": embedding.embedding_vector,
                    "metadata": {
                        "bbox": embedding.bbox,
                        "confidence": embedding.subject_confidence,
                        "hardware": embedding.hardware,
                        "model_profile_id": embedding.model_profile_id,
                        "normalization": embedding.normalization,
                        "timestamp": embedding.timestamp.isoformat()
                        if embedding.timestamp
                        else None,
                        "source_image": embedding.source_image_id,
                        "modality": embedding.modality.value,
                    },
                }

                # Add face embedding data if available
                if embedding.has_face_embedding:
                    subject["face_embedding"] = embedding.face_embedding
                    subject["metadata"]["face_confidence"] = embedding.face_confidence
                    subject["metadata"]["face_bbox"] = embedding.face_bbox

                # Optionally return full PersonEmbedding object
                if return_multi_modal:
                    subject["person_embedding"] = embedding

                all_subjects.append(subject)

        # Build response
        response = {
            "subjects": all_subjects,
            "warnings": warnings_list if warnings_list else None,
            "model_info": {
                "name": self.model_name,
                "device": str(self.device),
                "detector_backend": detector_backend or self.detector_backend,
                "feature_dim": self.embedding_pipeline.profile.feature_dim,
            },
        }

        # Add face model info if face embeddings were generated
        if generate_face_embeddings and face_generator:
            response["face_model_info"] = {
                "name": face_model_name,
                "detector_backend": face_detector_backend,
                "feature_dim": face_generator.feature_dim,
            }

        # Count multi-modal embeddings
        multi_modal_count = sum(
            1 for subject in all_subjects if subject.get("face_embedding") is not None
        )

        logger.info(
            f"Processed {len(normalized_images)} image(s), "
            f"generated {len(all_subjects)} embedding(s) "
            f"({multi_modal_count} multi-modal), "
            f"{len(warnings_list)} warning(s)"
        )

        return response

    def verify(
        self,
        img1_path: ImageInput,
        img2_path: ImageInput,
        model_name: str | None = None,
        detector_backend: str | None = None,
        distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = "cosine",
        threshold: float | None = None,
        normalization: Literal["base", "resnet", "circle"] = "resnet",
        enforce_detection: bool = True,
    ) -> dict[str, Any]:
        """
        Verify if two images show the same person.

        Compares embeddings from two images and determines if they represent
        the same individual based on distance metrics and thresholds.

        Args:
            img1_path: First image (str path, Path, PIL.Image, or np.ndarray)
            img2_path: Second image (str path, Path, PIL.Image, or np.ndarray)
            model_name: Override model name (uses instance default if None)
            detector_backend: Override detector backend (uses instance default if None)
            distance_metric: Distance metric ('cosine', 'euclidean', 'euclidean_l2')
            threshold: Distance threshold (None for model default from registry)
            normalization: Normalization method ('base', 'resnet', 'circle')
            enforce_detection: If True, raise error when no person detected; if False, return unverified result

        Returns:
            Dictionary containing:
                - verified: Boolean indicating if same person (distance <= threshold)
                - distance: Computed distance between embeddings
                - threshold: Threshold used for verification
                - distance_metric: Metric used
                - model: Model name used
                - detector_backend: Detector used
                - facial_areas: Bounding boxes for both images
                - body_distance: Body embedding distance
                - face_distance: Face embedding distance (if available)
                - fusion_score: Combined fusion score (if face embeddings available)
                - used_fusion: Whether fusion scoring was used
                - modality_available: Available modalities
                - warnings: List of warnings (e.g., multiple detections)

        Raises:
            ValueError: If no person detected and enforce_detection=True
            FileNotFoundError: If image file paths not found
            TypeError: If unsupported image type provided

        Examples:
            >>> # From file paths
            >>> result = dp.verify("person1_img1.jpg", "person1_img2.jpg")
            >>> if result["verified"]:
            ...     print(f"Same person! Distance: {result['distance']:.4f}")
            >>> else:
            ...     print(f"Different persons. Distance: {result['distance']:.4f}")
            >>>
            >>> # From PIL Images
            >>> from PIL import Image
            >>> pil_img1 = Image.open("person1.jpg")
            >>> pil_img2 = Image.open("person2.jpg")
            >>> result = dp.verify(pil_img1, pil_img2)
            >>>
            >>> # From NumPy arrays
            >>> import numpy as np
            >>> np_img1 = np.array(Image.open("person1.jpg"))
            >>> np_img2 = np.array(Image.open("person2.jpg"))
            >>> result = dp.verify(np_img1, np_img2)
            >>>
            >>> # Mixed types (PIL + path)
            >>> result = dp.verify(pil_img1, "person2.jpg")
            >>>
            >>> # Use different metric
            >>> result = dp.verify(pil_img1, pil_img2, distance_metric="euclidean")
        """

        # Generate display names for logging
        def _get_display_name(img_input: ImageInput) -> str:
            if isinstance(img_input, (str, Path)):
                return Path(img_input).name
            elif isinstance(img_input, Image.Image):
                return "PIL_Image"
            elif isinstance(img_input, np.ndarray):
                return "NumPy_array"
            else:
                return "Unknown"

        img1_name = _get_display_name(img1_path)
        img2_name = _get_display_name(img2_path)

        logger.info(
            f"Verifying images: {img1_name} vs {img2_name} "
            f"(metric={distance_metric}, threshold={threshold})"
        )

        # Use provided model or fall back to instance model
        effective_model = model_name or self.model_name

        # Generate embeddings for both images
        warnings_list = []

        # Process first image
        result1 = self.represent(
            img_path=img1_path,
            detector_backend=detector_backend,
            normalization=normalization,
            confidence_threshold=0.5,
            generate_face_embeddings=True,
        )

        # Check for detection issues in first image
        if len(result1["subjects"]) == 0:
            if enforce_detection:
                raise ValueError(
                    f"No person detected in first image: {Path(img1_path).name}"
                )
            else:
                logger.warning(
                    f"No person detected in first image: {Path(img1_path).name}"
                )
                return {
                    "verified": False,
                    "distance": float("inf"),
                    "threshold": threshold
                    or self.registry.get_verification_threshold(
                        effective_model, distance_metric
                    ),
                    "distance_metric": distance_metric,
                    "model": effective_model,
                    "detector_backend": detector_backend or self.detector_backend,
                    "facial_areas": {"img1": None, "img2": None},
                    "body_distance": float("inf"),
                    "face_distance": None,
                    "fusion_score": None,
                    "used_fusion": False,
                    "modality_available": {"body": False, "face": False},
                    "warnings": ["No person detected in first image"],
                }

        if len(result1["subjects"]) > 1:
            warning = f"Multiple persons detected in first image ({len(result1['subjects'])}), using first detection"
            logger.warning(warning)
            warnings_list.append(warning)

        # Process second image
        result2 = self.represent(
            img_path=img2_path,
            detector_backend=detector_backend,
            normalization=normalization,
            confidence_threshold=0.5,
            generate_face_embeddings=True,
        )

        # Check for detection issues in second image
        if len(result2["subjects"]) == 0:
            if enforce_detection:
                raise ValueError(
                    f"No person detected in second image: {Path(img2_path).name}"
                )
            else:
                logger.warning(
                    f"No person detected in second image: {Path(img2_path).name}"
                )
                return {
                    "verified": False,
                    "distance": float("inf"),
                    "threshold": threshold
                    or self.registry.get_verification_threshold(
                        effective_model, distance_metric
                    ),
                    "distance_metric": distance_metric,
                    "model": effective_model,
                    "detector_backend": detector_backend or self.detector_backend,
                    "facial_areas": {
                        "img1": result1["subjects"][0]["metadata"]["bbox"],
                        "img2": None,
                    },
                    "body_distance": float("inf"),
                    "face_distance": None,
                    "fusion_score": None,
                    "used_fusion": False,
                    "modality_available": {"body": True, "face": False},
                    "warnings": ["No person detected in second image"],
                }

        if len(result2["subjects"]) > 1:
            warning = f"Multiple persons detected in second image ({len(result2['subjects'])}), using first detection"
            logger.warning(warning)
            warnings_list.append(warning)

        # Extract embeddings (use first detection from each image)
        embedding1 = result1["subjects"][0]["embedding"]
        embedding2 = result2["subjects"][0]["embedding"]

        # Extract face embeddings if available
        face_embedding1 = result1["subjects"][0].get("face_embedding")
        face_embedding2 = result2["subjects"][0].get("face_embedding")

        # Extract facial areas for response
        facial_area1 = result1["subjects"][0]["metadata"]["bbox"]
        facial_area2 = result2["subjects"][0]["metadata"]["bbox"]

        # Initialize fusion scorer
        fusion_scorer = FusionScorer()

        # Compute body distance
        body_distance = compute_distance(embedding1, embedding2, metric=distance_metric)

        # Initialize fusion variables
        face_distance = None
        fusion_score = None
        fusion_metadata = None
        used_fusion = False
        modality_available = {"body": True, "face": False}

        # Check if face embeddings are available for fusion
        if face_embedding1 is not None and face_embedding2 is not None:
            try:
                # Compute face distance
                face_distance = compute_distance(
                    face_embedding1, face_embedding2, metric=distance_metric
                )

                # Convert distances to similarities for fusion scoring (0-1 range, higher = more similar)
                body_similarity = max(0.0, 1.0 - body_distance)
                face_similarity = max(0.0, 1.0 - face_distance)

                # Use fusion scoring to get combined similarity
                fusion_score, fusion_metadata = fusion_scorer.compute_fusion_score(
                    body_score=body_similarity,
                    face_score=face_similarity,
                    body_confidence=1.0,  # Default confidence
                    face_confidence=1.0,  # Default confidence
                )

                used_fusion = fusion_metadata["face_used"]
                modality_available["face"] = True

                logger.info(
                    f"Fusion scoring used: body_distance={body_distance:.4f}, "
                    f"face_distance={face_distance:.4f}, fusion_score={fusion_score:.4f}"
                )

            except Exception as e:
                logger.warning(f"Fusion scoring failed, falling back to body-only: {e}")
                face_distance = None
                fusion_metadata = None
        else:
            logger.info("Face embeddings NOT available, using body-only verification")

        # Get threshold (use provided or fetch from registry)
        if threshold is None:
            threshold = self.registry.get_verification_threshold(
                effective_model, distance_metric
            )
            logger.debug(f"Using default threshold from registry: {threshold}")

        # Determine verification result
        if used_fusion and fusion_score is not None:
            # For fusion scores (similarity), verified if score >= threshold
            verified = bool(fusion_score >= threshold)
            logger.info(
                f"Verification result: {verified} "
                f"(fusion_score={fusion_score:.4f}, threshold={threshold:.4f})"
            )
        else:
            # For body distance, verified if distance <= threshold
            verified = bool(body_distance <= threshold)
            logger.info(
                f"Verification result: {verified} "
                f"(body_distance={body_distance:.4f}, threshold={threshold:.4f})"
            )

        # Build response with enhanced structure
        response = {
            "verified": verified,
            "distance": float(body_distance),  # Maintain backward compatibility
            "threshold": float(threshold),
            "distance_metric": distance_metric,
            "model": effective_model,
            "detector_backend": detector_backend or self.detector_backend,
            "facial_areas": {"img1": facial_area1, "img2": facial_area2},
            # Enhanced fusion fields
            "body_distance": float(body_distance),
            "face_distance": float(face_distance)
            if face_distance is not None
            else None,
            "fusion_score": float(fusion_score) if fusion_score is not None else None,
            "face_weight": fusion_metadata.get("face_weight", 0.5)
            if used_fusion
            else 0.5,
            "body_weight": fusion_metadata.get("body_weight", 0.5)
            if used_fusion
            else 0.5,
            "used_fusion": used_fusion,
            "modality_available": modality_available,
        }

        # Add warnings if any
        if warnings_list:
            response["warnings"] = warnings_list

        return response

    def batch_represent(
        self,
        image_paths: list[ImageInput],
        generate_face_embeddings: bool = False,
        normalization: Literal["base", "resnet", "circle"] = "resnet",
        batch_size: int | None = None,
        confidence_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Generate person embeddings for multiple images using batch processing.

        Processes multiple images in batches for improved performance (2-5x speedup
        for 5+ images). Uses vectorized operations for detection and body embedding
        generation.

        Args:
            image_paths: List of images (str paths, Paths, PIL.Images, or np.ndarrays - all same type)
            generate_face_embeddings: Whether to generate face embeddings
            normalization: Normalization method for embeddings
            batch_size: Batch size hint for processing (auto-selected if None)
            confidence_threshold: Minimum detection confidence threshold

        Returns:
            Dictionary containing:
                - results: List of per-image results (same structure as represent())
                - batch_metadata: Batch processing statistics and performance metrics
                - processing_time: Total batch processing time in seconds
                - success_count: Number of successfully processed images
                - error_count: Number of failed images

        Examples:
            >>> # Batch process file paths
            >>> result = dp.batch_represent([
            ...     "img1.jpg",
            ...     "img2.jpg",
            ...     "img3.jpg"
            ... ])
            >>> print(f"Processed {result['success_count']} images")
            >>> for img_result in result['results']:
            ...     print(f"Found {len(img_result['subjects'])} persons")
            >>>
            >>> # Batch process PIL Images (all same type)
            >>> from PIL import Image
            >>> pil_images = [Image.open(f"img{i}.jpg") for i in range(1, 4)]
            >>> result = dp.batch_represent(pil_images)
            >>>
            >>> # Batch process NumPy arrays (all same type)
            >>> import numpy as np
            >>> np_images = [np.array(Image.open(f"img{i}.jpg")) for i in range(1, 4)]
            >>> result = dp.batch_represent(np_images)
        """
        import uuid

        from .entities import BatchMetadata
        from .utils import (
            TimingContext,
            cleanup_memory,
            get_hardware_info,
            get_memory_stats,
        )

        # Input validation
        if not image_paths:
            raise ValueError("image_paths must be a non-empty list")

        # Validate batch type consistency
        self._validate_batch_type_consistency(image_paths)

        # Normalize all images upfront
        normalized_images = []
        source_ids_list = []
        for img_input in image_paths:
            try:
                pil_img, source_id = self._normalize_image(img_input)
                normalized_images.append(pil_img)
                source_ids_list.append(source_id)
            except Exception as e:
                logger.error(f"Failed to normalize image: {e}")
                # Add placeholder for failed image (will be handled in processing)
                normalized_images.append(None)
                source_ids_list.append(f"error_{uuid.uuid4().hex[:8]}")

        # Create timing context for performance tracking
        timer = TimingContext()

        # Log initial memory state
        initial_memory = get_memory_stats(self.device)
        if self.device.type == "cuda":
            logger.info(
                f"Initial GPU memory: {initial_memory['allocated_gb']:.2f}GB allocated, "
                f"{initial_memory['free_gb']:.2f}GB free"
            )

        # Auto-select batch size if not provided
        if batch_size is None:
            # Conservative batch size based on available memory
            if self.device.type == "cuda" and initial_memory.get("free_gb", 0) > 0:
                # Rough estimate: ~100MB per image for detection + embedding
                estimated_per_image_gb = 0.1
                max_batch = int(initial_memory["free_gb"] / estimated_per_image_gb)
                batch_size = min(16, max(4, max_batch), len(image_paths))
            else:
                batch_size = min(16, len(image_paths))

        logger.info(f"Using batch_size={batch_size} for {len(image_paths)} images")

        results = []
        success_count = 0
        error_count = 0

        # Phase 1: Batch detection
        with timer.stage("detection"):
            try:
                # Filter out None images (normalization failures)
                valid_images = [img for img in normalized_images if img is not None]
                if len(valid_images) < len(normalized_images):
                    logger.warning(
                        f"{len(normalized_images) - len(valid_images)} image(s) failed normalization"
                    )

                all_detections = self.detector.detect_batch(
                    images=valid_images, confidence_threshold=confidence_threshold
                )

                # Re-inject empty detections for failed normalizations
                detection_idx = 0
                full_detections = []
                for img in normalized_images:
                    if img is not None:
                        full_detections.append(all_detections[detection_idx])
                        detection_idx += 1
                    else:
                        full_detections.append([])  # Empty detection for failed image
                all_detections = full_detections

            except RuntimeError as e:
                # Handle CUDA OOM errors
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "CUDA OOM during detection, cleaning up and retrying with smaller batches"
                    )
                    cleanup_memory(self.device)
                    # Retry with sequential detection
                    all_detections = []
                    for pil_img in normalized_images:
                        if pil_img is not None:
                            try:
                                dets = self.detector.detect(
                                    pil_img, confidence_threshold
                                )
                                all_detections.append(dets)
                            except Exception:
                                all_detections.append([])
                        else:
                            all_detections.append([])
                else:
                    raise
            except Exception as e:
                logger.error(f"Batch detection failed: {e}")
                # Create error results for all images
                for i, source_id in enumerate(source_ids_list):
                    results.append(
                        {
                            "subjects": [],
                            "image_index": i,
                            "processing_status": "error",
                            "error_message": f"Detection failed: {str(e)}",
                        }
                    )
                error_count = len(image_paths)
                all_detections = [[] for _ in image_paths]
                # Clean up memory before continuing
                cleanup_memory(self.device)

        # Phase 2: Process each image (crop, embed)
        for image_index, (pil_img, source_id, detections) in enumerate(
            zip(normalized_images, source_ids_list, all_detections)
        ):
            try:
                # Check if image normalization failed
                if pil_img is None:
                    results.append(
                        {
                            "subjects": [],
                            "image_index": image_index,
                            "processing_status": "error",
                            "error_message": "Image normalization failed",
                        }
                    )
                    error_count += 1
                    continue

                if len(detections) == 0:
                    # No persons detected
                    results.append(
                        {
                            "subjects": [],
                            "model_info": self._get_model_info(),
                            "image_index": image_index,
                            "processing_status": "success",
                            "error_message": None,
                        }
                    )
                    success_count += 1
                    continue

                # Crop detected persons
                person_crops = self.detector.crop_persons(pil_img, detections)

                # Extract metadata
                bboxes = [det.bbox for det in detections]
                confidences = [det.confidence for det in detections]

                # Generate body embeddings with OOM handling
                with timer.stage("body_embedding"):
                    try:
                        embeddings = self.embedding_pipeline.generate_embeddings_batch(
                            images=person_crops,
                            bboxes=bboxes,
                            confidences=confidences,
                            normalize_method=normalization,
                            batch_size=batch_size,
                            show_progress=False,
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(
                                f"CUDA OOM during embedding for image {image_index}, cleaning up and retrying"
                            )
                            cleanup_memory(self.device)
                            # Retry with smaller batch size
                            embeddings = (
                                self.embedding_pipeline.generate_embeddings_batch(
                                    images=person_crops,
                                    bboxes=bboxes,
                                    confidences=confidences,
                                    normalize_method=normalization,
                                    batch_size=max(1, batch_size // 2),
                                    show_progress=False,
                                )
                            )
                        else:
                            raise

                # Build subjects list
                subjects = []
                for embedding in embeddings:
                    subject = {
                        "embedding": embedding.embedding_vector,
                        "metadata": {
                            "confidence": embedding.subject_confidence,
                            "bbox": embedding.bbox,
                            "normalization": embedding.normalization,
                        },
                        "modality": embedding.modality.value,
                    }
                    subjects.append(subject)

                # Create result for this image
                result = {
                    "subjects": subjects,
                    "model_info": self._get_model_info(),
                    "image_index": image_index,
                    "processing_status": "success",
                    "error_message": None,
                }

                results.append(result)
                success_count += 1

            except Exception as e:
                logger.error(
                    f"Failed to process image {image_index} ({source_id}): {e}"
                )
                results.append(
                    {
                        "subjects": [],
                        "image_index": image_index,
                        "processing_status": "error",
                        "error_message": str(e),
                    }
                )
                error_count += 1

            # Periodic cleanup for very large batches (every 20 images)
            if (image_index + 1) % 20 == 0:
                cleanup_memory(self.device)
                current_memory = get_memory_stats(self.device)
                if self.device.type == "cuda":
                    logger.debug(
                        f"Memory after {image_index + 1} images: "
                        f"{current_memory['allocated_gb']:.2f}GB allocated"
                    )

        # Final cleanup
        cleanup_memory(self.device)
        final_memory = get_memory_stats(self.device)
        if self.device.type == "cuda":
            logger.info(
                f"Final GPU memory: {final_memory['allocated_gb']:.2f}GB allocated, "
                f"{final_memory['free_gb']:.2f}GB free"
            )

        # Get timing data
        timings = timer.get_timings()
        processing_time = timings.get("total", 0.0)

        # Create batch metadata
        hardware_info = get_hardware_info(self.device)
        hardware_info.update(
            {
                "initial_memory_gb": initial_memory.get("allocated_gb", 0.0),
                "final_memory_gb": final_memory.get("allocated_gb", 0.0),
                "peak_memory_gb": final_memory.get("reserved_gb", 0.0),
            }
        )

        batch_metadata = BatchMetadata(
            batch_id=str(uuid.uuid4()),
            total_images=len(image_paths),
            processed_images=success_count,
            failed_images=error_count,
            processing_stages={
                "detection_time": timings.get("detection", 0.0),
                "body_embedding_time": timings.get("body_embedding", 0.0),
            },
            model_versions={
                "detector": self.detector_backend,
                "body_model": self.model_name,
            },
            hardware_info=hardware_info,
        )

        # Create batch result
        batch_result = {
            "results": results,
            "batch_metadata": {
                "batch_id": batch_metadata.batch_id,
                "total_images": batch_metadata.total_images,
                "processed_images": batch_metadata.processed_images,
                "failed_images": batch_metadata.failed_images,
                "processing_stages": batch_metadata.processing_stages,
                "model_versions": batch_metadata.model_versions,
                "hardware_info": batch_metadata.hardware_info,
            },
            "processing_time": processing_time,
            "success_count": success_count,
            "error_count": error_count,
        }

        logger.info(
            f"Batch processing complete: {success_count}/{len(image_paths)} succeeded, "
            f"time={processing_time:.3f}s"
        )

        return batch_result

    def _get_model_info(self) -> dict[str, Any]:
        """Get model information for result metadata."""
        return {
            "model_name": self.model_name,
            "feature_dim": self.embedding_pipeline.feature_dim,
            "device": str(self.device),
        }
