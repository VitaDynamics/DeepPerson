"""
DeepPerson API Facade

Main public interface for the DeepPerson library, providing methods for:
- represent: Generate person embeddings from images
- verify: Compare two images for identity verification
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from .detectors import DetectorFactory, PersonDetector
from .embeddings import BodyEmbeddingGenerator
from .entities import PersonEmbedding
from .registry import get_registry
from .distance import compute_distance
from .fusion import FusionScorer
from .utils import select_device

logger = logging.getLogger(__name__)


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
        device: Optional[Union[str, torch.device]] = None,
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

    def represent(
        self,
        img_path: Union[str, Path, List[Union[str, Path]]],
        detector_backend: Optional[str] = None,
        normalization: Literal["base", "resnet", "circle"] = "resnet",
        batch_size: int = 16,
        confidence_threshold: float = 0.5,
        generate_face_embeddings: bool = False,
        face_model_name: str = "Facenet",
        face_detector_backend: str = "opencv",
        return_multi_modal: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate person embeddings from image(s) with optional multi-modal support.

        Detects persons in each image and generates embeddings for all detected subjects.
        Optionally generates face embeddings for multi-modal fusion capabilities.

        Args:
            img_path: Path to image file or list of paths
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
            >>> # Basic body-only embeddings
            >>> result = dp.represent("person.jpg")
            >>> print(f"Detected {len(result['subjects'])} person(s)")
            >>>
            >>> # Multi-modal body+face embeddings
            >>> result = dp.represent(
            ...     "person.jpg",
            ...     generate_face_embeddings=True,
            ...     face_model_name="Facenet"
            ... )
            >>> for subject in result["subjects"]:
            ...     body_emb = subject["embedding"]  # Body embedding
            ...     face_emb = subject.get("face_embedding")  # Face embedding (if detected)
            ...     print(f"Body: {body_emb.shape}, Face: {face_emb.shape if face_emb is not None else 'None'}")
            >>>
            >>> # Batch processing with multi-modal
            >>> result = dp.represent(
            ...     ["img1.jpg", "img2.jpg"],
            ...     generate_face_embeddings=True
            ... )
        """
        # Normalize input to list
        if isinstance(img_path, (str, Path)):
            img_paths = [Path(img_path)]
            single_image = True
        else:
            img_paths = [Path(p) for p in img_path]
            single_image = False

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
            except ImportError as e:
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
        for image_path in img_paths:
            # Validate image exists
            if not image_path.exists():
                warning_msg = f"Image not found: {image_path}"
                logger.warning(warning_msg)
                warnings_list.append(warning_msg)
                continue

            # Detect persons in image
            logger.debug(f"Processing image: {image_path}")
            detections = detector.detect(
                image=image_path, confidence_threshold=confidence_threshold
            )

            # Handle no detections
            if len(detections) == 0:
                warning_msg = f"No person detected in {image_path.name}"
                logger.warning(warning_msg)
                warnings_list.append(warning_msg)
                continue

            logger.debug(f"Detected {len(detections)} person(s) in {image_path.name}")

            # Crop detected persons
            cropped_persons = detector.crop_persons(
                image=image_path, detections=detections
            )

            # Prepare for batch embedding generation
            bboxes = [det.bbox for det in detections]
            confidences = [det.confidence for det in detections]
            source_ids = [str(image_path)] * len(detections)

            # Generate body embeddings (batch processing within image)
            body_embeddings: List[PersonEmbedding] = (
                self.embedding_pipeline.generate_embeddings_batch(
                    images=cropped_persons,
                    bboxes=bboxes,
                    confidences=confidences,
                    normalize_method=normalization,
                    source_image_ids=source_ids,
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
                        source_image_ids=source_ids,
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
            f"Processed {len(img_paths)} image(s), "
            f"generated {len(all_subjects)} embedding(s) "
            f"({multi_modal_count} multi-modal), "
            f"{len(warnings_list)} warning(s)"
        )

        return response

    def verify(
        self,
        img1_path: Union[str, Path],
        img2_path: Union[str, Path],
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = "cosine",
        threshold: Optional[float] = None,
        normalization: Literal["base", "resnet", "circle"] = "resnet",
        enforce_detection: bool = True,
    ) -> Dict[str, Any]:
        """
        Verify if two images show the same person.

        Compares embeddings from two images and determines if they represent
        the same individual based on distance metrics and thresholds.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image
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
            FileNotFoundError: If image files not found

        Examples:
            >>> result = dp.verify("person1_img1.jpg", "person1_img2.jpg")
            >>> if result["verified"]:
            ...     print(f"Same person! Distance: {result['distance']:.4f}")
            >>> else:
            ...     print(f"Different persons. Distance: {result['distance']:.4f}")
            >>>
            >>> # Use different metric
            >>> result = dp.verify("img1.jpg", "img2.jpg", distance_metric="euclidean")
        """
        logger.info(
            f"Verifying images: {Path(img1_path).name} vs {Path(img2_path).name} "
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
                face_distance = compute_distance(face_embedding1, face_embedding2, metric=distance_metric)

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
            "face_distance": float(face_distance) if face_distance is not None else None,
            "fusion_score": float(fusion_score) if fusion_score is not None else None,
            "face_weight": fusion_metadata.get("face_weight", 0.5) if used_fusion else 0.5,
            "body_weight": fusion_metadata.get("body_weight", 0.5) if used_fusion else 0.5,
            "used_fusion": used_fusion,
            "modality_available": modality_available,
        }

        # Add warnings if any
        if warnings_list:
            response["warnings"] = warnings_list

        return response
