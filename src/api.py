"""
DeepPerson API Facade

Main public interface for the DeepPerson library, providing methods for:
- represent: Generate person embeddings from images
- verify: Compare two embeddings for identity verification 
- find: Search a gallery for matching persons 
- build_gallery: Create a searchable gallery from embeddings 
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch

from .detectors import DetectorFactory, PersonDetector
from .embeddings import BodyEmbeddingGenerator
from .entities import PersonEmbedding
from .registry import get_registry
from .search import compute_distance
from .fusion import FusionScorer
from .utils import select_device

logger = logging.getLogger(__name__)


class DeepPerson:
    """
    Main facade for DeepPerson functionality.

    Provides high-level methods for person re-identification tasks including
    embedding generation, verification, and gallery search.

    Examples:
        >>> from components.deep_person.api import DeepPerson
        >>> dp = DeepPerson(model_name="resnet50_circle_dg")
        >>>
        >>> # Generate embeddings for persons in image
        >>> result = dp.represent("image.jpg")
        >>>
        >>> # Access embeddings
        >>> for subject in result["subjects"]:
        ...     print(f"Embedding shape: {subject['embedding'].shape}")
        ...     print(f"Confidence: {subject['metadata']['confidence']}")
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

        # Lazy initialization caches (Phase 6: Optimization)
        self._face_generator_cache: dict[str, Any] = {}  # Key: (model_name, detector_backend)
        self._storage_manager_cache: dict[str, Any] = {}  # Key: storage_path
        self._embedding_service_cache: dict[str, Any] = {}  # Key: storage_path
        self._gallery_api_cache: dict[str, Any] = {}  # Key: storage_path

        logger.info("DeepPerson initialized successfully")

    # ==================== Cached Component Access (Phase 6: Optimization) ====================

    def _get_face_generator(
        self, model_name: str = "Facenet", detector_backend: str = "opencv"
    ):
        """
        Get or create a cached face embedding generator.

        Args:
            model_name: Face recognition model name
            detector_backend: Face detector backend

        Returns:
            Cached or newly created FaceEmbeddingGenerator

        Examples:
            >>> dp = DeepPerson()
            >>> gen1 = dp._get_face_generator("Facenet", "opencv")
            >>> gen2 = dp._get_face_generator("Facenet", "opencv")
            >>> assert gen1 is gen2  # Same instance returned
        """
        cache_key = (model_name, detector_backend)

        if cache_key not in self._face_generator_cache:
            from .face_embeddings import FaceEmbeddingGenerator

            self._face_generator_cache[cache_key] = FaceEmbeddingGenerator(
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
            logger.debug(
                f"Created and cached face generator: {model_name}/{detector_backend}"
            )
        else:
            logger.debug(f"Using cached face generator: {model_name}/{detector_backend}")

        return self._face_generator_cache[cache_key]

    def _get_storage_manager(self, storage_path: Union[str, Path]):
        """
        Get or create a cached storage manager.

        Args:
            storage_path: Path to gallery storage directory

        Returns:
            Cached or newly created GalleryStorageManager

        Examples:
            >>> dp = DeepPerson()
            >>> mgr1 = dp._get_storage_manager("galleries/")
            >>> mgr2 = dp._get_storage_manager("galleries/")
            >>> assert mgr1 is mgr2  # Same instance returned
        """
        from .user_gallery.storage import GalleryStorageManager

        storage_path_str = str(Path(storage_path))

        if storage_path_str not in self._storage_manager_cache:
            self._storage_manager_cache[storage_path_str] = GalleryStorageManager(
                Path(storage_path_str)
            )
            logger.debug(f"Created and cached storage manager: {storage_path_str}")
        else:
            logger.debug(f"Using cached storage manager: {storage_path_str}")

        return self._storage_manager_cache[storage_path_str]

    def _get_embedding_service(self, storage_path: Union[str, Path]):
        """
        Get or create a cached embedding generation service.

        Args:
            storage_path: Path to gallery storage directory

        Returns:
            Cached or newly created EmbeddingGenerationService

        Examples:
            >>> dp = DeepPerson()
            >>> svc1 = dp._get_embedding_service("galleries/")
            >>> svc2 = dp._get_embedding_service("galleries/")
            >>> assert svc1 is svc2  # Same instance returned
        """
        from .user_gallery.services import EmbeddingGenerationService

        storage_path_str = str(Path(storage_path))

        if storage_path_str not in self._embedding_service_cache:
            storage_manager = self._get_storage_manager(storage_path_str)
            self._embedding_service_cache[storage_path_str] = EmbeddingGenerationService(
                storage_manager=storage_manager,
                model_name=self.model_name,
                device=str(self.device),
            )
            logger.debug(f"Created and cached embedding service: {storage_path_str}")
        else:
            logger.debug(f"Using cached embedding service: {storage_path_str}")

        return self._embedding_service_cache[storage_path_str]

    def _get_gallery_api(self, storage_path: Union[str, Path]):
        """
        Get or create a cached user gallery API.

        Args:
            storage_path: Path to gallery storage directory

        Returns:
            Cached or newly created _UserGalleryAPI (internal)

        Examples:
            >>> dp = DeepPerson()
            >>> api1 = dp._get_gallery_api("galleries/")
            >>> api2 = dp._get_gallery_api("galleries/")
            >>> assert api1 is api2  # Same instance returned
        """
        from .user_gallery.api import _UserGalleryAPI

        storage_path_str = str(Path(storage_path))

        if storage_path_str not in self._gallery_api_cache:
            self._gallery_api_cache[storage_path_str] = _UserGalleryAPI(
                storage_path=storage_path_str
            )
            logger.debug(f"Created and cached gallery API: {storage_path_str}")
        else:
            logger.debug(f"Using cached gallery API: {storage_path_str}")

        return self._gallery_api_cache[storage_path_str]

    def clear_cache(self):
        """
        Clear all cached components.

        Useful when you want to free memory or force recreation of components
        with different parameters.

        Examples:
            >>> dp = DeepPerson()
            >>> dp.represent("img.jpg", generate_face_embeddings=True)
            >>> dp.clear_cache()  # Free face generator from memory
        """
        self._face_generator_cache.clear()
        self._storage_manager_cache.clear()
        self._embedding_service_cache.clear()
        self._gallery_api_cache.clear()
        logger.info("Cleared all component caches")

    # ==================== Public API Methods ====================

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
            generate_face_embeddings: Whether to generate face embeddings (NEW)
            face_model_name: Face recognition model to use (default: "Facenet") (NEW)
            face_detector_backend: Face detection backend (default: "opencv") (NEW)
            return_multi_modal: Return multi-modal PersonEmbedding objects (NEW)

        Returns:
            Dictionary containing:
                - subjects: List of detected persons with embeddings and metadata
                - warnings: List of warning messages (e.g., no detection)
                - model_info: Model and hardware information
                - face_model_info: Face model information (if generate_face_embeddings=True)

        Examples:
            >>> # Basic body-only embeddings (backward compatible)
            >>> result = dp.represent("person.jpg")
            >>> print(f"Detected {len(result['subjects'])} person(s)")
            >>>
            >>> # Multi-modal body+face embeddings (NEW)
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

        # Get or create cached face embedding generator if needed
        face_generator = None
        if generate_face_embeddings:
            try:
                # Use cached face generator (Phase 6: Optimization)
                face_generator = self._get_face_generator(
                    model_name=face_model_name,
                    detector_backend=face_detector_backend
                )
                logger.info(f"Face embedding generator ready: {face_model_name}")
            except ImportError as e:
                warning_msg = (
                    "DeepFace not available, face embeddings will be skipped. "
                    "Install with: pip install deepface"
                )
                logger.warning(warning_msg)
                warnings_list = [warning_msg]
                generate_face_embeddings = False

        # Results containers
        all_subjects = []
        warnings_list = warnings_list if generate_face_embeddings and not face_generator else []

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
            "face_weight": fusion_metadata.get("face_weight", 0.5) if used_fusion else 0.5,  # Actual weights used
            "body_weight": fusion_metadata.get("body_weight", 0.5) if used_fusion else 0.5,  # Actual weights used
            "used_fusion": used_fusion,
            "modality_available": modality_available,
        }

        # Add warnings if any
        if warnings_list:
            response["warnings"] = warnings_list

        return response

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
        log_level: str = "info",
        reload: bool = False,
        gallery_storage_path: Union[str, Path] = "galleries/",
        enable_cors: bool = True,
    ) -> None:
        """
        Start FastAPI server with DeepPerson API endpoints.

        This method provides a convenient way to serve all DeepPerson functionality
        through HTTP endpoints using FastAPI and Uvicorn.

        Args:
            host: Host to bind to (default: "0.0.0.0")
            port: Port to listen on (default: 8000)
            workers: Number of worker processes (default: 1)
            log_level: Logging level (default: "info")
            reload: Enable auto-reload for development (default: False)
            gallery_storage_path: Path to gallery storage directory
            enable_cors: Enable CORS middleware (default: True)

        Examples:
            >>> dp = DeepPerson()
            >>> # Start server with default settings
            >>> dp.serve()
            >>>
            >>> # Start server with custom settings
            >>> dp.serve(
            ...     host="127.0.0.1",
            ...     port=9000,
            ...     log_level="debug",
            ...     workers=2
            ... )
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to serve the API. "
                "Install with: pip install uvicorn"
            )

        logger.info(
            f"Starting DeepPerson FastAPI server on {host}:{port} "
            f"(workers={workers}, reload={reload})"
        )

        # Set environment variables for configuration
        import os
        os.environ["DEEPPERSON_HOST"] = host
        os.environ["DEEPPERSON_PORT"] = str(port)
        os.environ["DEEPPERSON_WORKERS"] = str(workers)
        os.environ["DEEPPERSON_LOG_LEVEL"] = log_level
        os.environ["DEEPPERSON_GALLERY_STORAGE_PATH"] = str(gallery_storage_path)
        os.environ["DEEPPERSON_ENABLE_CORS"] = str(enable_cors)

        # Run uvicorn server
        try:
            # Import the FastAPI app factory
            try:
                from fastapi_service.app import create_app
                app = create_app()
            except ImportError:
                logger.error(
                    "FastAPI service not found. Make sure fastapi_service module is available."
                )
                raise

            # Start the server
            uvicorn.run(
                app,
                host=host,
                port=port,
                workers=workers if not reload else 1,  # Reload only works with 1 worker
                log_level=log_level.lower(),
                reload=reload,
                access_log=True,
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server failed to start: {e}", exc_info=True)
            raise

    # ==================== User Gallery Methods ====================

    def create_gallery(
        self,
        user_id: str,
        image_paths: list[Union[str, Path]],
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        modality_hints: Optional[dict[str, str]] = None,
        enable_clustering: bool = True,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Create a new user gallery with multiple images.

        This method creates a user gallery that can contain multiple body and face
        images, automatically organized into appearance variants.

        Args:
            user_id: Unique identifier for the user
            image_paths: List of paths to user images
            name: Optional display name for the user
            metadata: Optional metadata dictionary
            modality_hints: Optional dict mapping image paths to modality ("BODY", "FACE")
            enable_clustering: Whether to automatically cluster images
            gallery_storage_path: Optional custom storage path (default: "galleries/")

        Returns:
            Dictionary containing:
                - user_id: User identifier
                - status: Gallery status
                - total_images: Number of images processed
                - clusters_created: Number of variant clusters
                - modality_breakdown: Count of images by modality

        Raises:
            ValueError: If user_id already exists or validation fails

        Examples:
            >>> dp = DeepPerson()
            >>> result = dp.create_gallery(
            ...     user_id="user_001",
            ...     image_paths=["body1.jpg", "body2.jpg", "face1.jpg"],
            ...     name="John Doe",
            ...     modality_hints={
            ...         "body1.jpg": "BODY",
            ...         "body2.jpg": "BODY",
            ...         "face1.jpg": "FACE"
            ...     }
            ... )
            >>> print(f"Created gallery with {result['total_images']} images")
        """
        # Get cached gallery API (Phase 6: Optimization)
        gallery_api = self._get_gallery_api(gallery_storage_path)

        # Create gallery
        result = gallery_api.create_gallery(
            user_id=user_id,
            image_paths=image_paths,
            name=name,
            metadata=metadata,
            modality_hints=modality_hints,
            enable_clustering=enable_clustering,
        )

        logger.info(f"Created user gallery for '{user_id}' via DeepPerson API")
        return result

    def get_gallery(
        self,
        user_id: str,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Get information about a user gallery.

        Args:
            user_id: User identifier
            gallery_storage_path: Optional custom storage path

        Returns:
            Dictionary with comprehensive gallery information

        Raises:
            ValueError: If gallery doesn't exist

        Examples:
            >>> dp = DeepPerson()
            >>> gallery_info = dp.get_gallery("user_001")
            >>> print(f"Gallery has {gallery_info['total_images']} images")
        """
        # Get cached gallery API (Phase 6: Optimization)
        gallery_api = self._get_gallery_api(gallery_storage_path)

        return gallery_api.get_gallery(user_id)

    def list_galleries(
        self,
        status_filter: Optional[str] = None,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> list[dict[str, Any]]:
        """
        List all user galleries.

        Args:
            status_filter: Optional status to filter by ("ACTIVE", "INACTIVE", "PENDING_VERIFICATION")
            gallery_storage_path: Optional custom storage path

        Returns:
            List of gallery summary dictionaries

        Examples:
            >>> dp = DeepPerson()
            >>> galleries = dp.list_galleries(status_filter="ACTIVE")
            >>> print(f"Found {len(galleries)} active galleries")
        """
        # Get cached gallery API (Phase 6: Optimization)
        gallery_api = self._get_gallery_api(gallery_storage_path)

        return gallery_api.list_galleries(status_filter=status_filter)

    def delete_gallery(
        self,
        user_id: str,
        permanent: bool = False,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Delete a user gallery.

        Args:
            user_id: User identifier
            permanent: If True, permanently delete; if False, mark as inactive
            gallery_storage_path: Optional custom storage path

        Returns:
            Dictionary with deletion status

        Examples:
            >>> dp = DeepPerson()
            >>> result = dp.delete_gallery("user_001", permanent=False)
            >>> print(f"Gallery deleted: {result['success']}")
        """
        # Get cached gallery API (Phase 6: Optimization)
        gallery_api = self._get_gallery_api(gallery_storage_path)

        return gallery_api.delete_gallery(user_id, permanent=permanent)

    def update_gallery(
        self,
        user_id: str,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        status: Optional[str] = None,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Update gallery metadata and status.

        Args:
            user_id: User identifier
            name: Optional new display name
            metadata: Optional metadata to merge with existing
            status: Optional new status ("ACTIVE", "INACTIVE", "PENDING_VERIFICATION")
            gallery_storage_path: Optional custom storage path

        Returns:
            Updated gallery information

        Raises:
            ValueError: If gallery doesn't exist or invalid status

        Examples:
            >>> dp = DeepPerson()
            >>> result = dp.update_gallery(
            ...     user_id="user_001",
            ...     name="John Smith",
            ...     status="ACTIVE"
            ... )
        """
        gallery_api = self._get_gallery_api(gallery_storage_path)
        return gallery_api.update_gallery(
            user_id=user_id, name=name, metadata=metadata, status=status
        )

    def add_images(
        self,
        user_id: str,
        image_paths: list[Union[str, Path]],
        modality_hints: Optional[dict[str, str]] = None,
        recluster: bool = False,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Add new images to an existing gallery.

        Args:
            user_id: User identifier
            image_paths: List of new image paths
            modality_hints: Optional modality hints for new images
            recluster: Whether to recluster after adding images
            gallery_storage_path: Optional custom storage path

        Returns:
            Dictionary with update information

        Raises:
            ValueError: If gallery doesn't exist or validation fails

        Examples:
            >>> dp = DeepPerson()
            >>> result = dp.add_images(
            ...     user_id="user_001",
            ...     image_paths=["new_body.jpg", "new_face.jpg"],
            ...     modality_hints={"new_body.jpg": "BODY", "new_face.jpg": "FACE"}
            ... )
            >>> print(f"Added {result['images_added']} images")
        """
        gallery_api = self._get_gallery_api(gallery_storage_path)
        return gallery_api.add_images(
            user_id=user_id,
            image_paths=image_paths,
            modality_hints=modality_hints,
            recluster=recluster,
        )

    def gallery_exists(
        self,
        user_id: str,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> bool:
        """
        Check if a gallery exists for the given user.

        Args:
            user_id: User identifier
            gallery_storage_path: Optional custom storage path

        Returns:
            True if gallery exists, False otherwise

        Examples:
            >>> dp = DeepPerson()
            >>> if dp.gallery_exists("user_001"):
            ...     print("Gallery exists")
        """
        gallery_api = self._get_gallery_api(gallery_storage_path)
        return gallery_api.gallery_exists(user_id)

    def recluster_gallery(
        self,
        user_id: str,
        algorithm: Optional[str] = None,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Recluster an existing gallery.

        Requires embeddings to be generated first.

        Args:
            user_id: User identifier
            algorithm: Optional clustering algorithm override
            gallery_storage_path: Optional custom storage path

        Returns:
            Dictionary with clustering information

        Raises:
            ValueError: If gallery doesn't exist or no embeddings found

        Examples:
            >>> dp = DeepPerson()
            >>> result = dp.recluster_gallery("user_001", algorithm="DBSCAN")
            >>> print(f"Created {result['clusters_created']} clusters")
        """
        gallery_api = self._get_gallery_api(gallery_storage_path)
        return gallery_api.recluster_gallery(user_id=user_id, algorithm=algorithm)

    def represent_gallery(
        self,
        user_id: str,
        generate_face_embeddings: bool = False,
        force_regenerate: bool = False,
        batch_size: int = 16,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Generate embeddings for all images in a user gallery.

        This method extends the standard represent() functionality to work with
        user galleries, supporting both body and face embedding generation with
        automatic batch processing and error handling.

        Args:
            user_id: User identifier
            generate_face_embeddings: Whether to generate face embeddings using DeepFace
            force_regenerate: Force regeneration of existing embeddings
            batch_size: Batch size for embedding generation
            gallery_storage_path: Optional custom storage path

        Returns:
            Dictionary containing:
                - user_id: User identifier
                - processed_images: Number of images processed
                - generated_embeddings: Number of embeddings generated
                - face_embeddings_generated: Number of face embeddings generated
                - processing_time_ms: Total processing time in milliseconds
                - errors: List of errors (if any)
                - average_quality_score: Average embedding quality score

        Raises:
            ValueError: If gallery doesn't exist

        Examples:
            >>> dp = DeepPerson()
            >>> # Generate body embeddings only
            >>> result = dp.represent_gallery("user_001")
            >>> print(f"Generated {result['generated_embeddings']} embeddings")
            >>>
            >>> # Generate both body and face embeddings
            >>> result = dp.represent_gallery(
            ...     user_id="user_001",
            ...     generate_face_embeddings=True,
            ...     batch_size=32
            ... )
            >>> print(f"Face embeddings: {result['face_embeddings_generated']}")
            >>>
            >>> # Force regeneration of all embeddings
            >>> result = dp.represent_gallery(
            ...     user_id="user_001",
            ...     generate_face_embeddings=True,
            ...     force_regenerate=True
            ... )
        """
        logger.info(
            f"Generating embeddings for gallery '{user_id}' "
            f"(face_embeddings={generate_face_embeddings}, force={force_regenerate})"
        )

        # Get cached components (Phase 6: Optimization)
        embedding_service = self._get_embedding_service(gallery_storage_path)

        # Generate embeddings
        embeddings, generation_info = embedding_service.generate_embeddings_for_gallery(
            user_id=user_id,
            generate_face_embeddings=generate_face_embeddings,
            force_regenerate=force_regenerate,
            batch_size=batch_size,
        )

        # Calculate average quality score
        avg_quality = (
            sum(emb.quality_score for emb in embeddings) / len(embeddings)
            if len(embeddings) > 0
            else 0.0
        )

        # Prepare response
        response = {
            "user_id": generation_info["user_id"],
            "processed_images": generation_info["processed_images"],
            "generated_embeddings": generation_info["generated_embeddings"],
            "face_embeddings_generated": generation_info["face_embeddings_generated"],
            "processing_time_ms": generation_info["processing_time_ms"],
            "errors": generation_info.get("errors"),
            "average_quality_score": avg_quality,
            "force_regenerate": force_regenerate,
            "batch_size": batch_size,
        }

        logger.info(
            f"Embedding generation complete for gallery '{user_id}': "
            f"{generation_info['generated_embeddings']} embeddings generated"
        )

        return response

    def get_gallery_embedding_stats(
        self,
        user_id: str,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Get embedding statistics for a user gallery.

        Args:
            user_id: User identifier
            gallery_storage_path: Optional custom storage path

        Returns:
            Dictionary with embedding statistics including:
                - total_images: Total number of images
                - total_embeddings: Total number of embeddings
                - face_embeddings: Number of face embeddings
                - body_only_embeddings: Number of body-only embeddings
                - average_quality_score: Average embedding quality
                - processing_status: Breakdown by processing status
                - coverage: Embedding coverage ratio

        Raises:
            ValueError: If gallery doesn't exist

        Examples:
            >>> dp = DeepPerson()
            >>> stats = dp.get_gallery_embedding_stats("user_001")
            >>> print(f"Coverage: {stats['coverage']:.1%}")
            >>> print(f"Average quality: {stats['average_quality_score']:.3f}")
        """
        # Get cached embedding service (Phase 6: Optimization)
        embedding_service = self._get_embedding_service(gallery_storage_path)

        return embedding_service.get_embedding_statistics(user_id)

    def retrieve_from_gallery(
        self,
        probe_image_path: Union[str, Path],
        gallery_name: str = "user_gallery",
        top_k: int = 10,
        min_score: float = 0.0,
        generate_face_embeddings: bool = True,
        fusion_weights: Optional[dict[str, float]] = None,
        include_evidence: bool = True,
        gallery_storage_path: Union[str, Path] = "galleries/",
    ) -> dict[str, Any]:
        """
        Retrieve users from a user gallery using fusion-based similarity search.

        This method performs multi-modal person retrieval by:
        1. Processing the probe image to generate body and optional face embeddings
        2. Searching the user gallery using fusion scoring
        3. Ranking results by overall similarity score
        4. Providing evidence and explanations for matches

        Args:
            probe_image_path: Path to probe image for retrieval
            gallery_name: Name of the gallery to search (default: "user_gallery")
            top_k: Number of top results to return (default: 10)
            min_score: Minimum fusion score threshold (0.0-1.0, default: 0.0)
            generate_face_embeddings: Whether to generate face embeddings (default: True)
            fusion_weights: Optional custom fusion weights dict with 'face' and 'body' keys
            include_evidence: Whether to include evidence images in results (default: True)
            gallery_storage_path: Optional custom storage path (default: "galleries/")

        Returns:
            Dictionary containing:
                - results: List of retrieval results with user IDs, scores, and evidence
                - probe_id: Unique probe identifier
                - processing_time_ms: Total processing time in milliseconds
                - fusion_weights: Applied fusion weights
                - total_results: Number of results returned
                - face_embedding_used: Whether face embeddings were used

        Raises:
            FileNotFoundError: If probe image or gallery doesn't exist
            ValueError: If no person detected in probe image

        Examples:
            >>> dp = DeepPerson()
            >>> # Basic retrieval
            >>> results = dp.retrieve_from_gallery("probe.jpg", top_k=5)
            >>> for result in results['results']:
            ...     print(f"{result['user_id']}: {result['overall_score']:.3f}")
            >>>
            >>> # With custom fusion weights (emphasize face)
            >>> results = dp.retrieve_from_gallery(
            ...     "probe.jpg",
            ...     fusion_weights={"face": 0.7, "body": 0.3},
            ...     min_score=0.5
            ... )
            >>>
            >>> # Body-only retrieval
            >>> results = dp.retrieve_from_gallery(
            ...     "probe.jpg",
            ...     generate_face_embeddings=False
            ... )
        """
        from .user_gallery.fusion import FusionRetrievalService
        from .user_gallery.search import MultiModalSearcher
        from .user_gallery.utils import format_retrieval_results_for_display

        logger.info(
            f"Starting user gallery retrieval: probe={Path(probe_image_path).name}, "
            f"gallery={gallery_name}, top_k={top_k}"
        )

        # Initialize storage path
        gallery_dir = Path(gallery_storage_path) / gallery_name

        # Check if gallery exists
        if not gallery_dir.exists():
            raise FileNotFoundError(
                f"Gallery '{gallery_name}' not found at {gallery_dir}. "
                f"Please create a gallery first using create_gallery() and represent_gallery()."
            )

        # Get cached face embedding generator if needed (Phase 6: Optimization)
        face_generator = None
        if generate_face_embeddings:
            try:
                face_generator = self._get_face_generator(
                    model_name="Facenet", detector_backend="opencv"
                )
                logger.debug("Using core face embedding generator")
            except ImportError:
                logger.warning(
                    "DeepFace not available, face embeddings will be skipped. "
                    "Install with: pip install deepface"
                )
                generate_face_embeddings = False

        # Initialize fusion retrieval service
        retrieval_service = FusionRetrievalService(
            body_embedding_generator=self.embedding_pipeline,
            face_embedding_generator=face_generator,
            default_face_weight=fusion_weights.get("face", 0.5)
            if fusion_weights
            else 0.5,
            default_body_weight=fusion_weights.get("body", 0.5)
            if fusion_weights
            else 0.5,
        )

        # Load gallery searcher (now using dynamic dimensions from models)
        try:
            # Get dimensions from embedding generators
            body_dim = self.embedding_pipeline.feature_dim  # Dynamic from model profile
            face_dim = face_generator.feature_dim if generate_face_embeddings and face_generator else None

            searcher = MultiModalSearcher(
                body_dimension=body_dim,  # Dynamic from embedding pipeline
                face_dimension=face_dim,  # Dynamic from face generator
                metric="cosine",
                device=str(self.device),
            )
            searcher.load(gallery_dir)
            logger.info(
                f"Loaded gallery '{gallery_name}': {searcher.get_user_count()} users "
                f"(body_dim={body_dim}, face_dim={face_dim})"
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load gallery '{gallery_name}': {e}. "
                f"Ensure the gallery has been properly created and embeddings generated."
            ) from e

        # Perform retrieval
        try:
            results, retrieval_metadata = retrieval_service.retrieve_users(
                probe_image_path=probe_image_path,
                gallery_searcher=searcher,
                top_k=top_k,
                min_score=min_score,
                generate_face_embedding=generate_face_embeddings,
                fusion_weights=fusion_weights,
                include_evidence=include_evidence,
            )

            # Format results for display
            formatted_results = format_retrieval_results_for_display(
                results, include_explanations=include_evidence, max_evidence_images=3
            )

            # Build response
            response = {
                "results": formatted_results,
                "probe_id": retrieval_metadata["probe_id"],
                "probe_image": str(probe_image_path),
                "processing_time_ms": retrieval_metadata["processing_time_ms"],
                "fusion_weights": retrieval_metadata["fusion_weights"],
                "total_results": len(formatted_results),
                "face_embedding_used": retrieval_metadata["face_embedding_used"],
                "body_confidence": retrieval_metadata["body_confidence"],
                "face_confidence": retrieval_metadata.get("face_confidence", 0.0),
                "gallery_info": {
                    "name": gallery_name,
                    "path": str(gallery_dir),
                    "total_users": searcher.get_user_count(),
                },
                "model_info": {
                    "name": self.model_name,
                    "device": str(self.device),
                    "detector_backend": self.detector_backend,
                },
            }

            # Add warnings if any
            if retrieval_metadata.get("processing_errors"):
                response["warnings"] = retrieval_metadata["processing_errors"]

            logger.info(
                f"Retrieval completed: {len(formatted_results)} results in "
                f"{retrieval_metadata['processing_time_ms']:.1f}ms"
            )

            return response

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise
