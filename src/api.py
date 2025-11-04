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
from .embeddings import EmbeddingPipeline
from .entities import PersonEmbedding
from .registry import get_registry
from .search import compute_distance
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
        self.embedding_pipeline = EmbeddingPipeline(
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
    ) -> Dict[str, Any]:
        """
        Generate person embeddings from image(s).

        Detects persons in each image and generates embeddings for all detected subjects.

        Args:
            img_path: Path to image file or list of paths
            detector_backend: Override detector backend (None to use default)
            normalization: Normalization method for embeddings ('base', 'resnet', 'circle')
            batch_size: Batch size for embedding generation
            confidence_threshold: Minimum detection confidence threshold

        Returns:
            Dictionary containing:
                - subjects: List of detected persons with embeddings and metadata
                - warnings: List of warning messages (e.g., no detection)
                - model_info: Model and hardware information

        Examples:
            >>> result = dp.represent("person.jpg")
            >>> print(f"Detected {len(result['subjects'])} person(s)")
            >>>
            >>> # Batch processing
            >>> result = dp.represent(["img1.jpg", "img2.jpg", "img3.jpg"])
            >>>
            >>> # Access embeddings
            >>> for subject in result["subjects"]:
            ...     embedding = subject["embedding"]
            ...     bbox = subject["metadata"]["bbox"]
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

        # Results containers
        all_subjects = []
        warnings_list = []

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

            # Generate embeddings (batch processing within image)
            embeddings: List[PersonEmbedding] = (
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

            # Package subjects
            for embedding in embeddings:
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
                    },
                }
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

        logger.info(
            f"Processed {len(img_paths)} image(s), "
            f"generated {len(all_subjects)} embedding(s), "
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

        # Extract facial areas for response
        facial_area1 = result1["subjects"][0]["metadata"]["bbox"]
        facial_area2 = result2["subjects"][0]["metadata"]["bbox"]

        # Compute distance between embeddings
        distance = compute_distance(embedding1, embedding2, metric=distance_metric)

        # Get threshold (use provided or fetch from registry)
        if threshold is None:
            threshold = self.registry.get_verification_threshold(
                effective_model, distance_metric
            )
            logger.debug(f"Using default threshold from registry: {threshold}")

        # Determine verification result
        # Lower distance = more similar
        # Verified if distance <= threshold
        verified = bool(distance <= threshold)

        logger.info(
            f"Verification result: {verified} "
            f"(distance={distance:.4f}, threshold={threshold:.4f})"
        )

        # Build response following DeepFace pattern
        response = {
            "verified": verified,
            "distance": float(distance),
            "threshold": float(threshold),
            "distance_metric": distance_metric,
            "model": effective_model,
            "detector_backend": detector_backend or self.detector_backend,
            "facial_areas": {"img1": facial_area1, "img2": facial_area2},
        }

        # Add warnings if any
        if warnings_list:
            response["warnings"] = warnings_list

        return response

    def find(
        self,
        img_path: Union[str, Path],
        gallery_path: Union[str, Path],
        top_k: int = 5,
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = "cosine",
        threshold: Optional[float] = None,
        gallery_name: str = "gallery",
    ) -> Dict[str, Any]:
        """
        Find the best matches for a person in a gallery database.

        Args:
            img_path: Path to query image containing person to search for
            gallery_path: Path to gallery database directory
            top_k: Number of top matches to return
            model_name: Override model name (uses instance default if None)
            detector_backend: Override detector backend (uses instance default if None)
            distance_metric: Distance metric ('cosine', 'euclidean', 'euclidean_l2')
            threshold: Distance threshold for filtering matches (None for no filtering)
            gallery_name: Name of gallery within gallery_path directory

        Returns:
            Dictionary containing:
                - query: Information about query image and embedding
                - matches: List of top-k matches with subject IDs, distances, and metadata
                - gallery_info: Gallery metadata (total entries, metric used, etc.)
                - warnings: List of warnings if any

        Raises:
            FileNotFoundError: If gallery not found
            ValueError: If no person detected in query image

        Examples:
            >>> # Find person in gallery
            >>> result = dp.find("query.jpg", "./galleries/mydb", top_k=10)
            >>> for match in result["matches"]:
            ...     print(f"{match['subject_id']}: distance={match['distance']:.4f}")
            >>>
            >>> # With threshold filtering
            >>> result = dp.find("query.jpg", "./galleries/mydb", threshold=0.30)
        """
        logger.info(f"Searching for person in gallery: {gallery_path}/{gallery_name}")

        # Load gallery
        from .utils import load_gallery_index

        try:
            searcher = load_gallery_index(
                gallery_dir=Path(gallery_path),
                gallery_name=gallery_name,
                backend="auto",
                device=str(self.device),
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Gallery not found: {e}")

        # Generate embedding for query image
        result = self.represent(
            img_path=img_path,
            detector_backend=detector_backend,
            normalization="resnet",
            confidence_threshold=0.5,
        )

        # Check if person was detected
        if len(result["subjects"]) == 0:
            raise ValueError(
                f"No person detected in query image: {Path(img_path).name}"
            )

        warnings_list = []
        if len(result["subjects"]) > 1:
            warning = f"Multiple persons detected in query image ({len(result['subjects'])}), using first detection"
            logger.warning(warning)
            warnings_list.append(warning)

        # Get query embedding
        query_embedding = result["subjects"][0]["embedding"]
        query_metadata = result["subjects"][0]["metadata"]

        # Search gallery
        search_results = searcher.search(
            query_embedding=query_embedding, k=top_k, threshold=threshold
        )

        # Package results
        matches = []
        for match in search_results:
            matches.append(
                {
                    "subject_id": match["subject_id"],
                    "distance": match["distance"],
                    "metadata": match.get("metadata", {}),
                }
            )

        # Build response
        response = {
            "query": {
                "image_path": str(img_path),
                "bbox": query_metadata["bbox"],
                "confidence": query_metadata["confidence"],
            },
            "matches": matches,
            "gallery_info": {
                "path": str(gallery_path),
                "name": gallery_name,
                "total_entries": len(searcher.subject_ids),
                "metric": searcher.metric,
                "backend": searcher.__class__.__name__,
            },
            "model_info": {
                "name": model_name or self.model_name,
                "device": str(self.device),
                "distance_metric": distance_metric,
            },
        }

        if warnings_list:
            response["warnings"] = warnings_list

        logger.info(
            f"Search completed: {len(matches)} matches found (k={top_k}, threshold={threshold})"
        )

        return response

    def build_gallery(
        self,
        img_paths: List[Union[str, Path]],
        subject_ids: List[str],
        gallery_path: Union[str, Path],
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        batch_size: int = 16,
        normalization: Literal["base", "resnet", "circle"] = "resnet",
        gallery_name: str = "gallery",
        distance_metric: Literal["cosine", "euclidean", "euclidean_l2"] = "cosine",
        backend: str = "auto",
    ) -> Dict[str, Any]:
        """
        Build a searchable gallery database from images.

        Generates embeddings for all images and creates a searchable FAISS or sklearn index.

        Args:
            img_paths: List of image file paths (one per subject)
            subject_ids: List of subject IDs corresponding to images (must match length of img_paths)
            gallery_path: Path where gallery database will be saved
            model_name: Override model name (uses instance default if None)
            detector_backend: Override detector backend (uses instance default if None)
            batch_size: Batch size for embedding generation
            normalization: Normalization method ('base', 'resnet', 'circle')
            gallery_name: Name for the gallery (default: "gallery")
            distance_metric: Distance metric for similarity search ('cosine', 'euclidean', 'euclidean_l2')
            backend: Search backend ('auto', 'faiss', 'sklearn')

        Returns:
            Dictionary containing:
                - gallery_info: Gallery metadata (path, name, total entries, metric, backend)
                - processed: Number of images successfully processed
                - failed: Number of images that failed processing
                - warnings: List of warnings
                - model_info: Model and hardware information

        Raises:
            ValueError: If img_paths and subject_ids lengths don't match
            FileNotFoundError: If image files not found

        Examples:
            >>> # Build gallery from images
            >>> images = ["person1.jpg", "person2.jpg", "person3.jpg"]
            >>> ids = ["person_001", "person_002", "person_003"]
            >>> result = dp.build_gallery(images, ids, "./galleries/mydb")
            >>> print(f"Gallery created with {result['processed']} entries")
            >>>
            >>> # Build with custom settings
            >>> result = dp.build_gallery(
            ...     images, ids, "./galleries/custom",
            ...     distance_metric="euclidean",
            ...     backend="faiss"
            ... )
        """
        logger.info(
            f"Building gallery: {gallery_path}/{gallery_name} with {len(img_paths)} images"
        )

        # Validate inputs
        if len(img_paths) != len(subject_ids):
            raise ValueError(
                f"Length mismatch: img_paths ({len(img_paths)}) != subject_ids ({len(subject_ids)})"
            )

        # Process all images to generate embeddings
        result = self.represent(
            img_path=img_paths,
            detector_backend=detector_backend,
            normalization=normalization,
            batch_size=batch_size,
            confidence_threshold=0.5,
        )

        # Track processing results
        processed_count = 0
        failed_count = 0
        warnings_list = result.get("warnings", []) or []

        # Create searcher instance
        from .search import SearcherFactory

        feature_dim = result["model_info"]["feature_dim"]
        searcher = SearcherFactory.create_searcher(
            backend=backend,
            dimension=feature_dim,
            metric=distance_metric,
            device=str(self.device),
        )

        # Add embeddings to searcher
        if len(result["subjects"]) > 0:
            # Match embeddings to subject IDs
            # If multiple detections per image, take first detection
            for i, subject_id in enumerate(subject_ids):
                if i < len(result["subjects"]):
                    embedding = result["subjects"][i]["embedding"]
                    metadata = result["subjects"][i]["metadata"]

                    searcher.add_embedding(
                        embedding=embedding, subject_id=subject_id, metadata=metadata
                    )
                    processed_count += 1
                else:
                    failed_count += 1
                    warning = f"No embedding generated for subject: {subject_id}"
                    logger.warning(warning)
                    warnings_list.append(warning)

        # Save gallery
        from .utils import save_gallery_index

        gallery_dir = save_gallery_index(
            searcher=searcher,
            gallery_dir=Path(gallery_path),
            gallery_name=gallery_name,
            model_profile_id=model_name or self.model_name,
        )

        # Build response
        response = {
            "gallery_info": {
                "path": str(gallery_dir),
                "name": gallery_name,
                "total_entries": len(searcher.subject_ids),
                "metric": distance_metric,
                "backend": searcher.__class__.__name__,
            },
            "processed": processed_count,
            "failed": failed_count,
            "model_info": result["model_info"],
        }

        if warnings_list:
            response["warnings"] = warnings_list

        logger.info(
            f"Gallery built successfully: {processed_count} processed, "
            f"{failed_count} failed, saved to {gallery_dir}"
        )

        return response

    # ==================== User Gallery Methods ====================

    def create_gallery(
        self,
        user_id: str,
        image_paths: list[Union[str, Path]],
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        modality_hints: Optional[dict[str, str]] = None,
        enable_clustering: bool = True,
        gallery_storage_path: Optional[Union[str, Path]] = None,
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
        from .user_gallery.api import UserGalleryAPI

        # Initialize user gallery API
        storage_path = gallery_storage_path if gallery_storage_path else "galleries/"
        gallery_api = UserGalleryAPI(storage_path=storage_path)

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
        gallery_storage_path: Optional[Union[str, Path]] = None,
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
        from .user_gallery.api import UserGalleryAPI

        storage_path = gallery_storage_path if gallery_storage_path else "galleries/"
        gallery_api = UserGalleryAPI(storage_path=storage_path)

        return gallery_api.get_gallery(user_id)

    def list_galleries(
        self,
        status_filter: Optional[str] = None,
        gallery_storage_path: Optional[Union[str, Path]] = None,
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
        from .user_gallery.api import UserGalleryAPI

        storage_path = gallery_storage_path if gallery_storage_path else "galleries/"
        gallery_api = UserGalleryAPI(storage_path=storage_path)

        return gallery_api.list_galleries(status_filter=status_filter)

    def delete_gallery(
        self,
        user_id: str,
        permanent: bool = False,
        gallery_storage_path: Optional[Union[str, Path]] = None,
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
        from .user_gallery.api import UserGalleryAPI

        storage_path = gallery_storage_path if gallery_storage_path else "galleries/"
        gallery_api = UserGalleryAPI(storage_path=storage_path)

        return gallery_api.delete_gallery(user_id, permanent=permanent)

    def represent_gallery(
        self,
        user_id: str,
        generate_face_embeddings: bool = False,
        force_regenerate: bool = False,
        batch_size: int = 16,
        gallery_storage_path: Optional[Union[str, Path]] = None,
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
        from .user_gallery.api import UserGalleryAPI
        from .user_gallery.services import EmbeddingGenerationService
        from .user_gallery.storage import GalleryStorageManager

        logger.info(
            f"Generating embeddings for gallery '{user_id}' "
            f"(face_embeddings={generate_face_embeddings}, force={force_regenerate})"
        )

        # Initialize components
        storage_path = gallery_storage_path if gallery_storage_path else "galleries/"
        storage_manager = GalleryStorageManager(Path(storage_path))
        embedding_service = EmbeddingGenerationService(storage_manager)

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
        gallery_storage_path: Optional[Union[str, Path]] = None,
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
        from .user_gallery.services import EmbeddingGenerationService
        from .user_gallery.storage import GalleryStorageManager

        storage_path = gallery_storage_path if gallery_storage_path else "galleries/"
        storage_manager = GalleryStorageManager(Path(storage_path))
        embedding_service = EmbeddingGenerationService(storage_manager)

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
        gallery_storage_path: Optional[Union[str, Path]] = None,
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
        from .user_gallery.fusion import FaceEmbeddingGenerator, FusionRetrievalService
        from .user_gallery.search import MultiModalSearcher
        from .user_gallery.utils import format_retrieval_results_for_display

        logger.info(
            f"Starting user gallery retrieval: probe={Path(probe_image_path).name}, "
            f"gallery={gallery_name}, top_k={top_k}"
        )

        # Initialize storage path
        storage_path = Path(gallery_storage_path if gallery_storage_path else "galleries/")
        gallery_dir = storage_path / gallery_name

        # Check if gallery exists
        if not gallery_dir.exists():
            raise FileNotFoundError(
                f"Gallery '{gallery_name}' not found at {gallery_dir}. "
                f"Please create a gallery first using create_gallery() and represent_gallery()."
            )

        # Initialize face embedding generator if needed
        face_generator = None
        if generate_face_embeddings:
            try:
                face_generator = FaceEmbeddingGenerator(
                    model_name="Facenet", detector_backend="opencv", enforce_detection=False
                )
                logger.debug("Initialized face embedding generator")
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

        # Load gallery searcher
        try:
            searcher = MultiModalSearcher(
                body_dimension=2048,  # ResNet-50 Circle DG dimension
                face_dimension=512 if generate_face_embeddings else None,
                metric="cosine",
                device=str(self.device),
            )
            searcher.load(gallery_dir)
            logger.info(
                f"Loaded gallery '{gallery_name}': {searcher.get_user_count()} users"
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
