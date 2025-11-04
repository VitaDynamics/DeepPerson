"""
Face embedding generation and multi-modal fusion logic.

This module provides face embedding generation using DeepFace and implements
confidence-weighted fusion algorithms for combining body and face embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class FaceEmbeddingGenerator:
    """
    Generates face embeddings using DeepFace library.

    Provides face detection, alignment, and embedding generation
    with automatic fallback handling for failed detections.
    """

    def __init__(
        self,
        model_name: str = "Facenet",
        detector_backend: str = "opencv",
        enforce_detection: bool = False,
    ):
        """
        Initialize face embedding generator.

        Args:
            model_name: DeepFace model to use ('Facenet', 'VGG-Face', 'OpenFace', etc.)
            detector_backend: Face detector backend ('opencv', 'ssd', 'mtcnn', etc.)
            enforce_detection: If True, raise error when no face detected

        Examples:
            >>> generator = FaceEmbeddingGenerator(model_name="Facenet")
            >>> embedding, confidence = generator.generate_embedding("face.jpg")
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection

        # Lazy import DeepFace to avoid dependency issues
        self._deepface = None

        logger.info(
            f"Initialized FaceEmbeddingGenerator: model={model_name}, "
            f"detector={detector_backend}"
        )

    def _get_deepface(self):
        """Lazy load DeepFace module."""
        if self._deepface is None:
            try:
                from deepface import DeepFace

                self._deepface = DeepFace
                logger.debug("DeepFace module loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "DeepFace is not installed. Install with: pip install deepface"
                ) from e
        return self._deepface

    def generate_embedding(
        self, image_path: str | Path, align: bool = True
    ) -> tuple[Optional[np.ndarray], float]:
        """
        Generate face embedding from an image.

        Args:
            image_path: Path to image file
            align: Whether to align face before embedding generation

        Returns:
            Tuple of (embedding_vector, confidence_score)
            Returns (None, 0.0) if face detection fails and enforce_detection=False

        Raises:
            ValueError: If face detection fails and enforce_detection=True
            FileNotFoundError: If image file doesn't exist

        Examples:
            >>> generator = FaceEmbeddingGenerator()
            >>> embedding, confidence = generator.generate_embedding("face.jpg")
            >>> if embedding is not None:
            ...     print(f"Generated embedding with confidence {confidence:.3f}")
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug(f"Generating face embedding for {image_path.name}")

        try:
            DeepFace = self._get_deepface()

            # Generate embedding using DeepFace
            result = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection,
                align=align,
            )

            # DeepFace returns a list of face embeddings (one per detected face)
            if not result or len(result) == 0:
                logger.warning(f"No face detected in {image_path.name}")
                if self.enforce_detection:
                    raise ValueError(f"No face detected in {image_path.name}")
                return None, 0.0

            # Use first detected face
            face_data = result[0]
            embedding = np.array(face_data["embedding"], dtype=np.float32)

            # Extract confidence from facial area if available
            confidence = 1.0  # Default confidence
            if "facial_area" in face_data:
                # DeepFace doesn't directly provide confidence, use detection success as proxy
                confidence = 0.9  # High confidence if face was detected

            logger.debug(
                f"Generated face embedding: dim={len(embedding)}, confidence={confidence:.3f}"
            )

            return embedding, confidence

        except ValueError as e:
            # Face detection failed
            logger.warning(f"Face detection failed for {image_path.name}: {e}")
            if self.enforce_detection:
                raise
            return None, 0.0

        except Exception as e:
            logger.error(f"Error generating face embedding for {image_path.name}: {e}")
            if self.enforce_detection:
                raise
            return None, 0.0

    def generate_embeddings_batch(
        self, image_paths: list[str | Path], align: bool = True
    ) -> list[tuple[Optional[np.ndarray], float]]:
        """
        Generate face embeddings for multiple images.

        Args:
            image_paths: List of image file paths
            align: Whether to align faces before embedding generation

        Returns:
            List of (embedding_vector, confidence_score) tuples

        Examples:
            >>> generator = FaceEmbeddingGenerator()
            >>> results = generator.generate_embeddings_batch(["face1.jpg", "face2.jpg"])
            >>> for i, (emb, conf) in enumerate(results):
            ...     if emb is not None:
            ...         print(f"Image {i}: embedding generated with confidence {conf:.3f}")
        """
        logger.info(f"Generating face embeddings for {len(image_paths)} images")

        results = []
        for image_path in image_paths:
            embedding, confidence = self.generate_embedding(image_path, align=align)
            results.append((embedding, confidence))

        successful = sum(1 for emb, _ in results if emb is not None)
        logger.info(
            f"Face embedding generation complete: {successful}/{len(image_paths)} successful"
        )

        return results


class FusionScorer:
    """
    Implements confidence-weighted fusion for combining body and face embeddings.

    Uses late fusion strategy with dynamic weighting based on embedding quality
    and detection confidence.
    """

    def __init__(
        self,
        default_face_weight: float = 0.5,
        default_body_weight: float = 0.5,
        min_face_confidence: float = 0.7,
    ):
        """
        Initialize fusion scorer.

        Args:
            default_face_weight: Default weight for face modality (0.0-1.0)
            default_body_weight: Default weight for body modality (0.0-1.0)
            min_face_confidence: Minimum confidence to use face embeddings

        Raises:
            ValueError: If weights don't sum to 1.0 or are out of range

        Examples:
            >>> scorer = FusionScorer(default_face_weight=0.6, default_body_weight=0.4)
            >>> score = scorer.compute_fusion_score(
            ...     body_score=0.8, face_score=0.9,
            ...     body_confidence=0.95, face_confidence=0.85
            ... )
        """
        if not (0.0 <= default_face_weight <= 1.0):
            raise ValueError("Face weight must be between 0.0 and 1.0")
        if not (0.0 <= default_body_weight <= 1.0):
            raise ValueError("Body weight must be between 0.0 and 1.0")
        if abs(default_face_weight + default_body_weight - 1.0) > 1e-6:
            raise ValueError("Face and body weights must sum to 1.0")

        self.default_face_weight = default_face_weight
        self.default_body_weight = default_body_weight
        self.min_face_confidence = min_face_confidence

        logger.info(
            f"Initialized FusionScorer: face_weight={default_face_weight}, "
            f"body_weight={default_body_weight}, min_confidence={min_face_confidence}"
        )

    def compute_fusion_score(
        self,
        body_score: float,
        face_score: Optional[float],
        body_confidence: float = 1.0,
        face_confidence: Optional[float] = None,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute fused similarity score from body and face scores.

        Uses confidence-weighted late fusion:
        - If face score unavailable or low confidence: use body score only
        - Otherwise: weighted combination based on confidence

        Args:
            body_score: Body similarity score (0.0-1.0)
            face_score: Face similarity score (0.0-1.0) or None
            body_confidence: Body embedding confidence (0.0-1.0)
            face_confidence: Face embedding confidence (0.0-1.0) or None
            custom_weights: Optional custom weights dict with 'face' and 'body' keys

        Returns:
            Tuple of (fused_score, fusion_metadata)
            fusion_metadata contains applied weights and contribution details

        Examples:
            >>> scorer = FusionScorer()
            >>> # Both modalities available
            >>> score, meta = scorer.compute_fusion_score(
            ...     body_score=0.8, face_score=0.9,
            ...     body_confidence=0.95, face_confidence=0.85
            ... )
            >>> print(f"Fused score: {score:.3f}")
            >>> print(f"Face weight: {meta['face_weight']:.2f}")
            >>>
            >>> # Face unavailable
            >>> score, meta = scorer.compute_fusion_score(
            ...     body_score=0.8, face_score=None
            ... )
            >>> # Returns body score only
        """
        # Validate inputs
        if not 0.0 <= body_score <= 1.0:
            raise ValueError("Body score must be between 0.0 and 1.0")
        if face_score is not None and not 0.0 <= face_score <= 1.0:
            raise ValueError("Face score must be between 0.0 and 1.0")

        # Determine if face modality should be used
        use_face = (
            face_score is not None
            and face_confidence is not None
            and face_confidence >= self.min_face_confidence
        )

        # Compute weights
        if custom_weights:
            face_weight = custom_weights.get("face", self.default_face_weight)
            body_weight = custom_weights.get("body", self.default_body_weight)
        elif use_face and face_confidence is not None:
            # Dynamic weighting based on confidence
            total_confidence = body_confidence + face_confidence
            if total_confidence > 0:
                face_weight = face_confidence / total_confidence
                body_weight = body_confidence / total_confidence
            else:
                face_weight = self.default_face_weight
                body_weight = self.default_body_weight
        else:
            # Face not available or low confidence: use body only
            face_weight = 0.0
            body_weight = 1.0

        # Normalize weights to sum to 1.0
        weight_sum = face_weight + body_weight
        if weight_sum > 0:
            face_weight /= weight_sum
            body_weight /= weight_sum

        # Compute fused score
        if use_face:
            fused_score = body_weight * body_score + face_weight * face_score
        else:
            fused_score = body_score

        # Prepare metadata
        fusion_metadata = {
            "face_weight": face_weight,
            "body_weight": body_weight,
            "face_used": use_face,
            "body_score": body_score,
            "face_score": face_score if use_face else None,
            "body_confidence": body_confidence,
            "face_confidence": face_confidence if use_face else None,
        }

        logger.debug(
            f"Fusion score computed: {fused_score:.3f} "
            f"(body={body_score:.3f}*{body_weight:.2f}, "
            f"face={face_score if use_face else 'N/A'}*{face_weight:.2f})"
        )

        return fused_score, fusion_metadata

    def compute_batch_fusion_scores(
        self,
        body_scores: np.ndarray,
        face_scores: Optional[np.ndarray],
        body_confidences: Optional[np.ndarray] = None,
        face_confidences: Optional[np.ndarray] = None,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """
        Compute fusion scores for multiple candidates.

        Args:
            body_scores: Array of body similarity scores
            face_scores: Array of face similarity scores or None
            body_confidences: Array of body confidences or None
            face_confidences: Array of face confidences or None
            custom_weights: Optional custom weights

        Returns:
            Tuple of (fused_scores_array, fusion_metadata_list)

        Examples:
            >>> scorer = FusionScorer()
            >>> body_scores = np.array([0.8, 0.7, 0.9])
            >>> face_scores = np.array([0.9, 0.6, 0.85])
            >>> fused_scores, metadata = scorer.compute_batch_fusion_scores(
            ...     body_scores, face_scores
            ... )
        """
        n = len(body_scores)

        # Default confidences if not provided
        if body_confidences is None:
            body_confidences = np.ones(n)
        if face_scores is not None and face_confidences is None:
            face_confidences = np.ones(n)

        # Compute fusion for each candidate
        fused_scores = []
        metadata_list = []

        for i in range(n):
            body_score = float(body_scores[i])
            face_score = float(face_scores[i]) if face_scores is not None else None
            body_conf = float(body_confidences[i])
            face_conf = (
                float(face_confidences[i]) if face_confidences is not None else None
            )

            score, meta = self.compute_fusion_score(
                body_score=body_score,
                face_score=face_score,
                body_confidence=body_conf,
                face_confidence=face_conf,
                custom_weights=custom_weights,
            )

            fused_scores.append(score)
            metadata_list.append(meta)

        return np.array(fused_scores), metadata_list


def compute_embedding_quality_score(
    embedding: np.ndarray,
    detection_confidence: float = 1.0,
    normalization_check: bool = True,
) -> float:
    """
    Compute quality score for an embedding vector.

    Quality is based on:
    - Detection confidence
    - Embedding norm (should be close to 1.0 if normalized)
    - Embedding variance (should not be too low)

    Args:
        embedding: Embedding vector
        detection_confidence: Detection confidence score (0.0-1.0)
        normalization_check: Whether to check if embedding is normalized

    Returns:
        Quality score between 0.0 and 1.0

    Examples:
        >>> embedding = np.random.randn(512)
        >>> embedding = embedding / np.linalg.norm(embedding)  # Normalize
        >>> quality = compute_embedding_quality_score(embedding, detection_confidence=0.9)
        >>> print(f"Embedding quality: {quality:.3f}")
    """
    if embedding is None or len(embedding) == 0:
        return 0.0

    # Start with detection confidence
    quality = detection_confidence

    # Check normalization
    if normalization_check:
        norm = np.linalg.norm(embedding)
        # Penalize if not normalized (should be close to 1.0)
        norm_score = 1.0 - min(abs(norm - 1.0), 1.0)
        quality *= norm_score

    # Check variance (embeddings with very low variance are suspicious)
    variance = np.var(embedding)
    if variance < 1e-6:
        quality *= 0.5  # Penalize low variance

    # Ensure quality is in valid range
    quality = max(0.0, min(1.0, quality))

    return quality


class FusionRetrievalService:
    """
    High-level service for fusion-based user retrieval.

    Orchestrates the complete retrieval workflow:
    1. Probe image processing (body + face embedding generation)
    2. Multi-modal similarity search
    3. Fusion scoring and ranking
    4. Result aggregation and evidence tracking
    """

    def __init__(
        self,
        body_embedding_generator,
        face_embedding_generator: Optional[FaceEmbeddingGenerator] = None,
        fusion_scorer: Optional[FusionScorer] = None,
        default_face_weight: float = 0.5,
        default_body_weight: float = 0.5,
    ):
        """
        Initialize fusion retrieval service.

        Args:
            body_embedding_generator: Generator for body embeddings (e.g., EmbeddingPipeline)
            face_embedding_generator: Optional generator for face embeddings
            fusion_scorer: Optional fusion scorer (creates default if None)
            default_face_weight: Default weight for face modality
            default_body_weight: Default weight for body modality

        Examples:
            >>> from src.embeddings import EmbeddingPipeline
            >>> body_gen = EmbeddingPipeline(model_name="resnet50_circle_dg")
            >>> face_gen = FaceEmbeddingGenerator(model_name="Facenet")
            >>> service = FusionRetrievalService(body_gen, face_gen)
        """
        self.body_embedding_generator = body_embedding_generator
        self.face_embedding_generator = face_embedding_generator

        # Initialize fusion scorer
        if fusion_scorer is None:
            self.fusion_scorer = FusionScorer(
                default_face_weight=default_face_weight,
                default_body_weight=default_body_weight,
            )
        else:
            self.fusion_scorer = fusion_scorer

        logger.info(
            f"Initialized FusionRetrievalService: "
            f"face_enabled={face_embedding_generator is not None}"
        )

    def process_probe_image(
        self,
        probe_image_path: str | Path,
        generate_face_embedding: bool = True,
        detector_backend: Optional[str] = None,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], dict[str, Any]]:
        """
        Process probe image to generate body and optional face embeddings.

        Args:
            probe_image_path: Path to probe image
            generate_face_embedding: Whether to generate face embedding
            detector_backend: Optional detector backend override

        Returns:
            Tuple of (body_embedding, face_embedding, metadata)
            metadata contains confidence scores and processing info

        Raises:
            FileNotFoundError: If probe image doesn't exist
            ValueError: If no person detected in probe image

        Examples:
            >>> service = FusionRetrievalService(body_gen, face_gen)
            >>> body_emb, face_emb, meta = service.process_probe_image("probe.jpg")
            >>> print(f"Body confidence: {meta['body_confidence']:.3f}")
            >>> print(f"Face confidence: {meta['face_confidence']:.3f}")
        """
        probe_path = Path(probe_image_path)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe image not found: {probe_path}")

        logger.info(f"Processing probe image: {probe_path.name}")

        metadata = {
            "probe_image": str(probe_path),
            "body_confidence": 0.0,
            "face_confidence": 0.0,
            "processing_errors": [],
        }

        # Generate body embedding
        body_embedding = None
        try:
            # Use the body embedding generator (EmbeddingPipeline)
            # This should detect person and generate embedding
            from PIL import Image

            img = Image.open(probe_path)

            # Generate embedding using the pipeline
            # Note: This assumes the generator has a method to process single image
            # We'll need to adapt based on actual API
            embeddings = self.body_embedding_generator.generate_embeddings_batch(
                images=[img],
                bboxes=[None],  # Will auto-detect
                confidences=[1.0],
                normalize_method="resnet",
                source_image_ids=[str(probe_path)],
                batch_size=1,
                show_progress=False,
            )

            if len(embeddings) > 0:
                body_embedding = embeddings[0].embedding_vector
                metadata["body_confidence"] = embeddings[0].subject_confidence
                logger.debug(
                    f"Generated body embedding: dim={len(body_embedding)}, "
                    f"confidence={metadata['body_confidence']:.3f}"
                )
            else:
                raise ValueError(f"No person detected in probe image: {probe_path.name}")

        except Exception as e:
            error_msg = f"Failed to generate body embedding: {e}"
            logger.error(error_msg)
            metadata["processing_errors"].append(error_msg)
            raise ValueError(error_msg) from e

        # Generate face embedding if requested
        face_embedding = None
        if generate_face_embedding and self.face_embedding_generator is not None:
            try:
                face_embedding, face_confidence = (
                    self.face_embedding_generator.generate_embedding(probe_path)
                )
                metadata["face_confidence"] = face_confidence

                if face_embedding is not None:
                    logger.debug(
                        f"Generated face embedding: dim={len(face_embedding)}, "
                        f"confidence={face_confidence:.3f}"
                    )
                else:
                    logger.warning(f"No face detected in probe image: {probe_path.name}")

            except Exception as e:
                error_msg = f"Failed to generate face embedding: {e}"
                logger.warning(error_msg)
                metadata["processing_errors"].append(error_msg)
                # Don't raise - face embedding is optional

        return body_embedding, face_embedding, metadata

    def retrieve_users(
        self,
        probe_image_path: str | Path,
        gallery_searcher,
        top_k: int = 10,
        min_score: float = 0.0,
        generate_face_embedding: bool = True,
        fusion_weights: Optional[dict[str, float]] = None,
        include_evidence: bool = True,
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Retrieve users from gallery using fusion scoring.

        This is the main retrieval method that orchestrates the complete workflow:
        1. Process probe image to generate embeddings
        2. Perform multi-modal search
        3. Apply fusion scoring
        4. Rank and filter results

        Args:
            probe_image_path: Path to probe image
            gallery_searcher: MultiModalSearcher instance with loaded gallery
            top_k: Number of top results to return
            min_score: Minimum fusion score threshold
            generate_face_embedding: Whether to generate face embeddings
            fusion_weights: Optional custom fusion weights
            include_evidence: Whether to include evidence images in results

        Returns:
            Tuple of (results_list, retrieval_metadata)
            results_list contains RetrievalResult objects
            retrieval_metadata contains processing statistics

        Raises:
            FileNotFoundError: If probe image doesn't exist
            ValueError: If no person detected or gallery is empty

        Examples:
            >>> service = FusionRetrievalService(body_gen, face_gen)
            >>> searcher = MultiModalSearcher()
            >>> # ... load gallery into searcher ...
            >>> results, meta = service.retrieve_users(
            ...     "probe.jpg",
            ...     searcher,
            ...     top_k=5,
            ...     min_score=0.3
            ... )
            >>> for result in results:
            ...     print(f"{result.user_id}: {result.overall_score:.3f}")
        """
        import time
        from .models import RetrievalProbe

        start_time = time.time()

        logger.info(
            f"Starting user retrieval: probe={Path(probe_image_path).name}, "
            f"top_k={top_k}, min_score={min_score}"
        )

        # Step 1: Process probe image
        body_emb, face_emb, probe_meta = self.process_probe_image(
            probe_image_path, generate_face_embedding=generate_face_embedding
        )

        # Step 2: Create retrieval probe
        probe_id = f"probe_{int(time.time() * 1000)}"
        probe = RetrievalProbe(
            probe_id=probe_id,
            probe_image_path=str(probe_image_path),
            generated_embeddings={},
            fusion_weights=fusion_weights or {},
        )

        if body_emb is not None:
            probe.add_embedding("body", body_emb)
        if face_emb is not None:
            probe.add_embedding("face", face_emb)

        # Step 3: Perform multi-modal search
        results = gallery_searcher.search_multi_modal(
            probe=probe, k=top_k, min_score=min_score, fusion_weights=fusion_weights
        )

        # Step 4: Prepare retrieval metadata
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        retrieval_metadata = {
            "probe_id": probe_id,
            "probe_image": str(probe_image_path),
            "processing_time_ms": processing_time,
            "body_confidence": probe_meta["body_confidence"],
            "face_confidence": probe_meta["face_confidence"],
            "face_embedding_used": face_emb is not None,
            "total_results": len(results),
            "fusion_weights": fusion_weights
            or {
                "face": self.fusion_scorer.default_face_weight,
                "body": self.fusion_scorer.default_body_weight,
            },
            "processing_errors": probe_meta.get("processing_errors", []),
        }

        logger.info(
            f"Retrieval completed: {len(results)} results in {processing_time:.1f}ms"
        )

        return results, retrieval_metadata
