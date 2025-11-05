"""
Multi-modal fusion logic for combining body and face embeddings.

This module implements confidence-weighted fusion algorithms for combining
body and face embeddings in person re-identification tasks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


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
        face_embedding_generator=None,
        fusion_scorer: Optional[FusionScorer] = None,
        default_face_weight: float = 0.5,
        default_body_weight: float = 0.5,
    ):
        """
        Initialize fusion retrieval service.

        Args:
            body_embedding_generator: Generator for body embeddings (e.g., BodyEmbeddingGenerator)
            face_embedding_generator: Optional generator for face embeddings (from src.face_embeddings)
            fusion_scorer: Optional fusion scorer (creates default if None)
            default_face_weight: Default weight for face modality
            default_body_weight: Default weight for body modality

        Examples:
            >>> from src.embeddings import BodyEmbeddingGenerator
            >>> from src.face_embeddings import FaceEmbeddingGenerator
            >>> body_gen = BodyEmbeddingGenerator(model_name="resnet50_circle_dg")
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
            # Use the body embedding generator (BodyEmbeddingGenerator)
            # This should detect person and generate embedding
            from PIL import Image

            img = Image.open(probe_path)

            # Generate embedding using the pipeline
            # FIXME: This assumes the generator has a method to process single image
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
                from PIL import Image

                # Core FaceEmbeddingGenerator requires PIL Image, bbox, and confidence
                img = Image.open(probe_path)

                # Use the embeddings[0].bbox if available, otherwise use image dimensions
                bbox = (0, 0, img.width, img.height)
                if len(embeddings) > 0 and hasattr(embeddings[0], 'bbox'):
                    bbox = embeddings[0].bbox

                # Generate face embedding using core implementation
                face_person_embedding = self.face_embedding_generator.generate_embedding(
                    image=img,
                    bbox=bbox,
                    confidence=metadata["body_confidence"],
                    source_image_id=str(probe_path),
                )

                # Extract face embedding and confidence from PersonEmbedding
                if face_person_embedding.face_embedding is not None:
                    face_embedding = face_person_embedding.face_embedding
                    metadata["face_confidence"] = face_person_embedding.face_confidence or 0.0

                    logger.debug(
                        f"Generated face embedding: dim={len(face_embedding)}, "
                        f"confidence={metadata['face_confidence']:.3f}"
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
