"""
Multi-modal fusion logic for combining body and face embeddings.

This module implements confidence-weighted fusion algorithms for combining
body and face embeddings in person re-identification tasks.

The FusionScorer class provides fusion scoring functionality that can be used
across different components (verification, retrieval, gallery search) for
consistent multi-modal person matching.
"""

from __future__ import annotations

import logging
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