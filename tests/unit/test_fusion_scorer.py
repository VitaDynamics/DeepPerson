"""
Unit tests for FusionScorer class.

This module tests the fusion scoring functionality that combines body and face
embeddings for multi-modal person verification and retrieval.
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.fusion import FusionScorer


class TestFusionScorer:
    """Test cases for FusionScorer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = FusionScorer(
            default_face_weight=0.5,
            default_body_weight=0.5,
            min_face_confidence=0.7
        )

    @pytest.mark.unit
    def test_fusion_scorer_initialization_valid(self):
        """Test FusionScorer initialization with valid parameters."""
        scorer = FusionScorer(
            default_face_weight=0.6,
            default_body_weight=0.4,
            min_face_confidence=0.8
        )

        assert scorer.default_face_weight == 0.6
        assert scorer.default_body_weight == 0.4
        assert scorer.min_face_confidence == 0.8

    @pytest.mark.unit
    def test_fusion_scorer_initialization_weights_out_of_range(self):
        """Test FusionScorer initialization with weights out of range."""
        with pytest.raises(ValueError, match="Face weight must be between 0.0 and 1.0"):
            FusionScorer(default_face_weight=1.5, default_body_weight=-0.5)

        with pytest.raises(ValueError, match="Body weight must be between 0.0 and 1.0"):
            FusionScorer(default_face_weight=0.3, default_body_weight=1.2)

    @pytest.mark.unit
    def test_fusion_scorer_initialization_weights_not_summing_to_one(self):
        """Test FusionScorer initialization with weights not summing to 1.0."""
        with pytest.raises(ValueError, match="Face and body weights must sum to 1.0"):
            FusionScorer(default_face_weight=0.6, default_body_weight=0.6)  # Sum = 1.2

        with pytest.raises(ValueError, match="Face and body weights must sum to 1.0"):
            FusionScorer(default_face_weight=0.3, default_body_weight=0.6)  # Sum = 0.9

    @pytest.mark.unit
    def test_compute_fusion_score_both_modalities_available(self):
        """Test fusion scoring with both body and face embeddings available."""
        body_score = 0.8
        face_score = 0.9
        body_confidence = 0.95
        face_confidence = 0.85

        fused_score, metadata = self.scorer.compute_fusion_score(
            body_score=body_score,
            face_score=face_score,
            body_confidence=body_confidence,
            face_confidence=face_confidence
        )

        # Check that fusion score is between body and face scores
        assert min(body_score, face_score) <= fused_score <= max(body_score, face_score)

        # Check metadata
        assert metadata["body_score"] == body_score
        assert metadata["face_score"] == face_score
        assert metadata["body_confidence"] == body_confidence
        assert metadata["face_confidence"] == face_confidence
        assert metadata["face_used"] is True
        assert metadata["face_weight"] > 0
        assert metadata["body_weight"] > 0
        assert abs(metadata["face_weight"] + metadata["body_weight"] - 1.0) < 1e-6

    @pytest.mark.unit
    def test_compute_fusion_score_body_only_fallback(self):
        """Test fusion scoring fallback to body-only when face unavailable."""
        body_score = 0.8
        face_score = None  # Face not available

        fused_score, metadata = self.scorer.compute_fusion_score(
            body_score=body_score,
            face_score=face_score
        )

        # Should return body score only
        assert fused_score == body_score

        # Check metadata
        assert metadata["body_score"] == body_score
        assert metadata["face_score"] is None
        assert metadata["face_used"] is False
        assert metadata["face_weight"] == 0.0
        assert metadata["body_weight"] == 1.0

    @pytest.mark.unit
    def test_compute_fusion_score_low_face_confidence(self):
        """Test fusion scoring when face confidence is below threshold."""
        body_score = 0.8
        face_score = 0.9
        body_confidence = 0.95
        face_confidence = 0.5  # Below min_face_confidence (0.7)

        fused_score, metadata = self.scorer.compute_fusion_score(
            body_score=body_score,
            face_score=face_score,
            body_confidence=body_confidence,
            face_confidence=face_confidence
        )

        # Should fallback to body-only due to low face confidence
        assert fused_score == body_score
        assert metadata["face_used"] is False
        assert metadata["face_weight"] == 0.0
        assert metadata["body_weight"] == 1.0

    @pytest.mark.unit
    def test_compute_fusion_score_custom_weights(self):
        """Test fusion scoring with custom weights."""
        body_score = 0.8
        face_score = 0.9
        face_confidence = 0.8  # Above min_face_confidence (0.7)
        custom_weights = {"face": 0.7, "body": 0.3}

        fused_score, metadata = self.scorer.compute_fusion_score(
            body_score=body_score,
            face_score=face_score,
            face_confidence=face_confidence,
            custom_weights=custom_weights
        )

        # Check that custom weights are used
        expected_fused = 0.3 * body_score + 0.7 * face_score
        assert abs(fused_score - expected_fused) < 1e-6

        # Check metadata
        assert metadata["face_weight"] == 0.7
        assert metadata["body_weight"] == 0.3
        assert metadata["face_used"] is True

    @pytest.mark.unit
    def test_compute_fusion_score_invalid_scores(self):
        """Test fusion scoring with invalid score values."""
        # Invalid body score
        with pytest.raises(ValueError, match="Body score must be between 0.0 and 1.0"):
            self.scorer.compute_fusion_score(body_score=1.5, face_score=0.9)

        with pytest.raises(ValueError, match="Body score must be between 0.0 and 1.0"):
            self.scorer.compute_fusion_score(body_score=-0.2, face_score=0.9)

        # Invalid face score
        with pytest.raises(ValueError, match="Face score must be between 0.0 and 1.0"):
            self.scorer.compute_fusion_score(body_score=0.8, face_score=1.5)

        with pytest.raises(ValueError, match="Face score must be between 0.0 and 1.0"):
            self.scorer.compute_fusion_score(body_score=0.8, face_score=-0.2)

    @pytest.mark.unit
    def test_compute_batch_fusion_scores(self):
        """Test batch fusion scoring for multiple candidates."""
        body_scores = np.array([0.8, 0.7, 0.9])
        face_scores = np.array([0.9, 0.6, 0.85])
        body_confidences = np.array([0.95, 0.8, 0.9])
        face_confidences = np.array([0.85, 0.5, 0.8])

        fused_scores, metadata_list = self.scorer.compute_batch_fusion_scores(
            body_scores=body_scores,
            face_scores=face_scores,
            body_confidences=body_confidences,
            face_confidences=face_confidences
        )

        # Check output shapes
        assert len(fused_scores) == 3
        assert len(metadata_list) == 3

        # Check individual results
        for i in range(3):
            assert isinstance(fused_scores[i], (float, np.floating))
            assert isinstance(metadata_list[i], dict)
            assert "face_weight" in metadata_list[i]
            assert "body_weight" in metadata_list[i]
            assert "face_used" in metadata_list[i]

    @pytest.mark.unit
    def test_compute_batch_fusion_scores_no_face_scores(self):
        """Test batch fusion scoring when face scores unavailable."""
        body_scores = np.array([0.8, 0.7, 0.9])
        face_scores = None  # No face scores available

        fused_scores, metadata_list = self.scorer.compute_batch_fusion_scores(
            body_scores=body_scores,
            face_scores=face_scores
        )

        # All should fallback to body-only
        assert np.array_equal(fused_scores, body_scores)
        for metadata in metadata_list:
            assert metadata["face_used"] is False
            assert metadata["face_weight"] == 0.0
            assert metadata["body_weight"] == 1.0

    @pytest.mark.unit
    def test_compute_batch_fusion_scores_default_confidences(self):
        """Test batch fusion scoring with default confidences."""
        body_scores = np.array([0.8, 0.7, 0.9])
        face_scores = np.array([0.9, 0.6, 0.85])
        # No confidences provided - should default to 1.0

        fused_scores, metadata_list = self.scorer.compute_batch_fusion_scores(
            body_scores=body_scores,
            face_scores=face_scores
        )

        # Should work with default confidences
        assert len(fused_scores) == 3
        for metadata in metadata_list:
            assert metadata["body_confidence"] == 1.0
            assert metadata["face_confidence"] == 1.0

    @pytest.mark.unit
    def test_dynamic_weighting_based_on_confidence(self):
        """Test dynamic weighting based on confidence values."""
        body_score = 0.8
        face_score = 0.9
        body_confidence = 0.6  # Lower confidence
        face_confidence = 0.9  # Higher confidence

        fused_score, metadata = self.scorer.compute_fusion_score(
            body_score=body_score,
            face_score=face_score,
            body_confidence=body_confidence,
            face_confidence=face_confidence
        )

        # Face should get higher weight due to higher confidence
        assert metadata["face_weight"] > metadata["body_weight"]

        # Expected weights based on confidence ratio
        total_conf = body_confidence + face_confidence
        expected_face_weight = face_confidence / total_conf
        expected_body_weight = body_confidence / total_conf
        assert abs(metadata["face_weight"] - expected_face_weight) < 1e-6
        assert abs(metadata["body_weight"] - expected_body_weight) < 1e-6

    @pytest.mark.unit
    def test_edge_case_zero_confidence(self):
        """Test fusion scoring with zero confidence."""
        body_score = 0.8
        face_score = 0.9
        body_confidence = 0.0
        face_confidence = 0.0

        fused_score, metadata = self.scorer.compute_fusion_score(
            body_score=body_score,
            face_score=face_score,
            body_confidence=body_confidence,
            face_confidence=face_confidence
        )

        # Face confidence is below threshold, so should fallback to body-only
        assert metadata["face_used"] is False
        assert metadata["face_weight"] == 0.0
        assert metadata["body_weight"] == 1.0
        assert fused_score == body_score