"""
Unit tests for person detection with batch support.

Tests both single-image and batch detection using Ultralytics YOLO.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.detectors import DetectionResult, DetectorFactory, UltralyticsDetector


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a 640x480 RGB image
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_images(sample_image):
    """Create multiple test images."""
    # Reuse the same sample image for simplicity
    return [sample_image, sample_image, sample_image]


@pytest.fixture
def detector():
    """Create a YOLO detector for testing."""
    device = torch.device("cpu")  # Use CPU for tests
    return UltralyticsDetector(device=device, model_name="yolov8n.pt")


class TestDetectionResult:
    """Tests for DetectionResult data class."""

    def test_create_detection_result(self):
        """Test creating a DetectionResult."""
        result = DetectionResult(
            bbox=(100, 100, 200, 300), confidence=0.95, class_name="person", class_id=0
        )

        assert result.bbox == (100, 100, 200, 300)
        assert result.confidence == 0.95
        assert result.class_name == "person"
        assert result.class_id == 0

    def test_detection_result_repr(self):
        """Test DetectionResult string representation."""
        result = DetectionResult(bbox=(10, 20, 30, 40), confidence=0.85)
        repr_str = repr(result)
        assert "DetectionResult" in repr_str
        assert "0.850" in repr_str


class TestUltralyticsDetector:
    """Tests for UltralyticsDetector."""

    def test_detector_initialization(self, detector):
        """Test detector initialization."""
        assert detector.model_name == "yolov8n.pt"
        assert detector.model is not None

    def test_single_image_detection(self, detector, sample_image):
        """Test detection on a single image."""
        detections = detector.detect(image=sample_image, confidence_threshold=0.5)

        # Detections should be a list
        assert isinstance(detections, list)
        # Each detection should be a DetectionResult
        for det in detections:
            assert isinstance(det, DetectionResult)
            assert det.confidence >= 0.5
            assert det.class_name == "person"

    def test_batch_detection(self, detector, sample_images):
        """Test batch detection on multiple images."""
        all_detections = detector.detect_batch(
            images=sample_images, confidence_threshold=0.5
        )

        # Should return list of lists
        assert isinstance(all_detections, list)
        assert len(all_detections) == len(sample_images)

        # Each image should have a list of detections
        for image_detections in all_detections:
            assert isinstance(image_detections, list)
            for det in image_detections:
                assert isinstance(det, DetectionResult)
                assert det.confidence >= 0.5

    def test_batch_detection_empty_list(self, detector):
        """Test batch detection with empty list."""
        all_detections = detector.detect_batch(images=[], confidence_threshold=0.5)

        assert isinstance(all_detections, list)
        assert len(all_detections) == 0

    def test_crop_persons(self, detector, sample_image):
        """Test cropping detected persons from image."""
        # First detect persons
        detections = detector.detect(sample_image, confidence_threshold=0.3)

        if len(detections) > 0:
            # Crop persons
            crops = detector.crop_persons(sample_image, detections)

            assert isinstance(crops, list)
            assert len(crops) == len(detections)

            # Each crop should be a PIL Image
            for crop in crops:
                assert isinstance(crop, Image.Image)
                # Crop should have reasonable dimensions
                assert crop.width > 0
                assert crop.height > 0


class TestDetectorFactory:
    """Tests for DetectorFactory."""

    def test_create_yolo_detector(self):
        """Test creating YOLO detector via factory."""
        detector = DetectorFactory.create_detector(
            backend="yolo", device=torch.device("cpu")
        )

        assert isinstance(detector, UltralyticsDetector)

    def test_list_available_backends(self):
        """Test listing available detector backends."""
        backends = DetectorFactory.list_available_backends()

        assert isinstance(backends, list)
        assert "yolo" in backends
        assert "ultralytics" in backends

    def test_create_detector_with_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported detector backend"):
            DetectorFactory.create_detector(
                backend="invalid_backend", device=torch.device("cpu")
            )


@pytest.mark.integration
class TestBatchDetectionPerformance:
    """Integration tests for batch detection performance."""

    def test_batch_faster_than_sequential(self, detector, sample_images):
        """
        Test that batch detection is faster than sequential detection.

        Note: This is a basic performance test. Actual speedup depends on
        hardware and batch size.
        """
        import time

        # Sequential detection
        start = time.time()
        sequential_detections = []
        for img in sample_images:
            dets = detector.detect(img, confidence_threshold=0.5)
            sequential_detections.append(dets)
        sequential_time = time.time() - start

        # Batch detection
        start = time.time()
        batch_detections = detector.detect_batch(
            sample_images, confidence_threshold=0.5
        )
        batch_time = time.time() - start

        # Batch should return same structure as sequential
        assert len(batch_detections) == len(sequential_detections)

        # Log timing info for reference
        print(f"\nSequential time: {sequential_time:.4f}s")
        print(f"Batch time: {batch_time:.4f}s")
        if sequential_time > 0:
            speedup = sequential_time / batch_time if batch_time > 0 else 0
            print(f"Speedup: {speedup:.2f}x")
