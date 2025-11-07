"""
Unit tests for DeepPerson API image input types support.

Tests PIL Image and NumPy array input support across all public API methods:
- represent()
- verify()
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.api import DeepPerson


class TestImageInputNormalization:
    """Test the _normalize_image() helper method."""

    def test_normalize_str_path(self, tmp_path):
        """Test normalization of string file path."""
        # Create test image
        test_img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.jpg"
        test_img.save(img_path)

        # Normalize
        pil_img, source_id = DeepPerson._normalize_image(str(img_path))

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"
        assert source_id == str(img_path)

    def test_normalize_path_object(self, tmp_path):
        """Test normalization of Path object."""
        # Create test image
        test_img = Image.new("RGB", (100, 100), color="blue")
        img_path = tmp_path / "test.jpg"
        test_img.save(img_path)

        # Normalize
        pil_img, source_id = DeepPerson._normalize_image(img_path)

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"
        assert source_id == str(img_path)

    def test_normalize_pil_image(self):
        """Test normalization of PIL Image."""
        test_img = Image.new("RGB", (100, 100), color="green")

        pil_img, source_id = DeepPerson._normalize_image(test_img)

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"
        assert source_id.startswith("in_memory_")
        assert len(source_id) == 18  # "in_memory_" + 8 hex chars

    def test_normalize_numpy_array(self):
        """Test normalization of NumPy array."""
        # Create NumPy array (H, W, C)
        numpy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pil_img, source_id = DeepPerson._normalize_image(numpy_img)

        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"
        assert source_id.startswith("in_memory_")

    def test_normalize_nonexistent_path(self):
        """Test error handling for nonexistent file path."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            DeepPerson._normalize_image("/nonexistent/path.jpg")

    def test_normalize_invalid_type(self):
        """Test error handling for unsupported type."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            DeepPerson._normalize_image(123)

        with pytest.raises(TypeError, match="Unsupported image type"):
            DeepPerson._normalize_image({"image": "data"})


class TestBatchTypeConsistency:
    """Test the _validate_batch_type_consistency() helper method."""

    def test_empty_list(self):
        """Empty list should pass validation."""
        DeepPerson._validate_batch_type_consistency([])
        # No exception should be raised

    def test_single_item_list(self):
        """Single item list should always pass."""
        pil_img = Image.new("RGB", (10, 10))
        DeepPerson._validate_batch_type_consistency([pil_img])
        # No exception should be raised

    def test_all_pil_images(self):
        """List of all PIL Images should pass."""
        images = [Image.new("RGB", (10, 10)) for _ in range(3)]
        DeepPerson._validate_batch_type_consistency(images)
        # No exception should be raised

    def test_all_numpy_arrays(self):
        """List of all NumPy arrays should pass."""
        arrays = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
        DeepPerson._validate_batch_type_consistency(arrays)
        # No exception should be raised

    def test_all_strings(self):
        """List of all string paths should pass."""
        paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        DeepPerson._validate_batch_type_consistency(paths)
        # No exception should be raised

    def test_all_path_objects(self):
        """List of all Path objects should pass."""
        paths = [Path("img1.jpg"), Path("img2.jpg")]
        DeepPerson._validate_batch_type_consistency(paths)
        # No exception should be raised

    def test_mixed_str_and_path(self):
        """Mixed str and Path should pass (compatible types)."""
        mixed = ["img1.jpg", Path("img2.jpg"), "img3.jpg"]
        DeepPerson._validate_batch_type_consistency(mixed)
        # No exception should be raised

    def test_mixed_pil_and_numpy(self):
        """Mixed PIL and NumPy should raise TypeError."""
        pil_img = Image.new("RGB", (10, 10))
        numpy_img = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(TypeError, match="Mixed batch types not supported"):
            DeepPerson._validate_batch_type_consistency([pil_img, numpy_img])

    def test_mixed_path_and_pil(self):
        """Mixed path and PIL should raise TypeError."""
        pil_img = Image.new("RGB", (10, 10))

        with pytest.raises(TypeError, match="Mixed batch types not supported"):
            DeepPerson._validate_batch_type_consistency(["img1.jpg", pil_img])

    def test_mixed_numpy_and_path(self):
        """Mixed NumPy and path should raise TypeError."""
        numpy_img = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(TypeError, match="Mixed batch types not supported"):
            DeepPerson._validate_batch_type_consistency([numpy_img, "img1.jpg"])


class TestRepresentImageInputs:
    """Test represent() method with different input types."""

    @pytest.fixture
    def mock_dp(self):
        """Create DeepPerson instance with mocked dependencies."""
        with (
            patch("src.api.DetectorFactory"),
            patch("src.api.BodyEmbeddingGenerator"),
            patch("src.api.get_registry"),
        ):
            dp = DeepPerson()

            # Mock detector
            dp.detector = Mock()
            dp.detector.detect = Mock(return_value=[])
            dp.detector.detect_batch = Mock(return_value=[[], [], []])

            return dp

    def test_represent_single_pil_image(self, mock_dp):
        """Test represent() with single PIL Image."""
        pil_img = Image.new("RGB", (100, 100), color="red")

        result = mock_dp.represent(pil_img)

        assert "subjects" in result
        assert "model_info" in result
        assert isinstance(result["subjects"], list)

        # Verify detector was called with PIL Image
        mock_dp.detector.detect.assert_called_once()
        call_args = mock_dp.detector.detect.call_args
        assert isinstance(call_args[1]["image"], Image.Image)

    def test_represent_single_numpy_array(self, mock_dp):
        """Test represent() with single NumPy array."""
        numpy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = mock_dp.represent(numpy_img)

        assert "subjects" in result
        assert isinstance(result["subjects"], list)

    def test_represent_batch_pil_images(self, mock_dp):
        """Test represent() with list of PIL Images."""
        pil_images = [Image.new("RGB", (100, 100)) for _ in range(3)]

        result = mock_dp.represent(pil_images)

        assert "subjects" in result
        # With batch processing, detect_batch should be called once
        mock_dp.detector.detect_batch.assert_called_once()

    def test_represent_batch_numpy_arrays(self, mock_dp):
        """Test represent() with list of NumPy arrays."""
        numpy_images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(2)
        ]

        result = mock_dp.represent(numpy_images)

        assert "subjects" in result
        # With batch processing, detect_batch should be called once
        mock_dp.detector.detect_batch.assert_called_once()

    def test_represent_mixed_batch_raises_error(self, mock_dp):
        """Test represent() with mixed types raises TypeError."""
        pil_img = Image.new("RGB", (100, 100))
        numpy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(TypeError, match="Mixed batch types"):
            mock_dp.represent([pil_img, numpy_img])

    def test_represent_invalid_type_raises_error(self, mock_dp):
        """Test represent() with invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            mock_dp.represent(12345)


class TestVerifyImageInputs:
    """Test verify() method with different input types."""

    @pytest.fixture
    def mock_dp(self):
        """Create DeepPerson instance with mocked represent method."""
        with (
            patch("src.api.DetectorFactory"),
            patch("src.api.BodyEmbeddingGenerator"),
            patch("src.api.get_registry"),
        ):
            dp = DeepPerson()

            # Mock represent() to return fake embeddings
            def mock_represent(*args, **kwargs):
                return {
                    "subjects": [
                        {
                            "embedding": np.random.rand(512).astype(np.float32),
                            "metadata": {"bbox": [0, 0, 100, 100], "confidence": 0.95},
                        }
                    ],
                    "model_info": {"name": "test_model"},
                }

            dp.represent = Mock(side_effect=mock_represent)
            dp.registry = Mock()
            dp.registry.get_verification_threshold = Mock(return_value=0.4)

            return dp

    def test_verify_pil_images(self, mock_dp):
        """Test verify() with two PIL Images."""
        pil_img1 = Image.new("RGB", (100, 100), color="red")
        pil_img2 = Image.new("RGB", (100, 100), color="blue")

        result = mock_dp.verify(pil_img1, pil_img2)

        assert "verified" in result
        assert "distance" in result
        assert "threshold" in result
        assert isinstance(result["verified"], bool)

        # Verify represent was called twice
        assert mock_dp.represent.call_count == 2

    def test_verify_numpy_arrays(self, mock_dp):
        """Test verify() with two NumPy arrays."""
        numpy_img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        numpy_img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = mock_dp.verify(numpy_img1, numpy_img2)

        assert "verified" in result
        assert mock_dp.represent.call_count == 2

    def test_verify_mixed_pil_and_path(self, mock_dp, tmp_path):
        """Test verify() with mixed PIL Image and path."""
        # Create test image file
        test_img = Image.new("RGB", (100, 100), color="green")
        img_path = tmp_path / "test.jpg"
        test_img.save(img_path)

        pil_img = Image.new("RGB", (100, 100), color="red")

        result = mock_dp.verify(pil_img, str(img_path))

        assert "verified" in result
        assert mock_dp.represent.call_count == 2



class TestBatchRepresentImageInputs:
    """Test batch processing through represent() method with different input types."""

    @pytest.fixture
    def mock_dp(self):
        """Create DeepPerson instance with mocked dependencies."""
        with (
            patch("src.api.DetectorFactory"),
            patch("src.api.BodyEmbeddingGenerator"),
            patch("src.api.get_registry"),
        ):
            dp = DeepPerson()

            # Mock detector
            dp.detector = Mock()
            # For single image: return empty list
            dp.detector.detect = Mock(return_value=[])
            # For batch: return list of lists (one list per image)
            dp.detector.detect_batch = Mock(return_value=[[], [], []])

            # Mock embedding pipeline
            dp.embedding_pipeline = Mock()
            dp.embedding_pipeline.feature_dim = 512

            return dp

    def test_represent_batch_with_list_pil_images(self, mock_dp):
        """Test that represent() with list of PIL Images uses batch processing."""
        pil_images = [Image.new("RGB", (100, 100)) for _ in range(3)]

        result = mock_dp.represent(pil_images)

        assert "subjects" in result
        assert "model_info" in result
        # With batch processing and detect_batch available, it should be called once
        mock_dp.detector.detect_batch.assert_called_once()

    def test_represent_batch_with_list_numpy_arrays(self, mock_dp):
        """Test that represent() with list of NumPy arrays uses batch processing."""
        numpy_images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(2)
        ]

        result = mock_dp.represent(numpy_images)

        assert "subjects" in result
        # With batch processing and detect_batch available, it should be called once
        # (Note: The mock returns 3 empty lists, but we're only sending 2 images)
        # So we need to update the mock to match the input
        assert mock_dp.detector.detect_batch.call_count == 1

    def test_represent_batch_mixed_types_raises_error(self, mock_dp):
        """Test represent() with mixed types in batch raises TypeError."""
        pil_img = Image.new("RGB", (100, 100))
        numpy_img = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(TypeError, match="Mixed batch types"):
            mock_dp.represent([pil_img, numpy_img])

    def test_represent_empty_list(self, mock_dp):
        """Test represent() with empty list returns empty results."""
        result = mock_dp.represent([])

        assert "subjects" in result
        assert len(result["subjects"]) == 0


class TestSourceIDGeneration:
    """Test source_id generation for in-memory images."""

    @pytest.fixture
    def mock_dp(self):
        """Create DeepPerson instance with mocked dependencies."""
        with (
            patch("src.api.DetectorFactory"),
            patch("src.api.BodyEmbeddingGenerator"),
            patch("src.api.get_registry"),
        ):
            dp = DeepPerson()

            # Mock detector to return fake detections
            mock_detection = Mock()
            mock_detection.bbox = [10, 10, 90, 90]
            mock_detection.confidence = 0.95

            dp.detector = Mock()
            dp.detector.detect = Mock(return_value=[mock_detection])
            dp.detector.crop_persons = Mock(return_value=[Image.new("RGB", (80, 80))])

            # Mock embedding pipeline
            mock_embedding = Mock()
            mock_embedding.embedding_vector = np.random.rand(512).astype(np.float32)
            mock_embedding.subject_confidence = 0.95
            mock_embedding.bbox = [10, 10, 90, 90]
            mock_embedding.normalization = "resnet"
            mock_embedding.model_profile_id = "test_model"
            mock_embedding.hardware = "cpu"
            mock_embedding.timestamp = None
            mock_embedding.source_image_id = "in_memory_12345678"
            mock_embedding.modality = Mock()
            mock_embedding.modality.value = "body"
            mock_embedding.has_face_embedding = False

            dp.embedding_pipeline = Mock()
            dp.embedding_pipeline.generate_embeddings_batch = Mock(
                return_value=[mock_embedding]
            )
            dp.embedding_pipeline.profile = Mock()
            dp.embedding_pipeline.profile.feature_dim = 512

            return dp

    def test_source_id_for_pil_image(self, mock_dp):
        """Test that in-memory PIL Images get unique source IDs."""
        pil_img = Image.new("RGB", (100, 100), color="red")

        result = mock_dp.represent(pil_img)

        assert len(result["subjects"]) > 0
        source_id = result["subjects"][0]["metadata"]["source_image"]
        assert source_id.startswith("in_memory_")

    def test_source_id_for_numpy_array(self, mock_dp):
        """Test that NumPy arrays get unique source IDs."""
        numpy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        result = mock_dp.represent(numpy_img)

        assert len(result["subjects"]) > 0
        source_id = result["subjects"][0]["metadata"]["source_image"]
        assert source_id.startswith("in_memory_")

    def test_source_id_uniqueness(self):
        """Test that multiple in-memory images get unique source IDs."""
        pil_img1 = Image.new("RGB", (100, 100))
        pil_img2 = Image.new("RGB", (100, 100))

        _, source_id1 = DeepPerson._normalize_image(pil_img1)
        _, source_id2 = DeepPerson._normalize_image(pil_img2)

        assert source_id1 != source_id2
        assert source_id1.startswith("in_memory_")
        assert source_id2.startswith("in_memory_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
