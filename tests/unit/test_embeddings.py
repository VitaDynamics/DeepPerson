"""
Unit tests for body embedding generation with batch support.

Tests both single-image and batch embedding generation.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.embeddings import BodyEmbeddingGenerator, create_embedding_pipeline
from src.entities import Modality, PersonEmbedding


@pytest.fixture
def sample_person_image():
    """Create a sample person crop image."""
    # Create a 256x128 RGB image (standard person crop size)
    img_array = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_person_images(sample_person_image):
    """Create multiple sample person crops."""
    return [sample_person_image for _ in range(5)]


@pytest.fixture
def embedding_generator():
    """Create a BodyEmbeddingGenerator for testing."""
    device = torch.device("cpu")  # Use CPU for tests
    return BodyEmbeddingGenerator(model_name="resnet50_circle_dg", device=device)


class TestBodyEmbeddingGenerator:
    """Tests for BodyEmbeddingGenerator."""

    def test_initialization(self, embedding_generator):
        """Test embedding generator initialization."""
        assert embedding_generator.model_name == "resnet50_circle_dg"
        assert embedding_generator.feature_dim == 512
        assert embedding_generator.modality == Modality.BODY.value

    def test_preprocess_single_image(self, embedding_generator, sample_person_image):
        """Test preprocessing a single image."""
        images = [sample_person_image]
        batch_tensor = embedding_generator.preprocess_images(images)

        # Should return a batched tensor
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (1, 3, 256, 128)  # (B, C, H, W)

    def test_preprocess_batch_images(self, embedding_generator, sample_person_images):
        """Test preprocessing multiple images."""
        batch_tensor = embedding_generator.preprocess_images(sample_person_images)

        # Should return a batched tensor
        assert isinstance(batch_tensor, torch.Tensor)
        assert batch_tensor.shape == (len(sample_person_images), 3, 256, 128)

    def test_extract_features(self, embedding_generator, sample_person_images):
        """Test feature extraction from batched tensor."""
        # Preprocess
        batch_tensor = embedding_generator.preprocess_images(sample_person_images)

        # Extract features
        features = embedding_generator.extract_features(batch_tensor)

        # Should return numpy array with correct shape
        assert isinstance(features, np.ndarray)
        assert features.shape == (len(sample_person_images), 512)
        assert features.dtype == np.float32

    def test_generate_single_embedding(self, embedding_generator, sample_person_image):
        """Test generating embedding for a single person image."""
        bbox = (10, 20, 100, 200)
        confidence = 0.95

        embedding = embedding_generator.generate_embedding(
            image=sample_person_image,
            bbox=bbox,
            confidence=confidence,
            normalize_method="resnet",
        )

        # Check embedding object
        assert isinstance(embedding, PersonEmbedding)
        assert embedding.embedding_vector.shape == (512,)
        assert embedding.subject_confidence == confidence
        assert embedding.bbox == bbox
        assert embedding.modality == Modality.BODY

    def test_generate_embeddings_batch(self, embedding_generator, sample_person_images):
        """Test generating embeddings for multiple person images."""
        bboxes = [(10, 20, 100, 200) for _ in sample_person_images]
        confidences = [0.95, 0.90, 0.85, 0.92, 0.88]

        embeddings = embedding_generator.generate_embeddings_batch(
            images=sample_person_images,
            bboxes=bboxes,
            confidences=confidences,
            normalize_method="resnet",
            batch_size=8,
        )

        # Check embeddings list
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_person_images)

        # Check each embedding
        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, PersonEmbedding)
            assert embedding.embedding_vector.shape == (512,)
            assert embedding.subject_confidence == confidences[i]
            assert embedding.bbox == bboxes[i]
            assert embedding.modality == Modality.BODY

    def test_batch_embedding_smaller_batch_size(
        self, embedding_generator, sample_person_images
    ):
        """Test batch embedding with batch_size smaller than num images."""
        bboxes = [(10, 20, 100, 200) for _ in sample_person_images]
        confidences = [0.95 for _ in sample_person_images]

        # Use batch_size=2 for 5 images (should process in 3 batches)
        embeddings = embedding_generator.generate_embeddings_batch(
            images=sample_person_images,
            bboxes=bboxes,
            confidences=confidences,
            batch_size=2,
        )

        # Should still get all embeddings
        assert len(embeddings) == len(sample_person_images)

    def test_batch_embedding_empty_list(self, embedding_generator):
        """Test batch embedding with empty input list."""
        embeddings = embedding_generator.generate_embeddings_batch(
            images=[], bboxes=[], confidences=[], batch_size=8
        )

        assert isinstance(embeddings, list)
        assert len(embeddings) == 0

    def test_embedding_normalization(self, embedding_generator, sample_person_image):
        """Test that embeddings are properly normalized."""
        bbox = (10, 20, 100, 200)
        confidence = 0.95

        embedding = embedding_generator.generate_embedding(
            image=sample_person_image,
            bbox=bbox,
            confidence=confidence,
            normalize_method="resnet",
        )

        # L2 norm should be approximately 1.0 for normalized embeddings
        l2_norm = np.linalg.norm(embedding.embedding_vector)
        assert abs(l2_norm - 1.0) < 0.01  # Allow small numerical error


class TestCreateEmbeddingPipeline:
    """Tests for the create_embedding_pipeline convenience function."""

    def test_create_default_pipeline(self):
        """Test creating pipeline with default parameters."""
        pipeline = create_embedding_pipeline()

        assert isinstance(pipeline, BodyEmbeddingGenerator)
        assert pipeline.model_name == "resnet50_circle_dg"

    def test_create_pipeline_with_cpu(self):
        """Test creating pipeline with CPU device."""
        device = torch.device("cpu")
        pipeline = create_embedding_pipeline(device=device)

        assert pipeline.device.type == "cpu"

    def test_create_pipeline_auto_device(self):
        """Test pipeline auto-detects device."""
        pipeline = create_embedding_pipeline(prefer_cuda=False)

        # Should default to CPU when cuda not preferred
        assert pipeline.device.type == "cpu"


@pytest.mark.integration
class TestBatchEmbeddingPerformance:
    """Integration tests for batch embedding performance."""

    def test_batch_faster_than_sequential(
        self, embedding_generator, sample_person_images
    ):
        """
        Test that batch embedding is faster than sequential.

        Note: Speedup depends on hardware and batch size.
        """
        import time

        bboxes = [(10, 20, 100, 200) for _ in sample_person_images]
        confidences = [0.95 for _ in sample_person_images]

        # Sequential embedding
        start = time.time()
        sequential_embeddings = []
        for img, bbox, conf in zip(sample_person_images, bboxes, confidences):
            emb = embedding_generator.generate_embedding(
                image=img, bbox=bbox, confidence=conf
            )
            sequential_embeddings.append(emb)
        sequential_time = time.time() - start

        # Batch embedding
        start = time.time()
        batch_embeddings = embedding_generator.generate_embeddings_batch(
            images=sample_person_images,
            bboxes=bboxes,
            confidences=confidences,
            batch_size=8,
        )
        batch_time = time.time() - start

        # Should return same number of embeddings
        assert len(batch_embeddings) == len(sequential_embeddings)

        # Log timing info
        print(f"\nSequential time: {sequential_time:.4f}s")
        print(f"Batch time: {batch_time:.4f}s")
        if sequential_time > 0:
            speedup = sequential_time / batch_time if batch_time > 0 else 0
            print(f"Speedup: {speedup:.2f}x")
