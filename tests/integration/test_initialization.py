import os
import shutil
from pathlib import Path
import pytest
import torch
from src.api import DeepPerson
from src.model_manager import get_model_manager
from src.registry import get_registry

@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clear registry cache before and after each test."""
    get_registry().clear_cache()
    yield
    get_registry().clear_cache()

@pytest.mark.integration
def test_lazy_initialization():
    """Verify that DeepPerson initializes lazily."""
    dp = DeepPerson(model_name="resnet50_circle_dg")

    # Check components are None
    assert dp.detector is None
    assert dp.embedding_pipeline is None

    # Warmup
    dp.warmup()

    # Check components are initialized
    assert dp.detector is not None
    assert dp.embedding_pipeline is not None

    # Verify device - checks basic functionality
    assert isinstance(dp.device, torch.device)

@pytest.mark.integration
def test_custom_cache_dir(tmp_path):
    """Verify custom cache directory is respected."""
    custom_cache = tmp_path / "custom_cache"

    # Initialize with custom cache
    dp = DeepPerson(
        model_name="resnet50_circle_dg",
        cache_dir=custom_cache
    )

    # Check env var
    assert os.environ["DEEPFACE_HOME"] == str(custom_cache)

    # Check ModelManager cache dir
    mm = get_model_manager()
    assert mm.cache_dir == custom_cache

    # Warmup (should download/load models to this cache)
    dp.warmup()

    # Verify directory structure created
    assert custom_cache.exists()
    assert (custom_cache / "detection").exists()
    # Note: backbones might not be downloaded if they are somehow cached globally in memory or logic differs
    # But ensure_backbone_weights in ModelManager uses cache_dir.
    assert (custom_cache / "backbones").exists()

    # Check for actual files (e.g. YOLO weights)
    yolo_weights = list((custom_cache / "detection").glob("*.pt"))
    assert len(yolo_weights) > 0, "YOLO weights not found in custom cache"

@pytest.mark.integration
def test_face_model_warmup(tmp_path):
    """Verify face model warmup works (if DeepFace available)."""
    try:
        import deepface
    except ImportError:
        pytest.skip("DeepFace not installed")

    custom_cache = tmp_path / "face_cache"
    dp = DeepPerson(cache_dir=custom_cache)

    try:
        # Use a model that is hopefully small or likely to be downloadable
        # SFace is usually small
        dp.warmup(face_model_name="SFace")

        # Verify DEEPFACE_HOME is set
        assert os.environ["DEEPFACE_HOME"] == str(custom_cache)

    except Exception as e:
        pytest.fail(f"Face model warmup failed: {e}")
