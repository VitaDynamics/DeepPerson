# DeepPerson: Person Re-Identification Component

A high-performance person re-identification system featuring automatic model management, GPU acceleration, and multi-modal gallery search with fusion-based retrieval.

## Features

- **Automatic Person Detection**: YOLO-based person detection with automatic weight management
- **Multi-Modal Embeddings**: ResNet-50 Circle DG backbone for body embeddings + DeepFace for face embeddings
- **Identity Verification**: Compare embeddings with multiple distance metrics and fusion support
- **Gallery Search**: FAISS-accelerated similarity search with sklearn fallback
- **Hardware Optimization**: Automatic GPU/CPU detection and optimization
- **Model Management**: Automatic download and caching of required models
- **Production Ready**: Thread-safe operations with comprehensive error handling and observability

## Quick Start

```python
from src.api import DeepPerson

# Initialize (downloads models on first use)
dp = DeepPerson()

# Generate embeddings for person(s) in image
result = dp.represent("person.jpg")
for subject in result["subjects"]:
    print(f"Person detected with {subject['embedding'].shape} embedding")

# Verify if two images show same person
result = dp.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']} (distance: {result['distance']:.4f})")
```

## Core API

### DeepPerson Class

Main façade for person re-identification functionality, implementing a 5-principle architecture:
- **Registry Pattern** for model management
- **Pipeline Pattern** for processing flow
- **Hardware Optimization** for GPU/CPU detection
- **Gallery System** for multi-modal retrieval
- **Production Readiness** with observability

```python
dp = DeepPerson(
    model_name="resnet50_circle_dg",  # Model to use
    device="cuda",                    # Force device (None for auto-detect)
    detector_backend="yolo",          # Detection backend
    gallery_storage_path="galleries/" # Gallery storage location
)
```

### Methods

#### `represent(img_path, ...)`

Generate person embeddings from images (supports both body and face embeddings via multi-modal processing).

```python
# Single image (body embeddings only)
result = dp.represent("person.jpg")

# Multiple images (batch processing)
result = dp.represent(["img1.jpg", "img2.jpg", "img3.jpg"])

# Multi-modal: generate both body and face embeddings
result = dp.represent(
    img_path="person.jpg",
    generate_face_embeddings=True,
    face_model_name="Facenet",  # DeepFace model for face embeddings
    face_detector_backend="opencv"
)

# Custom settings
result = dp.represent(
    img_path="person.jpg",
    detector_backend="yolo",
    normalization="resnet",  # "base", "resnet", "circle"
    batch_size=16,
    confidence_threshold=0.5
)
```

**Returns:**
```python
{
    "subjects": [
        {
            "embedding": np.ndarray,  # (2048,) body embedding vector
            "face_embedding": np.ndarray,  # (512,) face embedding (if generated)
            "metadata": {
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.95,
                "hardware": "cuda",
                "model_profile_id": "resnet50_circle_dg",
                "normalization": "resnet",
                "face_confidence": 0.98,  # If face detected
                "face_bbox": (x1, y1, x2, y2)  # Face bounding box
            }
        }
    ],
    "warnings": [...],
    "model_info": {...}
}
```

#### `verify(img1_path, img2_path, ...)`

Compare two images for identity verification (supports multi-modal fusion).

```python
# Basic verification
result = dp.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']}")

# Multi-modal fusion verification (body + face)
result = dp.verify(
    img1_path="person1.jpg",
    img2_path="person2.jpg",
    generate_face_embeddings=True,
    fusion_weights={"face": 0.6, "body": 0.4},  # Weighted fusion
    distance_metric="cosine",  # "cosine", "euclidean", "euclidean_l2"
    threshold=0.40,           # Custom threshold (None for model default)
    normalization="resnet"
)
```

**Returns:**
```python
{
    "verified": True,
    "distance": 0.25,
    "threshold": 0.40,
    "distance_metric": "cosine",
    "model": "resnet50_circle_dg",
    "detector_backend": "yolo",
    "facial_areas": {
        "img1": (x1, y1, x2, y2),
        "img2": (x1, y1, x2, y2)
    },
    # Multi-modal fusion results (if enabled)
    "fusion_score": 0.85,
    "body_score": 0.80,
    "face_score": 0.90
}
```

#### User Gallery Management

Create and manage user galleries with multiple images per user.

```python
# Create a user gallery
result = dp.create_gallery(
    user_id="user_001",
    image_paths=["body1.jpg", "body2.jpg", "face1.jpg"],
    name="John Doe",
    metadata={"department": "Security"},
    modality_hints={
        "body1.jpg": "BODY",
        "body2.jpg": "BODY",
        "face1.jpg": "FACE"
    }
)
print(f"Created gallery with {result['total_images']} images")

# Generate embeddings for the gallery
emb_result = dp.represent_gallery(
    user_id="user_001",
    generate_face_embeddings=True  # Optional: include face embeddings
)
print(f"Generated {emb_result['generated_embeddings']} embeddings")

# Update gallery metadata
dp.update_gallery("user_001", name="John Smith", status="ACTIVE")

# Add more images
dp.add_images("user_001", ["new_body.jpg"], modality_hints={"new_body.jpg": "BODY"})

# List all galleries
galleries = dp.list_galleries(status_filter="ACTIVE")
print(f"Found {len(galleries)} active galleries")

# Check if gallery exists
if dp.gallery_exists("user_001"):
    print("Gallery exists")

# Get gallery information
info = dp.get_gallery("user_001")
print(f"Gallery has {info['total_images']} images")

# Delete gallery (soft delete)
dp.delete_gallery("user_001", permanent=False)
```

#### Search User Galleries

Retrieve users using multi-modal fusion search.

```python
# Search gallery with face and body embeddings
result = dp.retrieve_from_gallery(
    probe_image_path="unknown_person.jpg",
    gallery_name="user_gallery",
    top_k=10,
    min_score=0.5,  # Minimum fusion score threshold
    generate_face_embeddings=True,
    fusion_weights={"face": 0.6, "body": 0.4},  # Optional custom weights
    include_evidence=True
)

# Process results
for match in result["results"]:
    print(f"{match['user_id']}: score={match['overall_score']:.3f}")
    print(f"  Body score: {match['body_score']:.3f}")
    print(f"  Face score: {match.get('face_score', 'N/A')}")
```

## Advanced Usage

### Gallery Statistics and Management

```python
# Get detailed gallery statistics
stats = dp.get_gallery_embedding_stats("user_001")
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Face embeddings: {stats['face_embeddings']}")
print(f"Coverage: {stats['coverage']:.1%}")
print(f"Average quality: {stats['average_quality_score']:.3f}")

# Recluster gallery with different algorithm
result = dp.recluster_gallery("user_001", algorithm="DBSCAN")
print(f"Created {result['clusters_created']} clusters")

# Force regenerate all embeddings
emb_result = dp.represent_gallery(
    user_id="user_001",
    generate_face_embeddings=True,
    force_regenerate=True
)
```

### Multi-Modal Embedding Generation

```python
# Generate both body and face embeddings
result = dp.represent(
    img_path="person.jpg",
    generate_face_embeddings=True,
    face_model_name="Facenet",
    face_detector_backend="opencv"
)

# Access multi-modal data
for subject in result["subjects"]:
    body_emb = subject["embedding"]  # Body embedding
    face_emb = subject.get("face_embedding")  # Face embedding (if detected)

    print(f"Body: {body_emb.shape}")
    if face_emb is not None:
        print(f"Face: {face_emb.shape}")
        print(f"Face confidence: {subject['metadata']['face_confidence']}")
```

### Model Management

```python
from src.registry import get_registry
from src.model_manager import get_model_manager

# Get model registry instance (thread-safe model management)
registry = get_registry()

# Get model manager instance (automatic download/caching)
manager = get_model_manager()

# Check cache status
info = manager.get_cache_info()
print(f"Cache size: {info['cache_size_bytes'] / 1024 / 1024:.1f} MB")

# Ensure models are downloaded
backbone_dir = manager.ensure_backbone_weights()
yolo_path = manager.ensure_yolo_weights("yolov8n.pt")

# Clear cache (force re-download)
manager.clear_cache()
```

### Distance Metrics

```python
from src.search import compute_distance

# Compute various distances
emb1 = result1["subjects"][0]["embedding"]
emb2 = result2["subjects"][0]["embedding"]

cosine_dist = compute_distance(emb1, emb2, metric="cosine")
euclidean_dist = compute_distance(emb1, emb2, metric="euclidean")
euclidean_l2_dist = compute_distance(emb1, emb2, metric="euclidean_l2")
```

## Extension Guide

### Adding New Backbone Models

1. **Create Backbone Module:**

```python
# src/backbones/my_model.py
import torch
import torch.nn as nn

class MyPersonModel(nn.Module):
    def __init__(self, feature_dim=2048):
        super().__init__()
        # Define your model architecture
        self.backbone = ...
        self.classifier = ...

    def forward(self, x):
        # Forward pass
        features = self.backbone(x)
        return self.classifier(features)

def load_model(weights_path, device):
    """Load model with weights."""
    model = MyPersonModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
```

2. **Register Model Profile:**

```python
from src.registry import get_registry
from src.entities import ModelProfile
from pathlib import Path

# Create profile
profile = ModelProfile(
    identifier="my_person_model",
    backbone_path=Path("models/my_person_model.pth"),
    feature_dim=2048,
    requires_cuda=False,
    preprocess_config={
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": (256, 128),
        "interpolation": "bilinear"
    }
)

# Register profile
registry = get_registry()
registry.register_profile(profile)
```

3. **Update Registry Loading:**

```python
# In src/registry.py _load_model_from_profile method
elif profile.identifier == "my_person_model":
    from src.backbones import my_model
    model = my_model.load_model(
        weights_path=profile.backbone_path,
        device=device
    )
```

### Adding New Detection Backends

```python
# src/detectors.py

class CustomDetector(PersonDetector):
    def __init__(self, device: torch.device):
        self.device = device
        # Initialize your detector

    def detect(self, image, confidence_threshold=0.5):
        # Implement detection logic
        detections = your_detector.detect(image)
        return [DetectionResult(...) for det in detections]

    def crop_persons(self, image, detections):
        # Implement cropping logic
        crops = []
        for det in detections:
            crop = image.crop(det.bbox)
            crops.append(crop)
        return crops

# Register in DetectorFactory
class DetectorFactory:
    @staticmethod
    def create_detector(backend="yolo", device=None):
        if backend == "custom":
            return CustomDetector(device)
        # ... existing backends
```

### Custom Distance Metrics

```python
# src/search.py

class DistanceMetrics:
    @staticmethod
    def my_custom_distance(embedding1, embedding2):
        """Implement custom distance metric."""
        # Your distance computation
        emb1 = np.asarray(embedding1, dtype=np.float32)
        emb2 = np.asarray(embedding2, dtype=np.float32)

        # Example: Manhattan distance
        distance = np.sum(np.abs(emb1 - emb2))
        return float(distance)

# Update compute_distance function
def compute_distance(embedding1, embedding2, metric="cosine"):
    if metric == "my_custom":
        return DistanceMetrics.my_custom_distance(embedding1, embedding2)
    # ... existing metrics
```

### Custom Search Index

```python
# src/search.py

class CustomSearcher(SimilaritySearcher):
    def __init__(self, dimension, metric="cosine"):
        self.dimension = dimension
        self.metric = metric
        self.embeddings = []
        self.subject_ids = []

    def add_embedding(self, embedding, subject_id, metadata=None):
        # Custom addition logic
        self.embeddings.append(embedding)
        self.subject_ids.append(subject_id)

    def search(self, query_embedding, k=10, threshold=None):
        # Custom search logic
        distances = []
        for i, emb in enumerate(self.embeddings):
            dist = compute_distance(query_embedding, emb, self.metric)
            distances.append((dist, i, self.subject_ids[i]))

        # Sort and filter
        distances.sort(key=lambda x: x[0])
        results = []
        for dist, idx, subject_id in distances[:k]:
            if threshold is None or dist <= threshold:
                results.append({
                    "subject_id": subject_id,
                    "distance": dist,
                    "metadata": {}
                })
        return results
```

## Performance Optimization

### GPU Acceleration

```python
# Check GPU availability
import torch
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")

# Force GPU usage
dp = DeepPerson(device="cuda")

# GPU-accelerated search
result = dp.find("query.jpg", gallery_path, backend="faiss")
```

### Batch Processing

```python
# Large-scale embedding generation
images = [f"person_{i}.jpg" for i in range(1000)]
result = dp.represent(images, batch_size=32)  # Optimize batch size

# Gallery building in batches
def build_gallery_batch(images, subject_ids, batch_size=100):
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_ids = subject_ids[i:i+batch_size]

        result = dp.represent(batch_images, batch_size=batch_size)
        # Add embeddings to gallery...
```

### Memory Management

```python
# Clear model cache to free memory
from components.deep_person.model_manager import get_model_manager
manager = get_model_manager()
manager.clear_cache()

# Use CPU for large batches if GPU memory is limited
dp_cpu = DeepPerson(device="cpu")
large_batch_result = dp_cpu.represent(large_image_batch)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```python
   # Reduce batch size
   result = dp.represent(images, batch_size=8)

   # Use CPU fallback
   dp_cpu = DeepPerson(device="cpu")
   ```

2. **No Person Detected**:
   ```python
   # Lower confidence threshold
   result = dp.represent("image.jpg", confidence_threshold=0.3)

   # Check image quality
   # Try different lighting, angle, or resolution
   ```

3. **Slow Gallery Search**:
   ```python
   # Install FAISS GPU
   pip install faiss-gpu

   # Use smaller top_k
   result = dp.find("query.jpg", gallery_path, top_k=5)
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for specific components
logging.getLogger('src').setLevel(logging.DEBUG)
logging.getLogger('src.api').setLevel(logging.DEBUG)
```

## Architecture Overview

### Core Architecture Principles (5-Principle Design)

DeepPerson implements a production-ready architecture based on five core principles:

**I. Registry Pattern** (`src/registry.py`) - Thread-safe model profile management
- `ModelRegistry` manages both body models (ResNet-50 Circle DG) and face models (DeepFace)
- Lazy loading with `threading.RLock` for body models and `_face_lock` for face models
- Singleton access via `get_instance()` for global state consistency

**II. Pipeline Pattern** - Sequential processing: Detection → Embedding → Search
- Person Detection (`src/detectors.py`): YOLO-based with configurable thresholds
- Feature Extraction (`src/embeddings.py`): ResNet-50 Circle DG generating 2048-dim embeddings
- Similarity Search (`src/search.py`): FAISS/sklearn with multiple distance metrics

**III. Hardware Optimization** - Automatic GPU/CPU detection
- CUDA detection and GPU acceleration (~10-50x speedup)
- FAISS GPU acceleration for large galleries (>10K embeddings)
- CPU fallback fully functional

**IV. Gallery System** (`src/user_gallery/`) - Multi-modal (body+face) with fusion retrieval
- Separate embedding spaces: BODY (2048-dim) and FACE (512-dim with Facenet512)
- Fusion-based retrieval (default: 0.6 face, 0.4 body weights)
- Clustering for appearance variant grouping

**V. Production Readiness** - Observability and quality
- Structured logging with contextual information
- Thread safety validation for all shared state
- Performance metrics and comprehensive error handling

### Component Structure

```
src/
├── api.py                    # Main DeepPerson façade
├── detectors.py              # Person detection (YOLO)
├── embeddings.py             # Body embedding pipeline
├── face_embeddings.py        # Face embedding pipeline (DeepFace)
├── search.py                 # Similarity search (FAISS/sklearn)
├── fusion.py                 # Fusion scoring
├── registry.py               # Model profile registry
├── model_manager.py          # Model download/caching
├── entities.py               # Data models
├── interfaces.py             # Abstract interfaces
├── utils.py                  # Device selection, serialization
├── backbones/                # Model implementations
│   └── resnet50_circle_dg.py
└── user_gallery/             # Gallery system (multi-modal)
    ├── api.py                # _UserGalleryAPI (internal)
    ├── models.py             # Gallery data models
    ├── storage.py            # Storage management
    ├── services.py           # Registration, updates, embeddings
    ├── search.py             # Multi-modal search
    ├── fusion.py             # Fusion retrieval
    └── clustering.py         # Appearance variants
```

### Data Flow

```
Body Pipeline:
Input Image → YOLO Detection → Cropping → ResNet-50 Circle DG → 2048-dim Body Embedding

Face Pipeline:
Input Image → Face Detection (OpenCV/SSD/MTcnn) → Cropping → DeepFace Model → 512-dim Face Embedding

Fusion Scoring:
Weighted combination (configurable, default: 0.6 face, 0.4 body)
```

### Thread Safety

- All components use `threading.RLock` for thread safety
- Gallery operations support concurrent reads/writes
- Model registry is thread-safe for profile access
- All operations are idempotent

## Testing

### Unit Tests

```bash
# Test individual components
pytest tests/unit/test_embeddings.py
pytest tests/unit/test_detectors.py
pytest tests/unit/test_search.py
```

### Integration Tests

```bash
# Test full workflows
pytest tests/integration/test_full_pipeline.py
pytest tests/integration/test_gallery_operations.py
```

### Performance Tests

```bash
# Benchmark performance
pytest tests/performance/test_embedding_speed.py
pytest tests/performance/test_gallery_search.py
```

## References

- [Person ReID Baseline](https://github.com/layumi/Person_reID_baseline_pytorch) - ResNet-50 Circle DG model and other models
- [Ultralytics YOLO](https://docs.ultralytics.com/) - Person detection
- [FAISS Documentation](https://faiss.ai/) - Similarity search optimization
- [Circle Loss](https://arxiv.org/abs/2002.10857) - Loss function for person re-ID