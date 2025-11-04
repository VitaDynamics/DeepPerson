# DeepPerson: Person Re-Identification Component

A high-performance person re-identification system for the Vbot framework, featuring automatic model management, GPU acceleration, and scalable gallery search.

## Features

- **Automatic Person Detection**: YOLO-based person detection with automatic weight management
- **High-Quality Embeddings**: ResNet-50 (Circle DG) backbone for 2048-dimensional person embeddings
- **Identity Verification**: Compare embeddings with multiple distance metrics
- **Gallery Search**: FAISS-accelerated similarity search with sklearn fallback
- **Hardware Optimization**: Automatic GPU/CPU detection and optimization
- **Model Management**: Automatic download and caching of required models
- **Production Ready**: Thread-safe operations with comprehensive error handling

## Quick Start

```python
from components.deep_person.api import DeepPerson

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

Main façade for person re-identification functionality.

```python
dp = DeepPerson(
    model_name="resnet50_circle_dg",  # Model to use
    device="cuda",                    # Force device (None for auto-detect)
    detector_backend="yolo"           # Detection backend
)
```

### Methods

#### `represent(img_path, ...)`

Generate person embeddings from images.

```python
# Single image
result = dp.represent("person.jpg")

# Multiple images (batch processing)
result = dp.represent(["img1.jpg", "img2.jpg", "img3.jpg"])

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
            "embedding": np.ndarray,  # (2048,) embedding vector
            "metadata": {
                "bbox": (x1, y1, x2, y2),
                "confidence": 0.95,
                "hardware": "cuda",
                "model_profile_id": "resnet50_circle_dg",
                "normalization": "resnet"
            }
        }
    ],
    "warnings": [...],
    "model_info": {...}
}
```

#### `verify(img1_path, img2_path, ...)`

Compare two images for identity verification.

```python
# Basic verification
result = dp.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']}")

# Custom settings
result = dp.verify(
    img1_path="person1.jpg",
    img2_path="person2.jpg",
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
    }
}
```

#### `build_gallery(img_paths, subject_ids, ...)`

Create searchable gallery from known persons.

```python
# Build gallery
result = dp.build_gallery(
    img_paths=["person1.jpg", "person2.jpg"],
    subject_ids=["person_001", "person_002"],
    gallery_path="./galleries/staff",
    gallery_name="employees",
    distance_metric="cosine",
    backend="auto",  # "auto", "faiss", "sklearn"
    batch_size=16,
    normalization="resnet"
)

print(f"Gallery created with {result['processed']} entries")
```

#### `find(img_path, gallery_path, ...)`

Search gallery for matching persons.

```python
# Search gallery
result = dp.find(
    img_path="unknown.jpg",
    gallery_path="./galleries/staff",
    gallery_name="employees",
    top_k=10,
    distance_metric="cosine",
    threshold=0.50,  # Optional filter
)

# Process results
for match in result["matches"]:
    print(f"{match['subject_id']}: distance={match['distance']:.4f}")
```

## Advanced Usage

### Gallery Management

```python
from components.deep_person.utils import load_gallery_index, save_gallery_index
from components.deep_person.search import SearcherFactory

# Load existing gallery directly
searcher = load_gallery_index(
    gallery_dir=Path("./galleries/staff"),
    gallery_name="employees",
    backend="faiss",
    device="cuda"
)

# Direct search operations
results = searcher.search(query_embedding, k=5, threshold=0.40)

# Add new embeddings to gallery
searcher.add_embedding(new_embedding, "person_003", metadata)

# Save updated gallery
save_gallery_index(
    searcher=searcher,
    gallery_dir=Path("./galleries/staff"),
    gallery_name="employees_updated"
)
```

### Custom Search Backends

```python
# Use specific backend
searcher = SearcherFactory.create_searcher(
    backend="faiss",
    dimension=2048,
    metric="cosine",
    device="cuda"
)

# Search with custom index
searcher.add_batch(embeddings_matrix, subject_ids, metadata_list)
results = searcher.search(query_embedding, k=10, threshold=0.30)
```

### Model Management

```python
from components.deep_person.model_manager import get_model_manager

# Get model manager instance
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
from components.deep_person.search import compute_distance

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
# components/deep_person/backbones/my_model.py
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
from components.deep_person.registry import get_registry
from components.deep_person.entities import ModelProfile
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
# In registry.py _load_model_from_profile method
elif profile.identifier == "my_person_model":
    from .backbones import my_model
    model = my_model.load_model(
        weights_path=profile.backbone_path,
        device=device
    )
```

### Adding New Detection Backends

```python
# components/deep_person/detectors.py

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
# components/deep_person/search.py

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
# components/deep_person/search.py

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
logging.getLogger('components.deep_person').setLevel(logging.DEBUG)
```

## Architecture Overview

### Component Structure

```
components/deep_person/
├── api.py              # Main façade and orchestration
├── search.py           # FAISS/sklearn similarity search
├── embeddings.py       # Embedding generation pipeline
├── detectors.py        # Person detection (YOLO)
├── registry.py         # Model profile management
├── model_manager.py    # Automatic model download
├── entities.py         # Data models and validation
├── utils.py           # Utility functions and serialization
└── backbones/         # Model implementations
    ├── __init__.py
    └── resnet50_circle_dg.py
```

### Data Flow

1. **Input Image** → `detectors.py` (Person Detection)
2. **Detected Persons** → `embeddings.py` (Feature Extraction)
3. **Embeddings** → `search.py` (Similarity Search/Verification)
4. **Models** → `model_manager.py` (Download/Cache Management)

### Thread Safety

- All components use `threading.RLock` for thread safety
- Gallery operations support concurrent reads/writes
- Model registry is thread-safe for profile access

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