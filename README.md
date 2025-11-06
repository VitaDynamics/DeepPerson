# DeepPerson

A simple person re-identification system with automatic person detection and embedding generation.

## Quick Start

**Just 3 lines of code:**

```python
from src.api import DeepPerson

# Initialize (downloads models automatically)
dp = DeepPerson()

# Generate embeddings
result = dp.represent("person.jpg")

# Verify if two images show the same person
is_same = dp.verify("person1.jpg", "person2.jpg")
```

## Installation

```bash
# Install with all features
uv pip install -e .[all]

# Or with specific features
uv pip install -e .[faiss-gpu,yolo]  # GPU + YOLO
uv pip install -e .[faiss-cpu,yolo]  # CPU + YOLO
```

**Run the demo:**

```bash
# Place a test image named 'back.jpg' in the repository root
python main.py
```

## Core API

### Two Simple Methods

#### 1. `represent()` - Generate embeddings from images

```python
# Single image
result = dp.represent("person.jpg")
for subject in result["subjects"]:
    print(f"Embedding: {subject['embedding'].shape}")

# Multiple images (batch processing)
result = dp.represent(["img1.jpg", "img2.jpg", "img3.jpg"])
```

**Returns:**
```python
{
    "subjects": [
        {
            "embedding": np.ndarray,  # 2048-dimensional embedding
            "face_embedding": np.ndarray,  # Optional: 512-dim face embedding
            "metadata": {
                "confidence": 0.95,
                "bbox": (x1, y1, x2, y2)
            }
        }
    ]
}
```

#### 2. `verify()` - Identity verification

```python
# Compare two images
result = dp.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']} (distance: {result['distance']:.4f})")

# With custom threshold
result = dp.verify("person1.jpg", "person2.jpg", threshold=0.4)
```

**Returns:**
```python
{
    "verified": True,
    "distance": 0.25,
    "threshold": 0.40,
    "distance_metric": "cosine",
    "body_distance": 0.30,
    "face_distance": 0.20,
    "fusion_score": 0.24
}
```

## Key Features

- ✓ **Automatic detection** - YOLO-based person detection
- ✓ **Multi-modal** - Body + face embeddings with fusion scoring
- ✓ **GPU acceleration** - Automatic CUDA detection
- ✓ **Batch processing** - Handle multiple images efficiently
- ✓ **Multiple metrics** - Cosine, Euclidean, L2 normalization
- ✓ **Stateless API** - No complex gallery management required

## Requirements

- Python 3.11+
- Automatic model downloads on first use
- Optional: CUDA GPU for faster processing
