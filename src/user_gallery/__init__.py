# User Gallery Fusion Component

This module provides user gallery fusion retrieval functionality for the DeepPerson framework.

## Overview

The user gallery fusion component enables:
- Multi-modal person re-identification using both face and body embeddings
- User gallery management with variant clustering
- Confidence-weighted fusion scoring for improved retrieval accuracy
- Backward compatibility with existing DeepPerson workflows

## Architecture

```
user_gallery/
├── __init__.py              # Module initialization
├── models.py                # Data models (UserGallery, VariantCluster, etc.)
├── services.py              # Core services (gallery management, embedding generation)
├── fusion.py                # Multi-modal fusion logic
├── api.py                   # API extensions for user gallery functionality
├── utils.py                 # Utility functions and helpers
├── clustering.py            # Variant clustering algorithms
├── storage.py               # Gallery storage and persistence
├── fusion_algorithms.py     # Fusion scoring algorithms
├── search.py                # Multi-modal search implementation
├── probes.py                # Probe processing and handling
├── results.py               # Result aggregation and ranking
└── algorithms.py            # Core algorithm implementations
```

## Key Components

### Data Models
- **UserGallery**: Represents a unique person with aggregated media and embeddings
- **VariantCluster**: Captures appearance variants (e.g., different outfits, time periods)
- **ImageAsset**: Stores media inputs (body or face) with metadata and processing status
- **EmbeddingSet**: Encapsulates body and face embedding vectors with quality metrics

### Services
- **Gallery Management**: Create, update, and manage user galleries
- **Embedding Generation**: Generate body and face embeddings using DeepFace
- **Fusion Retrieval**: Multi-modal similarity search with confidence weighting
- **Variant Clustering**: Automatic clustering of similar images into variants

### Fusion Logic
- **Confidence-weighted Fusion**: Combines face and body scores based on confidence
- **Multi-modal Search**: FAISS-based search across body and face embeddings
- **Evidence Tracking**: Maintains provenance of matching evidence

## Usage

```python
from deep_person import DeepPerson

# Initialize DeepPerson with user gallery support
dp = DeepPerson()

# Create a user gallery with body and face images
gallery = dp.create_gallery(
    user_id="user_123",
    name="John Doe",
    images=[
        {"image_path": "/path/to/body1.jpg", "modality": "BODY"},
        {"image_path": "/path/to/face1.jpg", "modality": "FACE"}
    ]
)

# Generate embeddings for the gallery
result = dp.represent(user_id="user_123", generate_face_embeddings=True)

# Retrieve users from a probe image
results = dp.retrieve(probe_image="/path/to/probe.jpg", top_k=10)
```

## Configuration

The component supports various configuration options for fusion weights, clustering parameters, and performance tuning through the main DeepPerson configuration system.

## Testing

The module includes comprehensive unit tests, integration tests, and contract tests to ensure reliability and backward compatibility.

## Dependencies

- DeepPerson core
- DeepFace (for face embeddings)
- FAISS (for similarity search)
- PyTorch (for deep learning models)
- NumPy (for numerical computations)