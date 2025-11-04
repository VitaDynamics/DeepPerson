# Quickstart: User Gallery Fusion Retrieval

This guide demonstrates how to use the new user gallery fusion retrieval capabilities in DeepPerson.

## Prerequisites

- DeepPerson installed with face embedding support: `pip install deep-person[face-embeddings]`
- DeepFace dependency for face processing: `pip install deepface`
- Optional: GPU acceleration for faster processing

## Basic Usage

### 1. Create a User Gallery

```python
from deep_person import DeepPerson

# Initialize the DeepPerson client
deep_person = DeepPerson()

# Create a new user gallery with body and face images
user_gallery = {
    "user_id": "user_12345",
    "name": "John Doe",
    "metadata": {
        "department": "Security",
        "clearance": "Level 2"
    },
    "images": [
        {
            "image_path": "/path/to/body_image_1.jpg",
            "modality": "BODY",
            "source_camera": "camera_01"
        },
        {
            "image_path": "/path/to/body_image_2.jpg",
            "modality": "BODY",
            "source_camera": "camera_02"
        },
        {
            "image_path": "/path/to/face_image.jpg",
            "modality": "FACE",
            "source_camera": "camera_01"
        }
    ]
}

# Create the gallery
result = deep_person.create_gallery(user_gallery)
print(f"Gallery created: {result.user_id}")
print(f"Images processed: {result.image_count}")
```

### 2. Generate Embeddings

```python
# Generate embeddings for the user gallery
# Optional: Enable face embedding generation
represent_request = {
    "generate_face_embeddings": True,
    "force_regenerate": False
}

result = deep_person.represent("user_12345", represent_request)
print(f"Embeddings generated: {result.generated_embeddings}")
print(f"Face embeddings: {result.face_embeddings_generated}")
```

### 3. Perform User Retrieval

```python
# Retrieve users from a probe image
retrieve_request = {
    "probe_image": "/path/to/probe_image.jpg",
    "top_k": 5,
    "min_score": 0.3,
    "include_evidence": True
}

results = deep_person.retrieve(retrieve_request)

print(f"Retrieved {len(results.results)} users")
for i, result in enumerate(results.results):
    print(f"{i+1}. User: {result.user_id}")
    print(f"   Score: {result.overall_score:.3f}")
    print(f"   Confidence: {result.confidence_level}")
    print(f"   Face contribution: {result.face_score:.3f} (weight: {result.face_weight:.2f})")
    print(f"   Body contribution: {result.body_score:.3f} (weight: {result.body_weight:.2f})")
```

## Advanced Usage

### Custom Fusion Weights

```python
# Customize fusion weights for specific scenarios
retrieve_request = {
    "probe_image": "/path/to/probe_image.jpg",
    "top_k": 10,
    "fusion_weights": {
        "face_weight": 0.7,  # Emphasize face modality
        "body_weight": 0.3
    }
}

results = deep_person.retrieve(retrieve_request)
```

### Batch Processing

```python
# Process multiple users efficiently
user_ids = ["user_12345", "user_12346", "user_12347"]

for user_id in user_ids:
    # Generate embeddings with face support
    result = deep_person.represent(user_id, {
        "generate_face_embeddings": True,
        "batch_size": 16  # Smaller batch for memory efficiency
    })
    print(f"Processed {user_id}: {result.processing_time}ms")
```

### Gallery Management

```python
# Update existing gallery with new images
update_request = {
    "images": [
        {
            "image_path": "/path/to/new_body_image.jpg",
            "modality": "BODY"
        }
    ]
}

result = deep_person.update_gallery("user_12345", update_request)
print(f"Updated gallery: {result.image_count} total images")

# Get gallery details
gallery = deep_person.get_gallery("user_12345")
print(f"User: {gallery.name}")
print(f"Clusters: {gallery.cluster_count}")
print(f"Embeddings: {gallery.embedding_count}")
```

## Configuration

### Environment Variables

```bash
# Enable GPU acceleration
export DEEP_PERSON_USE_GPU=true

# Set default fusion weights
export DEEP_PERSON_FACE_WEIGHT=0.5
export DEEP_PERSON_BODY_WEIGHT=0.5

# Configure face embedding provider
export DEEP_PERSON_FACE_PROVIDER="deepface"
export DEEP_PERSON_FACE_MODEL="Facenet"
```

### Programmatic Configuration

```python
from deep_person import DeepPersonConfig

config = DeepPersonConfig(
    use_gpu=True,
    face_embedding_enabled=True,
    default_face_weight=0.6,
    default_body_weight=0.4,
    min_face_confidence=0.8
)

deep_person = DeepPerson(config=config)
```

## Monitoring and Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Monitor processing times
import time

start_time = time.time()
results = deep_person.retrieve(retrieve_request)
processing_time = time.time() - start_time

print(f"Retrieval completed in {processing_time:.2f} seconds")
print(f"Face embedding generation: {results.face_embedding_time}ms")
print(f"Similarity search: {results.search_time}ms")
```

## Error Handling

```python
try:
    result = deep_person.create_gallery(user_gallery)
except ValueError as e:
    print(f"Validation error: {e}")
except RuntimeError as e:
    print(f"Processing error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Batch Processing**: Use appropriate batch sizes (32-64) for optimal GPU utilization
2. **Caching**: Enable embedding caching to avoid redundant computations
3. **Face Detection**: Pre-filter images to ensure face detection success
4. **Memory Management**: Monitor memory usage with large galleries
5. **GPU Utilization**: Ensure CUDA is properly configured for GPU acceleration

## Example Workflow

```python
# Complete workflow example
def process_user_gallery(user_data):
    """Complete workflow for processing a new user gallery."""

    # 1. Create gallery
    gallery = deep_person.create_gallery(user_data)

    # 2. Generate embeddings with face support
    represent_result = deep_person.represent(
        gallery.user_id,
        {"generate_face_embeddings": True}
    )

    # 3. Verify gallery is ready for retrieval
    if represent_result.embedding_count > 0:
        print(f"Gallery {gallery.user_id} ready for retrieval")
        return gallery.user_id
    else:
        print(f"Failed to generate embeddings for {gallery.user_id}")
        return None

def search_user(probe_image_path):
    """Search for users using a probe image."""

    results = deep_person.retrieve({
        "probe_image": probe_image_path,
        "top_k": 5,
        "include_evidence": True
    })

    return results

# Usage
user_id = process_user_gallery(user_gallery_data)
if user_id:
    matches = search_user("/path/to/probe.jpg")
    for match in matches.results:
        print(f"Match: {match.user_id} (score: {match.overall_score:.3f})")
```

This quickstart provides the foundation for implementing user gallery fusion retrieval in your applications. Refer to the API documentation for complete parameter details and advanced features.