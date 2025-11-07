# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup

**From PyPI:**
```bash
# Install with all features
pip install deep-person[all]

pip install deep-person              # Core features only
```

**Development mode (from repository root):**
```bash
# Install in development mode with all features
pip install -e .[all]


# Using uv (recommended for development)
uv pip install -e .[all]
uv sync --all-extras
```

### Code Quality and Testing
```bash
# Format and lint code (ruff does both)
ruff check --fix src/ tests/  # Auto-fix linting issues
ruff format src/ tests/        # Format code

# Check without fixing
ruff check src/ tests/

# Type checking
mypy src/

# Run tests
pytest tests/ -v --tb=short

# Run specific test types
pytest tests/unit/ -v         # Unit tests only
pytest tests/integration/ -v  # Integration tests only
pytest tests/contract/ -v     # Contract tests only

# Run specific test file
pytest tests/unit/test_fusion_scorer.py -v

# Run tests with marker
pytest -m unit -v
pytest -m integration -v
pytest -m slow -v
```

### Running the Application
```bash
# Quick demo (requires back.jpg in repo root)
python main.py

# Run with specific GPU device
CUDA_VISIBLE_DEVICES=0 python main.py

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Architecture Overview

DeepPerson is a stateless person re-identification library focused on **embedding generation pipelines** and **multi-modal fusion algorithms**. The library provides two core capabilities:

1. **Multi-modal embedding generation** (body + face)
2. **Fusion scoring algorithms** for combining embeddings from different modalities

### Core Architecture Principles

**I. Pipeline Pattern** - Sequential processing with clean interfaces
- Detection → Embedding → Verification/Fusion
- Each stage independently testable
- Interface-based polymorphism (`EmbeddingGenerator`)

**II. Multi-modal Design** - Body and face embeddings with fusion
- Body embeddings: 512-dim vectors from ResNet-50 Circle DG
- Face embeddings: 128-512-dim vectors from DeepFace models (Facenet, VGG-Face, etc.)
- Fusion scoring: Confidence-weighted combination of modalities

**III. Registry Pattern** - Centralized model profile management
- `ModelRegistry` is single source of truth for model configurations
- Thread-safe model loading and caching
- Automatic model download and weight management

**IV. Hardware Optimization** - Automatic GPU/CPU detection
- CUDA detection with graceful CPU fallback
- GPU acceleration for body embeddings (~10-50x speedup)
- Face embeddings typically CPU-based (DeepFace)

**V. Stateless API** - No gallery management, purely functional
- `represent()`: Generate embeddings from images
- `verify()`: Compare two images for identity verification
- No persistent state or database dependencies

### Main API (`src/api.py`)

**DeepPerson class** - Primary façade for all functionality

Core methods:
- `represent(img_path, ...)` - Generate person embeddings with optional face embeddings
  - Single image or batch processing
  - Returns: `{"subjects": [...], "model_info": {...}, "face_model_info": {...}}`
  - Each subject contains: `embedding` (body), `face_embedding` (optional), `metadata`

- `verify(img1, img2, ...)` - Identity verification with multi-modal fusion
  - Automatically generates body + face embeddings for both images
  - Returns: `{"verified": bool, "distance": float, "body_distance": float, "face_distance": float, "fusion_score": float, ...}`
  - Uses `FusionScorer` to combine body and face similarities

### Key Components

**Embedding Generation Pipeline** (`src/embeddings.py`, `src/face_embeddings.py`)

Both implement the `EmbeddingGenerator` interface:
```python
class EmbeddingGenerator(Protocol):
    @property
    def feature_dim(self) -> int: ...

    @property
    def modality(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate_embedding(...) -> PersonEmbedding: ...

    def generate_embeddings_batch(...) -> List[PersonEmbedding]: ...
```

- **BodyEmbeddingGenerator**: ResNet-50 based body embeddings
  - Input: Cropped person images (PIL Images)
  - Preprocessing: Resize (256x128) → ToTensor → Normalize (ImageNet stats)
  - Output: 512-dim embedding vectors (float32)
  - Supports batch processing with configurable batch size

- **FaceEmbeddingGenerator**: DeepFace-based face embeddings
  - Input: Cropped person images (face detected automatically)
  - Models: Facenet (128-dim), Facenet512 (512-dim), VGG-Face (4096-dim), etc.
  - Output: Model-dependent dimensional embeddings
  - Sequential processing (DeepFace limitation)

**Fusion Scoring** (`src/fusion.py`)

`FusionScorer` class implements confidence-weighted late fusion:
- **Input**: Body similarity score, face similarity score, confidences
- **Algorithm**: Weighted combination with dynamic weights based on confidence
  - Default weights: face=0.5, body=0.5
  - Dynamic weighting: `face_weight = face_confidence / (body_confidence + face_confidence)`
  - Minimum face confidence threshold: 0.7 (configurable)
- **Output**: Fused similarity score (0.0-1.0) + fusion metadata
- **Batch processing**: `compute_batch_fusion_scores()` for multiple candidates

Key methods:
```python
scorer = FusionScorer(default_face_weight=0.5, default_body_weight=0.5)

# Single fusion
score, metadata = scorer.compute_fusion_score(
    body_score=0.8,
    face_score=0.9,
    body_confidence=0.95,
    face_confidence=0.85
)

# Batch fusion
scores, metadata_list = scorer.compute_batch_fusion_scores(
    body_scores=np.array([0.8, 0.7, 0.9]),
    face_scores=np.array([0.9, 0.6, 0.85])
)
```

**Distance Metrics** (`src/distance.py`)

`DistanceMetrics` class provides:
- `cosine_distance(emb1, emb2)`: Range [0, 2], 0 = identical
- `euclidean_distance(emb1, emb2)`: L2 distance, range [0, ∞)
- `euclidean_l2_distance(emb1, emb2)`: L2 distance on normalized embeddings, range [0, 2]

Helper function:
```python
from src.distance import compute_distance

distance = compute_distance(emb1, emb2, metric="cosine")  # "euclidean", "euclidean_l2"
```

**Person Detection** (`src/detectors.py`)

- YOLO-based person detection (Ultralytics YOLOv8)
- Factory pattern: `DetectorFactory.create_detector(backend="yolo", device=device)`
- Returns: List of detections with bboxes and confidence scores
- Cropping: `detector.crop_persons(image, detections)` → List of PIL Images

**Data Models** (`src/entities.py`)

Core entities:
- `PersonEmbedding`: Multi-modal embedding container
  - `embedding_vector`: Body embedding (required)
  - `face_embedding`: Face embedding (optional)
  - `modality`: Enum (BODY, FACE, BODY_FACE)
  - `subject_confidence`, `face_confidence`: Detection confidences
  - `bbox`, `face_bbox`: Bounding boxes
  - Helper: `has_face_embedding`, `is_multi_modal`, `with_face_embedding()`

- `Modality`: Enum for embedding types
  - `BODY`: Body-only embeddings
  - `FACE`: Face-only embeddings
  - `BODY_FACE`: Multi-modal embeddings

**Model Registry** (`src/registry.py`)

- Thread-safe singleton: `get_registry()`
- Model profile management: `get_profile(model_name)`, `load_model(model_name, device)`
- Face model caching: `load_face_model(model_name, detector_backend)`
- Verification thresholds: `get_verification_threshold(model_name, metric)`

Default thresholds (distance <= threshold → verified):
```python
DEFAULT_THRESHOLDS = {
    "resnet50_circle_dg": {
        "cosine": 0.40,
        "euclidean": 10.0,
        "euclidean_l2": 0.85
    }
}
```

### Data Flow Architecture

**Basic embedding generation**:
```
Input Image → YOLO Detection → Cropping → ResNet-50 → 512-dim Body Embedding
```

**Multi-modal embedding generation**:
```
Input Image → YOLO Detection → Cropping → ResNet-50 → 512-dim Body Embedding
                                       ↓
                              DeepFace Detection → Face Model → 128-512-dim Face Embedding
                                                                        ↓
                                                            PersonEmbedding(BODY_FACE)
```

**Identity verification with fusion**:
```
Image 1 → Multi-modal Embedding 1 (body + face)
Image 2 → Multi-modal Embedding 2 (body + face)
          ↓
    Body Distance = distance(body1, body2)
    Face Distance = distance(face1, face2)
          ↓
    Convert to similarities: body_sim = 1 - body_dist, face_sim = 1 - face_dist
          ↓
    FusionScorer.compute_fusion_score(body_sim, face_sim, confidences)
          ↓
    Fused Similarity Score (0.0-1.0)
          ↓
    Verified = (fusion_score >= threshold)
```

## Development Guidelines

### Code Style
- **Python**: 3.12+
- **Line length**: 88 characters (ruff)
- **Type hints**: REQUIRED for all public interfaces (mypy strict mode)
- **Formatter**: `ruff format`
- **Linter**: `ruff check` (replaces flake8, isort, pycodestyle)
- **Import style**: stdlib → third-party → local (auto-formatted by ruff)

### Project Structure
```
src/
├── api.py                    # Main DeepPerson façade
├── detectors.py              # Person detection (YOLO)
├── embeddings.py             # Body embedding pipeline
├── face_embeddings.py        # Face embedding pipeline
├── fusion.py                 # Fusion scoring algorithms
├── distance.py               # Distance metrics
├── registry.py               # Model profile registry
├── model_manager.py          # Model download/caching
├── entities.py               # Data models (PersonEmbedding, etc.)
├── interfaces.py             # Protocol definitions
├── utils.py                  # Device selection, normalization
└── backbones/
    └── resnet50_circle_dg.py # Model implementation

tests/
├── contract/                 # API contract tests
├── integration/              # End-to-end tests
└── unit/                     # Component tests
    └── test_fusion_scorer.py # Fusion algorithm tests
```

### Testing Strategy
**Test Markers**:
- `@pytest.mark.unit` - Individual component testing
- `@pytest.mark.integration` - End-to-end pipeline testing
- `@pytest.mark.slow` - Performance/benchmarking tests

**Coverage Targets**:
- 80% minimum for core components
- 90% for critical paths (fusion scoring, distance metrics)

### Performance Considerations
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Batch Processing**: Configurable batch sizes for body embedding generation
- **Memory Management**: Model caching and cleanup utilities
- **Embedding Generation Performance**:
  - Body embeddings: ~100ms per person (GPU, batch_size=1)
  - Face embeddings: ~200-500ms per person (CPU, DeepFace)
  - Batch processing: Linear speedup up to GPU memory limits

## Extension Guide

### Adding New Embedding Models

**For body embeddings**:
1. Implement backbone in `src/backbones/your_model.py`
2. Create `ModelProfile` in `src/registry.py`
3. Update registry loading logic in `_load_model_from_profile()`
4. Add weights to `ModelManager` for automatic download

**For face embeddings**:
1. DeepFace already supports many models (Facenet, VGG-Face, OpenFace, etc.)
2. Update `FaceEmbeddingGenerator.MODEL_DIMENSIONS` if needed
3. No code changes required for existing DeepFace models

### Adding New Distance Metrics
1. Add method to `DistanceMetrics` class in `src/distance.py`
2. Update `compute_distance()` function
3. Add default threshold to `DEFAULT_THRESHOLDS` in `src/registry.py`

### Adding New Fusion Algorithms
1. Create new scorer class in `src/fusion.py`
2. Implement same interface as `FusionScorer`
3. Update `verify()` method in `src/api.py` to support new scorer

## Dependencies

### Core
- `torch>=2.0.0`, `torchvision>=0.15.0` - Deep learning framework
- `numpy>=1.24.0` - Numerical computations
- `pillow>=9.0.0` - Image processing
- `scikit-learn>=1.2.0` - ML utilities
- `gdown>=5.0.0` - Model downloading
- `ultralytics>=8.3.224` - YOLO detection
- `tf-keras>=2.18.0` - Keras integration

### Optional
- `deepface` - Face recognition (required for face embeddings)
- `pytest>=7.0.0`, `pytest-cov>=4.0.0` - Testing
- `ruff>=0.8.0` - Linting and formatting
- `mypy>=1.0.0` - Type checking

## Quick Start Example

```python
from deep_person import DeepPerson

# Initialize (downloads models on first use)
dp = DeepPerson()

# Generate embeddings (body only)
result = dp.represent("person.jpg")
for subject in result["subjects"]:
    print(f"Body embedding: {subject['embedding'].shape}")

# Generate multi-modal embeddings (body + face)
result = dp.represent("person.jpg", generate_face_embeddings=True)
for subject in result["subjects"]:
    body_emb = subject["embedding"]
    face_emb = subject.get("face_embedding")
    print(f"Body: {body_emb.shape}, Face: {face_emb.shape if face_emb is not None else 'None'}")

# Verify identity (with automatic fusion scoring)
result = dp.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']}")
print(f"Body distance: {result['body_distance']:.4f}")
print(f"Face distance: {result['face_distance']:.4f}")
print(f"Fusion score: {result['fusion_score']:.4f}")
print(f"Fusion used: {result['used_fusion']}")

# Batch processing
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
result = dp.represent(images, generate_face_embeddings=True, batch_size=8)
print(f"Total subjects detected: {len(result['subjects'])}")

# Custom fusion weights
from src.fusion import FusionScorer

scorer = FusionScorer(default_face_weight=0.6, default_body_weight=0.4)
fusion_score, metadata = scorer.compute_fusion_score(
    body_score=0.8,
    face_score=0.9,
    body_confidence=0.95,
    face_confidence=0.85
)
print(f"Custom fusion score: {fusion_score:.3f}")
print(f"Applied weights: face={metadata['face_weight']:.2f}, body={metadata['body_weight']:.2f}")
```

## Key Files for Understanding

- **`src/api.py`** - Main API, `DeepPerson` class with `represent()` and `verify()` methods
- **`src/fusion.py`** - Fusion scoring algorithms (`FusionScorer`)
- **`src/distance.py`** - Distance metrics (cosine, euclidean, euclidean_l2)
- **`src/embeddings.py`** - Body embedding pipeline (`BodyEmbeddingGenerator`)
- **`src/face_embeddings.py`** - Face embedding pipeline (`FaceEmbeddingGenerator`)
- **`src/entities.py`** - Data models (`PersonEmbedding`, `Modality`)
- **`src/interfaces.py`** - Protocol definitions (`EmbeddingGenerator`)
- **`src/registry.py`** - Model profile management and verification thresholds
- **`main.py`** - Complete usage examples and demo


# Code Style Guidelines 

## Core Coding Philosophy 

1. **Good Taste**

   * Redesign to eliminate special cases rather than patching with conditions.
   * Elegance is simplicity: fewer branches, fewer exceptions.
   * Experience drives intuition, but rigor validates decisions.

2. **Never Break Userspace**

   * Any change that disrupts existing behavior is a bug.
   * Backward compatibility is non-negotiable.
   * Always test against real-world use, not hypothetical cases.

3. **Pragmatism with Rigor**

   * Solve only real, demonstrated problems.
   * Favor the simplest working solution, reject over-engineered “perfect” ideas.
   * Every design choice must be justified with data, tests, or analysis.

4. **Simplicity Obsession**

   * Functions must be small, focused, and clear.
   * Complexity breeds bugs; minimalism is survival.

5. **Minimal Change Discipline**

   * Only change what’s necessary.
   * Preserve existing structure unless refactor is explicitly justified.
   * Keep scope tight: no speculative “improvements.”


6. **Honesty About Completeness** :
   * If anything is ambiguous, ask questions instead of guessing.
   * If a full solution is impossible (missing specs, unclear APIs, etc.), don’t fake completeness. 
   * Deliver a defensible partial solution, state what’s missing, why, and next steps.

---

## Communication Principles

* **Style**: Direct, sharp, zero fluff. Call out garbage code and explain why.
* **Focus**: Attack technical flaws, never people.
* **Clarity over Comfort**: Being “nice” never overrides technical truth.

## Active Technologies
- Python 3.11+ (project standard) + PyTorch>=2.0.0, Ultralytics>=8.3.224, DeepFace, NumPy>=1.24.0 (001-batch-embedding)
- N/A (stateless library, in-memory processing) (001-batch-embedding)

## Recent Changes
- 001-batch-embedding: Added Python 3.11+ (project standard) + PyTorch>=2.0.0, Ultralytics>=8.3.224, DeepFace, NumPy>=1.24.0
