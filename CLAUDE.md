# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup
```bash
# Install in development mode (from repository root)
uv pip install -e .[all]

# Install with specific feature sets
uv pip install -e .[faiss-gpu,yolo,dev]  # GPU + YOLO + dev tools
uv pip install -e .[faiss-cpu,yolo,dev]  # CPU + YOLO + dev tools

# Sync all dependencies (recommended)
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

DeepPerson is a person re-identification component for the Vbot framework, implementing a **5-principle architecture** defined in `.specify/memory/constitution.md`:

### Core Architecture Principles (Constitution v1.0.0)

**I. Registry Pattern** (`src/registry.py`) - Thread-safe model profile management
- `ModelRegistry` is the single source of truth for model profiles
- All model loading routes through the registry with `threading.RLock`
- Model profiles include: identifier, backbone_path, feature_dim, preprocessing, hardware requirements

**II. Pipeline Pattern** - Sequential processing: Detection → Embedding → Search
- Person Detection (`src/detectors.py`): YOLO-based with configurable thresholds
- Feature Extraction (`src/embeddings.py`): ResNet-50 Circle DG generating 2048-dim embeddings
- Similarity Search (`src/search.py`): FAISS/sklearn with multiple distance metrics (cosine, euclidean, euclidean_l2)
- Each stage independently testable with clean interfaces

**III. Hardware Optimization** - Automatic GPU/CPU detection with graceful fallback
- CUDA detection and GPU acceleration (~10-50x speedup for embeddings)
- FAISS GPU acceleration for large galleries (>10K embeddings)
- Explicit device selection in constructors
- CPU fully functional as complete fallback

**IV. Gallery System** (`src/user_gallery/`) - Multi-modal (body+face) with fusion retrieval
- Standardized storage format: `{gallery}_embeddings.npy`, `{gallery}_ids.npy`, `{gallery}_metadata.pkl`, `{gallery}_config.json`
- BODY and FACE modalities with separate embedding spaces and fusion weights
- Clustering for appearance variant grouping (robust matching across poses/lighting)
- All operations idempotent and thread-safe

**V. Production Readiness** - Comprehensive observability and quality
- Structured logging (DEBUG/INFO/WARNING/ERROR) with contextual information
- Error handling: specific and actionable, no generic exceptions
- Performance metrics: embedding time, detection confidence, search latency, gallery scores
- Test coverage: 80% minimum (90% for critical paths)
- Type hints required for all public interfaces (mypy strict)
- Thread safety validation for all shared state

### Main API (`src/api.py`)

**DeepPerson class** - Primary façade for all functionality

Core methods:
- `represent(img_path, ...)` - Generate person embeddings with multi-modal support
- `verify(img1, img2, ...)` - Identity verification with configurable distance metrics
- `create_gallery(user_id, image_paths, ...)` - Create user galleries with metadata
- `retrieve_from_gallery(probe_image, gallery_name, ...)` - Multi-modal fusion search

Gallery management methods:
- `update_gallery()`, `add_images()`, `delete_gallery()`, `list_galleries()`
- `represent_gallery()`, `get_gallery()`, `gallery_exists()`, `recluster_gallery()`

### Key Design Patterns

- **Factory Pattern**: `DetectorFactory` (detectors.py)
- **Registry Pattern**: Centralized model profile management with lazy loading (`registry.py`)
- **Pipeline Pattern**: Sequential processing with independent testability
- **Façade Pattern**: `DeepPerson` class orchestrates all functionality
- **Thread Safety**: All components use `threading.RLock` for concurrent access

### Data Flow Architecture

```
Input Image → Detection (YOLO) → Cropping → Feature Extraction (ResNet-50 Circle DG) → 2048-dim Embedding → Similarity Search/Verification
```

**Multi-modal extension**:
```
Body: Detection → Cropping → ResNet-50 → 2048-dim Body Embedding
Face: Detection → Face Cropping → Face Model → 512-dim Face Embedding
↓
Fusion Scoring (Weighted combination: default 0.6 face, 0.4 body)
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
├── search.py                 # Similarity search (FAISS/sklearn)
├── fusion.py                 # Fusion scoring
├── registry.py               # Model profile registry
├── model_manager.py          # Model download/caching
├── entities.py               # Data models (PersonEmbedding, etc.)
├── utils.py                  # Device selection, serialization
├── backbones/
│   └── resnet50_circle_dg.py # Model implementation
└── user_gallery/             # Gallery system (multi-modal)
    ├── api.py                # _UserGalleryAPI (internal)
    ├── models.py             # Gallery data models
    ├── storage.py            # Storage management
    ├── services.py           # Registration, updates, embeddings
    ├── search.py             # Multi-modal search
    ├── fusion.py             # Fusion retrieval
    └── clustering.py         # Appearance variants

tests/
├── contract/                 # API contract tests
├── integration/              # End-to-end tests
└── unit/                     # Component tests
```

### Testing Strategy
**Test Markers**:
- `@pytest.mark.unit` - Individual component testing
- `@pytest.mark.integration` - End-to-end pipeline testing
- `@pytest.mark.slow` - Performance/benchmarking tests

**Coverage Targets**:
- 80% minimum for core components
- 90% for critical paths (detection → embedding → search pipeline)

**Running Tests**:
- All tests: `pytest tests/ -v`
- By type: `pytest -m unit|integration|slow -v`
- Specific file: `pytest tests/integration/test_verify_api.py -v`

### Performance Considerations
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Batch Processing**: Configurable batch sizes for embedding generation
- **Memory Management**: Model caching and cleanup utilities
- **Search Performance**:
  - Embedding generation: ~100ms per person (GPU, batch_size=1)
  - Gallery search: Top-10 results in ~50ms (galleries up to 10K embeddings)
  - FAISS GPU acceleration for large galleries (>10K embeddings)

## Extension Guide

### Adding New Models
1. Implement in `src/backbones/your_model.py`
2. Create `ModelProfile` in `src/registry.py`
3. Update registry loading logic in `_load_model_from_profile()`
4. Add weights to `ModelManager` for automatic download

### Adding New Detectors
1. Inherit from `PersonDetector` in `src/detectors.py`
2. Implement `detect()` and `crop_persons()` methods
3. Register in `DetectorFactory.create_detector()`

### Adding New Distance Metrics
1. Add method to `DistanceMetrics` class in `src/search.py`
2. Update `compute_distance()` function
3. Document metric behavior and use cases

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
- `faiss-gpu>=1.12.0` - GPU-accelerated similarity search
- `faiss-cpu>=1.12.0` - CPU-based similarity search
- `pytest>=7.0.0`, `pytest-cov>=4.0.0` - Testing
- `ruff>=0.8.0` - Linting and formatting
- `mypy>=1.0.0` - Type checking

## Gallery Storage Format

```
gallery_dir/
├── {gallery_name}_embeddings.npy    # Embedding matrix
├── {gallery_name}_ids.npy          # Subject ID array
├── {gallery_name}_metadata.pkl     # Metadata dictionary
└── {gallery_name}_config.json      # Search configuration
```

**Multi-modal galleries**:
- Separate embedding spaces for BODY and FACE modalities
- Fusion weights configurable (default: face=0.6, body=0.4)
- Clustering groups appearance variants for robust matching

## Quick Start Example

```python
from src.api import DeepPerson

# Initialize (downloads models on first use)
dp = DeepPerson()

# Generate embeddings
result = dp.represent("person.jpg")
for subject in result["subjects"]:
    print(f"Embedding: {subject['embedding'].shape}")

# Verify identity
result = dp.verify("person1.jpg", "person2.jpg")
print(f"Same person: {result['verified']} (distance: {result['distance']:.4f})")

# Create gallery
dp.create_gallery(
    user_id="user_001",
    image_paths=["img1.jpg", "img2.jpg"],
    modality_hints={"img1.jpg": "BODY", "img2.jpg": "FACE"}
)

# Generate gallery embeddings
dp.represent_gallery(user_id="user_001", generate_face_embeddings=True)

# Multi-modal search
result = dp.retrieve_from_gallery(
    probe_image_path="unknown.jpg",
    gallery_name="user_gallery",
    top_k=10,
    fusion_weights={"face": 0.6, "body": 0.4}
)
```

## Recent Architecture Changes

**Gallery System Consolidation** (Latest):
- Unified to single User Gallery system
- Removed old `build_gallery()` and `find()` methods
- All gallery operations through User Gallery API
- `_UserGalleryAPI` internal - use `DeepPerson` class
- Simplified defaults: `gallery_storage_path` defaults to `"galleries/"`
- New methods: `update_gallery()`, `add_images()`, `gallery_exists()`, `recluster_gallery()`

## Key Files for Understanding

- **`.specify/memory/constitution.md`** - 5 core principles (Registry, Pipeline, Hardware, Gallery, Observability)
- **`src/api.py`** - Main API, all public methods
- **`main.py`** - Complete usage examples
- **`src/registry.py`** - Model profile management
- **`src/detectors.py`** - Person detection implementations
- **`src/embeddings.py`** - Body embedding pipeline
- **`src/search.py`** - Similarity search (FAISS/sklearn)
- **`src/user_gallery/`** - Multi-modal gallery system
