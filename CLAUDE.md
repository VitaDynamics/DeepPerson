# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup
```bash
# Install in development mode (from Vbot root)
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

# Run tests (when tests directory exists)
pytest tests/ -v --tb=short
pytest tests/unit/test_*.py -v  # Run specific test files
```

### Running the Application
```bash
# Basic usage example
python main.py

# Run with specific GPU device
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Architecture Overview

DeepPerson is a person re-identification component designed for the Vbot framework. It provides end-to-end person detection, embedding generation, and similarity search capabilities.

### Core Components

**Main API (`src/api.py`)**
- `DeepPerson` class: Primary façade exposing all public functionality
- Methods: `represent()`, `verify()`, `find()`, `build_gallery()`

**Processing Pipeline**
1. **Detection** (`src/detectors.py`): YOLO-based person detection with automatic cropping
2. **Embedding Generation** (`src/embeddings.py`): Feature extraction using deep learning models
3. **Similarity Search** (`src/search.py`): FAISS/sklearn-based gallery search with multiple distance metrics

**Model Management**
- **Registry** (`src/registry.py`): Thread-safe model profile management and caching
- **Model Manager** (`src/model_manager.py`): Automatic model downloading and caching
- **Backbones** (`src/backbones/`): Model implementations (ResNet-50 Circle DG)

**Utilities** (`src/utils.py`, `src/entities.py`)
- Device selection and hardware optimization
- Data structures and validation
- Gallery serialization/deserialization

### Data Flow Architecture

```
Input Image → Person Detection (YOLO) → Person Cropping → Feature Extraction (ResNet) → Embedding (2048-dim) → Similarity Search/Verification
```

### Key Design Patterns

- **Factory Pattern**: Used for detectors (`DetectorFactory`) and searchers (`SearcherFactory`)
- **Registry Pattern**: Centralized model profile management with lazy loading
- **Pipeline Pattern**: Sequential processing stages in embedding generation
- **Thread Safety**: All components use `threading.RLock` for concurrent access

## Development Guidelines

### Code Style (from pyproject.toml)
- **Line length**: 88 characters (Ruff)
- **Python version**: 3.12+
- **Import style**: Group imports (stdlib → third-party → local) with Ruff
- **Type hints**: Required for all public interfaces (mypy strict mode)
- **Formatter**: Use `ruff format` (replaces Black)
- **Linter**: Use `ruff check` (replaces flake8, isort, and more)

### Project Structure
```
src/
├── api.py              # Main DeepPerson façade
├── detectors.py        # Person detection implementations
├── embeddings.py       # Feature extraction pipeline
├── search.py           # Similarity search (FAISS/sklearn)
├── registry.py         # Model profile registry
├── model_manager.py    # Model download and caching
├── entities.py         # Data models and validation
├── utils.py           # Utilities (device, serialization)
└── backbones/         # Model architectures
    └── resnet50_circle_dg.py
```

### Adding New Models
1. Implement model in `src/backbones/`
2. Create `ModelProfile` in `src/registry.py`
3. Update registry loading logic
4. Add model weights to model manager

### Adding New Detectors
1. Inherit from `PersonDetector` in `src/detectors.py`
2. Implement `detect()` and `crop_persons()` methods
3. Register in `DetectorFactory.create_detector()`

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Performance Tests**: Benchmarking embedding speed and search accuracy
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`

## Dependencies and Optional Features

### Core Dependencies
- `torch>=2.0.0`: Deep learning framework
- `torchvision>=0.15.0`: Vision models and transforms
- `numpy>=1.24.0`: Numerical computations
- `pillow>=9.0.0`: Image processing
- `scikit-learn>=1.2.0`: Machine learning utilities
- `gdown>=5.0.0`: Model downloading

### Optional Components
- **FAISS GPU**: `faiss-gpu>=1.7.4` for accelerated similarity search
- **FAISS CPU**: `faiss-cpu>=1.7.4` for CPU-based search
- **YOLO Detection**: `ultralytics>=8.3.224` for person detection

## Model and Gallery Management

### Automatic Model Handling
- Models download on first use via `model_manager.py`
- Cached in local directory for subsequent runs
- Thread-safe access with registry pattern

### Gallery Storage Format
```
gallery_dir/
├── {gallery_name}_embeddings.npy    # Embedding matrix
├── {gallery_name}_ids.npy          # Subject ID array
├── {gallery_name}_metadata.pkl     # Metadata dictionary
└── {gallery_name}_config.json      # Search configuration
```

### Performance Considerations
- **GPU Acceleration**: Automatic CUDA detection with CPU fallback
- **Batch Processing**: Configurable batch sizes for embedding generation
- **Memory Management**: Model caching and cleanup utilities
- **Distance Metrics**: Cosine, Euclidean, Euclidean L2 supported

## Recent Changes
- 001-user-gallery-fusion: Added [if applicable, e.g., PostgreSQL, CoreData, files or N/A]
