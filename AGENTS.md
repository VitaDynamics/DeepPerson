# Agent Guidelines for DeepPerson Component

## Build/Lint/Test Commands
- **Install**: `pip install -e ..[deep_person]` (from Vbot root)
- **Test single file**: `python -m pytest components/deep_person/test_*.py -v`
- **Test component**: `python -m pytest components/deep_person/ -v --tb=short`
- **Lint**: `python -m flake8 components/deep_person/ --max-line-length=100`
- **Format**: `python -m ruff components/deep_person/`

## Architecture Overview
DeepPerson is a minimal person re-identification library providing embedding generation, verification, and gallery search. Key components:
- **api.py**: Main `DeepPerson` facade class
- **detectors/**: Person detection backends (YOLO, torchvision, etc.)
- **embeddings/**: Feature extraction pipelines
- **search/**: Similarity search with FAISS/sklearn backends
- **backbones/**: Model architectures (ResNet50 with circle loss)
- **registry.py**: Model management and caching
- **utils.py**: Device selection, serialization, gallery management

## Code Style Guidelines
- **Imports**: Standard library → third-party → local, grouped with blank lines
- **Types**: Use full type hints with `typing` module
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Docstrings**: Google-style with Args/Returns/Raises sections
- **Error handling**: Custom exceptions, descriptive messages, logging
- **Paths**: Use `pathlib.Path`, not strings
- **Logging**: Use module-level loggers, appropriate levels (info/debug/warning)
- **Constants**: Define at module level, document their derivation
- **Classes**: Properties over public attributes, validate inputs
