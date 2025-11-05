# User Gallery Unit Tests

This module contains unit tests for the user gallery fusion component.

## Test Structure

```
tests/unit/test_user_gallery/
├── __init__.py              # Module initialization
├── test_models.py           # Data model tests
├── test_services.py         # Service layer tests
├── test_embeddings.py       # Embedding generation tests
├── test_fusion.py           # Fusion logic tests
├── test_utils.py            # Utility function tests
└── test_api.py              # API integration tests
```

## Test Coverage

### Core Models
- UserGallery model validation and business logic
- VariantCluster lifecycle management
- ImageAsset processing states and validation
- EmbeddingSet quality scoring and normalization
- RetrievalProbe and RetrievalResult functionality

### Services
- Gallery creation and management
- Embedding generation workflows
- Variant clustering algorithms
- Fusion scoring and retrieval
- Storage and persistence operations

### Utilities
- Image preprocessing and validation
- FAISS index management
- Configuration handling
- Error handling and logging

### API Integration
- REST API endpoint testing
- Request/response validation
- Error condition handling
- Authentication and authorization

## Testing Strategy

### Unit Test Principles
1. **Isolation**: Each test runs independently with mocked dependencies
2. **Fast Execution**: Tests complete quickly for rapid feedback
3. **Deterministic**: Tests produce consistent results across runs
4. **Comprehensive**: All public methods and edge cases are tested

### Mocking Strategy
- External services (DeepFace, FAISS) are mocked
- File system operations use temporary directories
- Database operations use in-memory storage
- Network calls are intercepted and mocked

### Test Data
- Real image files for integration scenarios
- Synthetic data for unit test isolation
- Edge cases and error conditions
- Performance benchmarks for critical paths

## Running Tests

```bash
# Run all user gallery tests
pytest tests/unit/test_user_gallery/ -v

# Run specific test modules
pytest tests/unit/test_user_gallery/test_models.py -v
pytest tests/unit/test_user_gallery/test_services.py -v

# Run with coverage
pytest tests/unit/test_user_gallery/ --cov=src.user_gallery -v

# Run with performance profiling
pytest tests/unit/test_user_gallery/ --profile -v
```

## Test Configuration

Tests use pytest fixtures for common setup and teardown:
- Temporary directory management
- Mock service configuration
- Test data generation
- Performance measurement

## Continuous Integration

Tests are executed in CI/CD pipeline with:
- Unit test execution on every commit
- Integration test execution on pull requests
- Performance regression detection
- Coverage reporting and thresholds