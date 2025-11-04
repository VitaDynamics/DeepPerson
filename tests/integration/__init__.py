# Integration Tests

This module contains integration tests for the user gallery fusion component.

## Test Structure

```
tests/integration/
├── __init__.py                          # Module initialization
├── test_user_gallery_workflow.py       # End-to-end gallery workflow tests
├── test_fusion_retrieval.py            # Multi-modal retrieval integration tests
├── test_embedding_generation.py        # Face and body embedding integration
├── test_gallery_storage.py             # Storage and persistence integration
├── test_api_integration.py             # Full API integration tests
└── test_performance.py                 # Performance and scalability tests
```

## Test Coverage

### Workflow Integration
- Complete user gallery creation workflows
- Multi-modal embedding generation pipelines
- Fusion retrieval end-to-end scenarios
- Gallery management and lifecycle operations

### Component Integration
- DeepFace face embedding integration
- FAISS similarity search integration
- YOLO person detection integration
- Storage backend integration
- API endpoint integration

### Data Flow Integration
- Image preprocessing and validation pipelines
- Embedding generation and quality scoring
- Variant clustering and aggregation
- Result ranking and evidence tracking

### Performance Integration
- Large gallery handling (10k+ users)
- Multi-modal search performance
- Memory usage and optimization
- Batch processing scenarios

## Testing Strategy

### Integration Test Principles
1. **Real Components**: Use real implementations where possible
2. **Minimal Mocking**: Only mock external dependencies
3. **Real Data**: Use real images and datasets
4. **Performance Focus**: Measure execution time and resource usage

### Test Environment
- Real image datasets for testing
- Local FAISS index for search testing
- DeepFace model downloads for face recognition
- GPU acceleration where available

### Data Management
- Test datasets with known characteristics
- Performance benchmark datasets
- Edge case and failure scenario data
- Cleanup and reset between tests

## Running Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific integration test modules
pytest tests/integration/test_user_gallery_workflow.py -v
pytest tests/integration/test_fusion_retrieval.py -v

# Run with performance measurement
pytest tests/integration/ --benchmark -v

# Run with real DeepFace models
pytest tests/integration/ --use-real-deepface -v
```

## Test Configuration

Integration tests use pytest fixtures for environment setup:
- Real model downloads and caching
- Test dataset management
- Performance measurement setup
- Cleanup and teardown procedures

## Continuous Integration

Integration tests are executed in CI/CD pipeline with:
- Integration test execution on pull requests
- Performance regression detection
- Real model testing in dedicated environments
- Resource usage monitoring