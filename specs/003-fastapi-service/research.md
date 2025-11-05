# Phase 0 Research: FastAPI Service Implementation

**Date**: 2025-11-05
**Feature**: FastAPI Stateless Service for DeepPerson

## Technical Decisions

### FastAPI Framework Selection

**Decision**: FastAPI as the web framework
**Rationale**:
- Native OpenAPI/Swagger documentation generation (FR-009)
- High performance comparable to Node.js and Go
- Built-in request validation with Pydantic
- Automatic API documentation at `/docs` and `/openapi.json`
- Excellent async support for concurrent request handling
- Type hints integration aligns with existing codebase standards

**Alternatives considered**:
- Flask: Requires manual OpenAPI generation, more boilerplate
- Django REST: Overkill for stateless microservice, heavier dependencies

### Pydantic for Schema Validation

**Decision**: Use Pydantic models for request/response validation
**Rationale**:
- Automatic validation of input parameters (FR-012)
- Built-in OpenAPI schema generation
- Type safety with Python 3.12+
- Clear error messages for validation failures
- Integration with FastAPI is seamless

### Stateless Architecture

**Decision**: Pure stateless design with external galleries directory
**Rationale**:
- Horizontal scaling support - any instance can handle any request
- No session affinity required in load balancers
- Gallery data persists across service restarts (SC-006)
- Simplified deployment and recovery
- Aligns with container/cloud deployment patterns

### Image Input Methods

**Decision**: Support both multipart/form-data and base64-encoded images
**Rationale**:
- multipart/form-data: Standard for file uploads, works with curl, Postman, browsers
- base64: Enables direct JSON requests, useful for microservices communication
- Future cloud file path support planned (deferred per clarification)

### Error Handling Strategy

**Decision**: Standardized error format with code, message, details
**Rationale**:
- Consistent client error handling
- Follows RFC 7807 Problem Details pattern
- Enables automated error handling in client code
- Provides debugging information without exposing internals

**Error Response Format**:
```json
{
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "The provided image format is not supported",
    "details": {
      "supported_formats": ["JPEG", "PNG", "WEBP"],
      "received": "GIF"
    }
  }
}
```

### Concurrent Request Handling

**Decision**: Async endpoints with thread-safe DeepPerson library usage
**Rationale**:
- DeepPerson's ModelRegistry already implements thread safety
- Async allows efficient handling of I/O-bound operations (image processing)
- Support 100+ concurrent requests (SC-003)
- GPU operations remain synchronous but can be offloaded to thread pool

### Logging Strategy

**Decision**: Structured logging with contextual information
**Rationale**:
- Request ID correlation for tracing
- Performance metrics (timing, model load times)
- Error context for debugging
- Integration with monitoring tools
- Levels: DEBUG (development), INFO (requests), WARNING (recoverable), ERROR (failures)

**Log Format**:
```
2025-11-05T10:30:45Z INFO request_id=req-123 method=POST endpoint=/represent duration=1.234s status=200
```

### Health Check Implementation

**Decision**: Multi-level health checks
**Rationale**:
- **Liveness**: Service process running (basic ping)
- **Readiness**: Models loaded, GPU available, storage accessible
- **Performance**: Response time < 100ms (SC-005)

**Endpoint**: GET `/health`
**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-05T10:30:45Z",
  "uptime_seconds": 3600,
  "models": {
    "body_model": "loaded",
    "face_model": "loaded"
  },
  "hardware": {
    "device": "cuda:0",
    "memory_used": "2.1GB",
    "memory_total": "8.0GB"
  },
  "version": "1.0.0"
}
```

## Research Findings

### FastAPI Best Practices

1. **Dependency Injection**: Use FastAPI's built-in dependency system for DeepPerson initialization
2. **Background Tasks**: For long-running operations (gallery processing)
3. **Streaming Responses**: For large batch results
4. **Middleware**: For request logging, CORS, and error handling

### Performance Optimization

1. **Model Caching**: Leverage existing ModelRegistry caching
2. **Batch Processing**: Group multiple images per request (FR-008)
3. **Connection Pooling**: Not needed (stateless, no database)
4. **GPU Memory Management**: Use existing DeepPerson cleanup mechanisms

### Security Considerations

1. **Input Validation**: Strict file type and size checks
2. **Path Traversal**: Validate gallery IDs, sanitize paths
3. **Rate Limiting**: Deferred to future (FR-014 mentions future consideration)
4. **CORS**: Configurable per deployment (FR-010)

## Deployment Architecture

### Container Deployment

**Decision**: Standard container with environment-based configuration
**Rationale**:
- **Port**: Configurable via PORT env var (default 8000)
- **Workers**: Use Uvicorn with multiple workers for CPU-bound tasks
- **Storage**: Mount galleries directory as volume
- **Health**: Kubernetes/healthcheck endpoints

**Example Dockerfile structure**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e .
EXPOSE 8000
CMD ["uvicorn", "src.api_service:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

**Configuration**:
- `PORT`: Service port (default: 8000)
- `GALLERY_STORAGE_PATH`: Gallery directory (default: ./galleries)
- `MAX_IMAGE_SIZE`: Maximum upload size in bytes (default: 10MB)
- `LOG_LEVEL`: Logging level (default: INFO)
- `WORKERS`: Number of uvicorn workers (default: 1)

## Testing Strategy

### Unit Tests
- Schema validation
- Error handling
- Utility functions

### Integration Tests
- API endpoint testing with sample images
- Gallery CRUD operations
- Concurrent request handling

### Contract Tests
- OpenAPI schema validation
- Response format verification
- Error response structure

## Summary

All technical decisions align with the feature specification and constitutional requirements. The implementation will leverage existing DeepPerson components while adding a robust HTTP service layer with comprehensive documentation, error handling, and observability.
