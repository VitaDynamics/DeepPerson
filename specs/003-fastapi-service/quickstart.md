# Quick Start Guide: FastAPI Service

**Feature**: FastAPI Stateless Service for DeepPerson
**Date**: 2025-11-05

## Overview

The DeepPerson FastAPI service provides HTTP endpoints for person recognition and embedding generation. This guide helps you get started quickly.

**Note**: Gallery management functionality will be implemented in a future release with OceanBase Dataset integration. Current MVP focuses on Core API and Service Health/Documentation only.

## Installation

### Prerequisites
- Python 3.12+
- 4GB RAM minimum (8GB+ recommended)
- GPU optional but recommended for optimal performance

### Install Dependencies

```bash
# From repository root
pip install fastapi uvicorn python-multipart

# Install DeepPerson in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "import fastapi; print(f'FastAPI {fastapi.__version__} installed')"
```

## Running the Service

### Basic Start

```bash
# From repository root
python main.py

# Or with uvicorn directly
uvicorn src.api_service:app --host 0.0.0.0 --port 8000 --reload
```

### With Custom Configuration

```bash
# Set environment variables
export PORT=8080
export GALLERY_STORAGE_PATH=/data/galleries
export LOG_LEVEL=DEBUG

# Start service
uvicorn src.api_service:app --host 0.0.0.0 --port $PORT
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Service port |
| `MAX_IMAGE_SIZE` | 10485760 | Max image size in bytes (10MB) |
| `LOG_LEVEL` | INFO | Logging level |
| `WORKERS` | 1 | Number of uvicorn workers |

## Service Verification

### 1. Check Health

```bash
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-05T10:30:45Z",
  "uptime_seconds": 15,
  "models": {
    "body_model": {"status": "loaded"},
    "face_model": {"status": "loaded"}
  },
  "hardware": {
    "device": "cuda:0",
    "memory_used": "2.1GB",
    "memory_total": "8.0GB"
  },
  "version": "1.0.0"
}
```

### 2. View API Documentation

Open browser to: http://localhost:8000/docs

This provides interactive API documentation with:
- All endpoints listed
- Request/response schemas
- Try-it-out functionality

### 3. View OpenAPI Spec

```bash
curl http://localhost:8000/openapi.json | jq .
```

## Core API Usage

### Generate Person Embeddings

```bash
# Upload an image and get embeddings
curl -X POST "http://localhost:8000/represent" \
  -F "image=@/path/to/person.jpg" \
  -F "generate_face_embeddings=true" \
  | jq .
```

**Response**:
```json
{
  "subjects": [
    {
      "embedding": [0.123, -0.456, ...],
      "face_embedding": [0.789, -0.321, ...],
      "metadata": {
        "bbox": [100, 150, 200, 300],
        "confidence": 0.95,
        "modality": "BODY_FACE"
      }
    }
  ],
  "model_info": {
    "name": "resnet50_circle_dg",
    "device": "cuda:0",
    "feature_dim": 2048
  }
}
```

### Verify Identity

```bash
# Compare two images
curl -X POST "http://localhost:8000/verify" \
  -F "img1_path=@/path/to/person1.jpg" \
  -F "img2_path=@/path/to/person2.jpg" \
  | jq .
```

**Response**:
```json
{
  "verified": true,
  "distance": 0.234,
  "threshold": 0.5,
  "fusion_score": 0.87,
  "used_fusion": true,
  "modality_available": {
    "body": true,
    "face": true
  }
}
```

### Batch Processing

```bash
# Process multiple images
curl -X POST "http://localhost:8000/represent" \
  -F "image=@/path/to/image1.jpg" \
  -F "image=@/path/to/image2.jpg" \
  -F "image=@/path/to/image3.jpg" \
  -F "batch_id=batch_001" \
  | jq .
```

## Base64 Image Support

### Represent with Base64

```bash
# Send base64-encoded image
curl -X POST "http://localhost:8000/represent" \
  -H "Content-Type: application/json" \
  -d '{
    "base64_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "generate_face_embeddings": true
  }'
```

### Verify with Base64

```bash
# Send both images as base64
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "base64_img1": "data:image/jpeg;base64,/9j/4AAQ...",
    "base64_img2": "data:image/jpeg;base64,/9j/4AAQ...",
    "distance_metric": "cosine"
  }'
```

## Error Handling

All errors follow standardized format:

```json
{
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "The provided image format is not supported",
    "details": {
      "supported_formats": ["JPEG", "PNG", "WEBP"],
      "received": "GIF"
    }
  },
  "request_id": "req-12345"
}
```

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `INVALID_IMAGE_FORMAT` | Unsupported file type | Use JPEG, PNG, or WEBP |
| `IMAGE_TOO_LARGE` | Exceeds size limit | Reduce image size < 10MB |
| `NO_PERSON_DETECTED` | No persons found | Check image quality/lighting |
| `MODEL_NOT_LOADED` | Model loading failed | Check model download/storage |

## Performance Tips

### 1. Use GPU for Faster Processing

```bash
# Check GPU availability
nvidia-smi

# Service auto-detects GPU, no config needed
```

### 2. Batch Multiple Images

```bash
# Process multiple images in one request (faster)
curl -X POST "http://localhost:8000/represent" \
  -F "image=@img1.jpg" \
  -F "image=@img2.jpg" \
  -F "image=@img3.jpg"
```

### 3. Adjust Batch Size

```bash
# For high-throughput scenarios
curl -X POST "http://localhost:8000/represent" \
  -F "image=@large_batch.jpg" \
  -F "batch_size=32"
```

### 4. Enable Face Embeddings Selectively

```bash
# Faster body-only processing
curl -X POST "http://localhost:8000/represent" \
  -F "image=@person.jpg" \
  -F "generate_face_embeddings=false"
```

## Testing

### Unit Tests

```bash
# Run unit tests
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_api_service.py -v
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v

# Run contract tests
pytest tests/contract/ -v
```

### Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# List API endpoints
curl http://localhost:8000/docs

# Check logs
tail -f logs/service.log
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install fastapi uvicorn python-multipart
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "src.api_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepperson-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepperson-service
  template:
    metadata:
      labels:
        app: deepperson-service
    spec:
      containers:
      - name: service
        image: deepperson:latest
        ports:
        - containerPort: 8000
        env:
        - name: PORT
          value: "8000"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

### Cloud Deployment

- **AWS**: Use ECS or EKS with GPU instances (g4dn.xlarge)
- **GCP**: Use Cloud Run with GPU acceleration
- **Azure**: Use Container Instances with GPU

## Monitoring

### Health Check

```bash
# Simple health check
curl http://localhost:8000/health

# Detailed health check with timeout
curl --max-time 5 http://localhost:8000/health
```

### Logging

Logs are written to:
- `logs/service.log` - Application logs
- `logs/error.log` - Error logs only
- `logs/access.log` - HTTP access logs

### Metrics

Monitor these key metrics:
- Request latency (p50, p95, p99)
- Error rate (4xx, 5xx responses)
- Throughput (requests/second)
- Model load time
- GPU memory usage

## Troubleshooting

### Service Won't Start

```bash
# Check port availability
lsof -i :8000

# Check logs
tail -50 logs/error.log
```

### Slow Performance

```bash
# Check GPU status
nvidia-smi

# Monitor resource usage
htop

# Check model loading
curl http://localhost:8000/health | jq '.models'
```

### Out of Memory

```bash
# Reduce batch size
export MAX_BATCH_SIZE=8

# Reduce workers
export WORKERS=1

# Use smaller images
# Compress images to <5MB
```

## Support

- **Documentation**: http://localhost:8000/docs
- **OpenAPI Spec**: http://localhost:8000/openapi.json
- **Health Check**: http://localhost:8000/health
- **Issues**: See repository issues

## Next Steps

1. Review API documentation: http://localhost:8000/docs
2. Try the interactive examples in Swagger UI
3. Integrate with your application using the OpenAPI spec
4. Set up monitoring and alerting for production

**Future Enhancement**: Gallery management functionality with persistent storage will be implemented in a future release using OceanBase Dataset integration.
