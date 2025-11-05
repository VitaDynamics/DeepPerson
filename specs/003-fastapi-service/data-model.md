# Data Model: FastAPI Service

**Date**: 2025-11-05
**Feature**: FastAPI Stateless Service for DeepPerson

## Core Entities

### 1. ServiceConfiguration

**Purpose**: Configuration settings for the FastAPI service

**Attributes**:
- `host: str` - Server host address (default: "0.0.0.0")
- `port: int` - Server port (default: 8000)
- `gallery_storage_path: Path` - Gallery data storage directory
- `max_image_size: int` - Maximum image upload size in bytes (10MB default)
- `log_level: str` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `workers: int` - Number of Uvicorn workers
- `enable_cors: bool` - CORS enabled flag
- `model_cache_enabled: bool` - Model caching enabled

**State Transitions**: None (static configuration on startup)

---

### 2. ImageRequest

**Purpose**: Request payload for image-based operations

**Attributes**:
- `image: Union[UploadFile, str]` - Image file (multipart) or base64 string
- `image_url: Optional[str]` - Future: Cloud storage URL (deferred)
- `batch_id: Optional[str]` - Unique identifier for batch tracking
- `metadata: Optional[dict]` - Optional metadata key-value pairs

**Validation Rules**:
- Image is required (one of image or image_url)
- Image size ≤ max_image_size
- Supported formats: JPEG, PNG, WEBP
- Base64 strings must be valid

---

### 3. EmbeddingResponse

**Purpose**: Response from `/represent` endpoint

**Attributes**:
- `subjects: List[PersonEmbedding]` - List of detected persons
- `warnings: Optional[List[str]]` - Warning messages
- `model_info: ModelInfo` - Model metadata
- `face_model_info: Optional[ModelInfo]` - Face model metadata (if generated)
- `request_id: str` - Unique request identifier

**Sub-entity: PersonEmbedding**:
- `embedding: List[float]` - Body embedding vector (2048-dim)
- `face_embedding: Optional[List[float]]` - Face embedding vector (512-dim)
- `metadata: EmbeddingMetadata` - Embedding metadata

**Sub-entity: EmbeddingMetadata**:
- `bbox: List[int]` - Bounding box [x, y, width, height]
- `confidence: float` - Detection confidence (0-1)
- `modality: str` - "BODY_ONLY" or "BODY_FACE"
- `source_image: str` - Source image identifier
- `timestamp: datetime` - Generation timestamp

**Sub-entity: ModelInfo**:
- `name: str` - Model name
- `device: str` - Device used (cpu/cuda:0)
- `feature_dim: int` - Embedding dimension

---

### 4. VerificationResponse

**Purpose**: Response from `/verify` endpoint

**Attributes**:
- `verified: bool` - Same person check result
- `distance: float` - Distance between embeddings (backward compatibility)
- `threshold: float` - Verification threshold used
- `distance_metric: str` - Metric used (cosine, euclidean, euclidean_l2)
- `model: str` - Model name used
- `facial_areas: Dict[str, List[int]]` - Bounding boxes for both images
- `body_distance: float` - Body embedding distance
- `face_distance: Optional[float]` - Face embedding distance (if available)
- `fusion_score: Optional[float]` - Multi-modal fusion score
- `used_fusion: bool` - Whether fusion was used
- `modality_available: Dict[str, bool]` - Available modalities
- `request_id: str` - Unique request identifier

---

### 5. Gallery

**Purpose**: User gallery data structure

**Attributes**:
- `user_id: str` - Unique gallery identifier
- `name: Optional[str]` - Display name
- `created_at: datetime` - Creation timestamp
- `updated_at: datetime` - Last update timestamp
- `status: GalleryStatus` - ACTIVE, INACTIVE, PENDING_VERIFICATION
- `metadata: Dict[str, Any]` - Additional gallery metadata
- `total_images: int` - Total images in gallery
- `total_embeddings: int` - Total embeddings generated
- `modality_breakdown: Dict[str, int]` - Count by modality

**Validation Rules**:
- user_id must be unique
- user_id format: alphanumeric + underscore
- Maximum gallery count per instance: configurable

**State Transitions**:
```
PENDING_VERIFICATION → ACTIVE (after first embeddings)
PENDING_VERIFICATION → INACTIVE (manual deactivation)
ACTIVE → INACTIVE (manual deactivation)
INACTIVE → ACTIVE (manual reactivation)
```

---

### 6. GalleryImage

**Purpose**: Individual image within a gallery

**Attributes**:
- `image_id: str` - Unique image identifier
- `user_id: str` - Associated gallery ID
- `file_path: str` - Stored image path
- `original_filename: str` - Uploaded filename
- `modality_hint: Optional[Modality]` - BODY or FACE hint
- `processing_status: ProcessingStatus` - PENDING, PROCESSED, FAILED
- `created_at: datetime` - Upload timestamp
- `metadata: Dict[str, Any]` - Image metadata

**Sub-entity: ProcessingStatus**:
- `status: str` - PENDING, PROCESSED, FAILED
- `error_message: Optional[str]` - Error if failed
- `processed_at: Optional[datetime]` - Completion timestamp

---

### 7. GallerySearchRequest

**Purpose**: Request for gallery retrieval operations

**Attributes**:
- `probe_image: Union[UploadFile, str]` - Image to search with
- `gallery_name: str` - Target gallery (default: "user_gallery")
- `top_k: int` - Number of results (default: 10, max: 100)
- `min_score: float` - Minimum score threshold (0.0-1.0)
- `fusion_weights: Optional[Dict[str, float]]` - Custom fusion weights
- `include_evidence: bool` - Include evidence images (default: True)

---

### 8. GallerySearchResult

**Purpose**: Gallery search response

**Attributes**:
- `results: List[SearchMatch]` - Ranked matching results
- `probe_id: str` - Unique probe identifier
- `processing_time_ms: float` - Total processing time
- `fusion_weights: Dict[str, float]` - Applied fusion weights
- `total_results: int` - Number of results returned
- `face_embedding_used: bool` - Whether face embeddings were used

**Sub-entity: SearchMatch**:
- `user_id: str` - Matched gallery ID
- `overall_score: float` - Combined similarity score (0-1)
- `body_score: float` - Body similarity score
- `face_score: Optional[float]` - Face similarity score (if available)
- `rank: int` - Position in results (1-indexed)
- `evidence: List[EvidenceItem]` - Supporting evidence

**Sub-entity: EvidenceItem**:
- `image_id: str` - Reference image ID
- `score: float` - Similarity score
- `image_path: str` - Evidence image path
- `modality: str` - Evidence modality

---

### 9. HealthStatus

**Purpose**: Service health check response

**Attributes**:
- `status: HealthLevel` - healthy, degraded, unhealthy
- `timestamp: datetime` - Check timestamp
- `uptime_seconds: int` - Service uptime
- `models: Dict[str, ModelStatus]` - Model loading status
- `hardware: HardwareInfo` - Hardware utilization
- `version: str` - Service version

**Sub-entity: ModelStatus**:
- `status: str` - loaded, loading, failed
- `load_time_ms: Optional[int]` - Load duration
- `error: Optional[str]` - Error if failed

**Sub-entity: HardwareInfo**:
- `device: str` - Primary device (cpu/cuda:0)
- `memory_used: str` - Memory used (GB)
- `memory_total: str` - Total memory (GB)
- `gpu_utilization: Optional[int]` - GPU usage percentage

---

### 10. ErrorResponse

**Purpose**: Standardized error response format

**Attributes**:
- `error: ErrorDetail` - Error details object
- `request_id: str` - Unique request identifier

**Sub-entity: ErrorDetail**:
- `code: str` - Machine-readable error code
- `message: str` - Human-readable error message
- `details: Optional[Dict[str, Any]]` - Additional error context

**Common Error Codes**:
- `INVALID_IMAGE_FORMAT` - Unsupported image type
- `IMAGE_TOO_LARGE` - Exceeds size limit
- `NO_PERSON_DETECTED` - No persons found in image
- `GALLERY_NOT_FOUND` - Gallery doesn't exist
- `GALLERY_ALREADY_EXISTS` - Duplicate gallery creation
- `MODEL_NOT_LOADED` - Model loading failed
- `INVALID_REQUEST` - Malformed request data
- `INTERNAL_ERROR` - Unexpected server error

---

## Entity Relationships

```
ServiceConfiguration (1) → (N) Gallery
Gallery (1) → (N) GalleryImage
Gallery (1) → (N) EmbeddingResponse
Request (1) → (1) HealthStatus
Request (1) → (1) ErrorResponse
```

## Data Validation Rules

### Image Validation
- Supported formats: JPEG, PNG, WEBP
- Maximum size: 10MB
- Minimum dimensions: 32x32 pixels
- Maximum dimensions: 4096x4096 pixels

### Gallery Validation
- user_id: 3-64 characters, alphanumeric + underscore
- name: 0-128 characters
- Maximum galleries: 10,000 per instance
- Images per gallery: unlimited

### Batch Processing
- Maximum batch size: 32 images
- Batch timeout: 300 seconds
- Partial failure handling: Return results for successful images, errors for failed

## Storage Schema

### Gallery Directory Structure
```
galleries/
├── user_gallery/
│   ├── embeddings/
│   │   ├── body_embeddings.npy
│   │   ├── face_embeddings.npy
│   │   └── metadata.pkl
│   ├── images/
│   │   ├── user_001/
│   │   └── user_002/
│   └── config.json
```

### JSON Serialization
All entities support JSON serialization for API responses:
- datetime → ISO 8601 string
- Enum → string value
- numpy arrays → List[float]
- Path → string
