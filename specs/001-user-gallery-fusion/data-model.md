# Data Model: User Gallery Fusion Retrieval

## Core Entities

### 1. UserGallery
**Description**: Represents a unique person within the system with aggregated media and embeddings
**Fields**:
- `user_id` (str): Unique identifier for the user (primary key)
- `name` (str, optional): User's display name
- `metadata` (dict): Additional user information (demographics, status flags, etc.)
- `created_at` (datetime): Gallery creation timestamp
- `updated_at` (datetime): Last modification timestamp
- `status` (enum): ACTIVE, INACTIVE, PENDING_VERIFICATION
- `variant_clusters` (list[VariantCluster]): Appearance variant groupings

**Relationships**:
- 1-to-many with VariantCluster
- 1-to-many with ImageAsset
- 1-to-many with EmbeddingSet

**Validation Rules**:
- `user_id` must be unique and non-empty
- `status` must be one of defined enum values
- At least one VariantCluster must exist

### 2. VariantCluster
**Description**: Captures a specific appearance cluster for a user (e.g., outfit, time period)
**Fields**:
- `cluster_id` (str): Unique identifier within user context
- `user_id` (str): Foreign key to UserGallery
- `cluster_name` (str, optional): Descriptive name (e.g., "Summer Outfit 2024")
- `description` (str, optional): Additional cluster information
- `created_at` (datetime): Cluster creation timestamp
- `image_count` (int): Number of images in this cluster
- `primary_image_id` (str, optional): Representative image for the cluster

**Relationships**:
- Many-to-1 with UserGallery
- 1-to-many with ImageAsset
- 1-to-1 with EmbeddingSet (cluster-level embeddings)

**Validation Rules**:
- `cluster_id` must be unique within user context
- `image_count` must be >= 1
- `primary_image_id` must reference existing image in cluster

### 3. ImageAsset
**Description**: Stores media inputs (body or face) with source descriptors and processing status
**Fields**:
- `image_id` (str): Unique identifier (primary key)
- `user_id` (str): Foreign key to UserGallery
- `cluster_id` (str, optional): Foreign key to VariantCluster
- `image_path` (str): File system path or URL to image
- `modality` (enum): BODY, FACE, UNKNOWN
- `source_camera` (str, optional): Camera identifier or source description
- `capture_time` (datetime, optional): Image capture timestamp
- `metadata` (dict): Additional image-specific metadata
- `processing_status` (enum): PENDING, PROCESSING, COMPLETED, FAILED
- `face_detection_confidence` (float, optional): Confidence score for face detection (0.0-1.0)
- `created_at` (datetime): Asset creation timestamp

**Relationships**:
- Many-to-1 with UserGallery
- Many-to-1 with VariantCluster (optional)
- 1-to-1 with EmbeddingSet

**Validation Rules**:
- `modality` must be one of defined enum values
- `face_detection_confidence` must be between 0.0 and 1.0 if provided
- `processing_status` must be one of defined enum values

### 4. EmbeddingSet
**Description**: Encapsulates body and face embedding vectors with provider metadata
**Fields**:
- `embedding_id` (str): Unique identifier (primary key)
- `image_id` (str): Foreign key to ImageAsset (if image-level)
- `cluster_id` (str, optional): Foreign key to VariantCluster (if cluster-level)
- `user_id` (str): Foreign key to UserGallery
- `body_embedding` (list[float]): 2048-dimensional body embedding vector
- `face_embedding` (list[float], optional): Face embedding vector (if available)
- `embedding_provider` (str): Provider name (e.g., "resnet50_circle_dg", "deepface_facenet")
- `embedding_version` (str): Model version used for generation
- `quality_score` (float): Overall embedding quality (0.0-1.0)
- `generated_at` (datetime): Embedding generation timestamp
- `metadata` (dict): Additional embedding metadata

**Relationships**:
- 1-to-1 with ImageAsset (if image-level)
- 1-to-1 with VariantCluster (if cluster-level)
- Many-to-1 with UserGallery

**Validation Rules**:
- `body_embedding` must be exactly 2048 dimensions
- `quality_score` must be between 0.0 and 1.0
- At least one embedding type (body or face) must be present

### 5. RetrievalProbe
**Description**: Represents a query instance containing image input and derived embeddings
**Fields**:
- `probe_id` (str): Unique identifier (primary key)
- `user_id` (str, optional): User who initiated the probe
- `probe_image_path` (str): File system path to probe image
- `probe_modality` (enum): BODY, FACE, AUTO_DETECT
- `generated_embeddings` (dict): Generated embeddings by modality
- `fusion_weights` (dict): Applied fusion weights for this probe
- `retrieval_config` (dict): Configuration used for this retrieval
- `created_at` (datetime): Probe creation timestamp

**Relationships**:
- 1-to-many with RetrievalResult

**Validation Rules**:
- `probe_modality` must be one of defined enum values
- `generated_embeddings` must contain at least one modality

### 6. RetrievalResult
**Description**: Stores retrieval results with user-level scores and evidence
**Fields**:
- `result_id` (str): Unique identifier (primary key)
- `probe_id` (str): Foreign key to RetrievalProbe
- `user_id` (str): Matched user identifier
- `overall_score` (float): Final fusion score (0.0-1.0)
- `face_score` (float, optional): Face modality score contribution
- `body_score` (float, optional): Body modality score contribution
- `face_weight` (float): Weight applied to face score
- `body_weight` (float): Weight applied to body score
- `evidence_images` (list[str]): Image IDs contributing to the match
- `confidence_level` (enum): HIGH, MEDIUM, LOW
- `retrieved_at` (datetime): Result generation timestamp

**Relationships**:
- Many-to-1 with RetrievalProbe

**Validation Rules**:
- `overall_score` must be between 0.0 and 1.0
- `face_weight` + `body_weight` must equal 1.0
- `confidence_level` must be one of defined enum values

## Database Schema Considerations

### Indexing Strategy
- Primary index on `user_id` for UserGallery
- Composite index on `(user_id, cluster_id)` for VariantCluster
- Index on `processing_status` for ImageAsset (for batch processing)
- Index on `modality` for ImageAsset (for modality-specific queries)
- Composite index on `(user_id, generated_at)` for EmbeddingSet (for versioning)

### Storage Optimization
- Body embeddings: 2048 * 4 bytes = ~8KB per embedding
- Face embeddings: Varies by model (typically 512-2048 dimensions)
- Gallery storage format maintains backward compatibility with existing numpy arrays
- Metadata stored in JSON format for flexibility

### State Transitions

#### ImageAsset Processing States
```
PENDING → PROCESSING → COMPLETED
           ↓
         FAILED
```

#### UserGallery Status States
```
PENDING_VERIFICATION ↔ ACTIVE ↔ INACTIVE
```

## API Contract Entities

### UserGalleryRequest
- `user_id` (required): Unique user identifier
- `name` (optional): User display name
- `metadata` (optional): Additional user information
- `images` (required): List of ImageAsset specifications

### UserGalleryResponse
- `user_id`: Unique user identifier
- `status`: Current gallery status
- `created_at`: Gallery creation timestamp
- `cluster_count`: Number of variant clusters
- `image_count`: Total number of images
- `embedding_count`: Number of successfully generated embeddings

### RepresentRequest
- `user_id` (required): Target user for representation
- `generate_face_embeddings` (optional): Flag to enable face embedding generation
- `force_regenerate` (optional): Force regeneration of existing embeddings

### RepresentResponse
- `user_id`: Target user identifier
- `processed_images`: Number of images processed
- `generated_embeddings`: Number of embeddings generated
- `face_embeddings_generated`: Number of face embeddings generated
- `processing_time`: Total processing time in milliseconds
- `errors`: List of processing errors (if any)

### RetrieveRequest
- `probe_image` (required): Path to probe image
- `top_k` (optional): Number of results to return (default: 10)
- `min_score` (optional): Minimum score threshold (default: 0.0)
- `include_evidence` (optional): Include evidence images in response (default: true)

### RetrieveResponse
- `probe_id`: Unique probe identifier
- `results`: List of RetrievalResult objects
- `processing_time`: Total processing time in milliseconds
- `fusion_weights`: Applied fusion weights for this query

## Validation and Constraints

### Business Rules
1. A user must have at least one variant cluster
2. Each variant cluster must contain at least one image
3. Image assets must have valid processing status
4. Embeddings must have valid quality scores
5. Fusion weights must sum to 1.0
6. Confidence levels must be properly categorized

### Data Integrity
1. Foreign key relationships maintained
2. Unique constraints on primary keys
3. Required fields must be present
4. Enum values must be valid
5. Timestamps must be valid datetime objects

This data model provides a comprehensive foundation for user gallery fusion retrieval while maintaining backward compatibility with existing DeepPerson functionality.