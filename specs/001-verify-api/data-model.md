# Data Model: Image Verification API

**Feature**: Image Verification API Enhancement
**Date**: 2025-11-05

## Overview

This document defines the data models and entity structures for the enhanced Image Verification API supporting fusion scoring (body+face embeddings) with body-only fallback.

## Core Entities

### 1. VerificationRequest

**Description**: Input parameters for the verify() API method

**Attributes**:
- `img1_path`: Union[str, Path] - Path to first image
- `img2_path`: Union[str, Path] - Path to second image
- `model_name`: Optional[str] - Override model name (default: instance default)
- `detector_backend`: Optional[str] - Override detector backend (default: instance default)
- `distance_metric`: Literal["cosine", "euclidean", "euclidean_l2"] - Distance metric for comparison
- `threshold`: Optional[float] - Custom threshold (default: from registry)
- `normalization`: Literal["base", "resnet", "circle"] - Embedding normalization method
- `enforce_detection`: bool - Raise error on no detection (default: True)

**Validation Rules**:
- Image paths must exist and be readable
- Threshold must be >= 0 if provided
- Distance metric must be one of supported values
- Normalization must be one of supported values

### 2. VerificationResult

**Description**: Response from verify() API method

**Attributes**:
- `verified`: bool - True if same person (score >= threshold or distance <= threshold)
- `distance`: float - Body embedding distance between images
- `threshold`: float - Threshold used for verification
- `distance_metric`: str - Distance metric used for comparison
- `model`: str - Model name used for embedding generation
- `detector_backend`: str - Detector backend used
- `facial_areas`: Dict[str, Optional[BoundingBox]] - Bounding boxes for detected subjects
  - `img1`: Optional[BoundingBox] - Bounding box for first image
  - `img2`: Optional[BoundingBox] - Bounding box for second image

**Enhanced Attributes** (New):
- `body_distance`: float - Body embedding distance (same as `distance`)
- `face_distance`: Optional[float] - Face embedding distance (if face embeddings available)
- `fusion_score`: Optional[float] - Combined similarity score from fusion (0.0-1.0)
- `face_weight`: float - Weight applied to face modality (default: 0.5)
- `body_weight`: float - Weight applied to body modality (default: 0.5)
- `used_fusion`: bool - True if fusion scoring was used, False if body-only
- `modality_available`: Dict[str, bool] - Which modalities were available
  - `body`: bool - Always True (primary requirement)
  - `face`: bool - True if face embedding generated
- `warnings`: Optional[List[str]] - Warning messages (e.g., missing face, multiple detections)

**Validation Rules**:
- `verified` must be boolean
- `distance` must be >= 0
- `threshold` must be >= 0
- `fusion_score` must be None or 0.0 <= score <= 1.0
- `face_weight` + `body_weight` must equal 1.0
- `facial_areas["img1"]` and `facial_areas["img2"]` must be valid BoundingBox or None

### 3. BoundingBox

**Description**: Rectangle coordinates for detected subject location

**Attributes**:
- `x1`: float - Top-left x coordinate
- `y1`: float - Top-left y coordinate
- `x2`: float - Bottom-right x coordinate
- `y2`: float - Bottom-right y coordinate

**Validation Rules**:
- `x2` > `x1`
- `y2` > `y1`
- All coordinates must be >= 0

**Example**:
```python
{
    "x1": 100.5,
    "y1": 50.2,
    "x2": 300.8,
    "y2": 450.3
}
```

### 4. EmbeddingVector

**Description**: Numerical representation of person features

**Attributes**:
- `vector`: np.ndarray - 1D numpy array of features
- `dimension`: int - Length of vector (e.g., 2048 for ResNet50)
- `normalization`: str - Normalization method applied
- `modality`: str - "BODY" or "FACE"

**Validation Rules**:
- Vector must be 1D numpy array
- Dimension must match vector length
- Normalization must be one of: "base", "resnet", "circle"
- Modality must be one of: "BODY", "FACE"

### 5. PersonDetection

**Description**: Detected person in an image with metadata

**Attributes**:
- `bbox`: BoundingBox - Location in image
- `confidence`: float - Detection confidence (0.0-1.0)
- `embedding`: EmbeddingVector - Generated body embedding
- `face_embedding`: Optional[EmbeddingVector] - Generated face embedding (if available)
- `source_image`: str - Path to source image

**Validation Rules**:
- `confidence` must be 0.0 <= confidence <= 1.0
- `embedding.modality` must be "BODY"
- If `face_embedding` present, `face_embedding.modality` must be "FACE"

## State Transitions

### Verification Flow

```
State: START
  ↓
State: IMAGE_LOADING
  ↓ (both images loaded successfully)
State: PERSON_DETECTION
  ↓ (persons detected in both images)
State: EMBEDDING_GENERATION
  ↓ (embeddings generated)
State: FUSION_SCORING
  ↓ (fusion score calculated or fallback to body-only)
State: THRESHOLD_COMPARISON
  ↓ (verified result determined)
State: RESULT_FORMATTING
  ↓
State: COMPLETE
```

### Fusion Scoring States

**State: FULL_FUSION**
- Trigger: Both images have face embeddings
- Action: Compute body_distance, face_distance, fusion_score
- Result: used_fusion = True

**State: BODY_ONLY_FALLBACK**
- Trigger: One or both images lack face embeddings
- Action: Compute body_distance only
- Result: used_fusion = False, fusion_score = None

## Entity Relationships

```
VerificationRequest
    ↓ (input to)
verify() Method
    ↓ (produces)
VerificationResult
    ↓ (contains)
├── FacialAreas (BoundingBox x2)
├── ModalityAvailability
└── Scores (body, face, fusion)
```

## Data Flow

```
Input Images
    ↓
represent() API
    ↓ (produces)
PersonDetection[]
    ↓ (extract primary)
PersonDetection (img1) + PersonDetection (img2)
    ↓
Embedding Extraction
    ↓ (produces)
Body Embeddings + Face Embeddings (if available)
    ↓
Distance Computation
    ↓ (produces)
body_distance + face_distance (if available)
    ↓
Fusion Scoring (conditional)
    ↓ (produces)
fusion_score (if used_fusion)
    ↓
Threshold Comparison
    ↓ (produces)
verified boolean
    ↓
VerificationResult
```

## Validation Rules Summary

### Input Validation
- Image files must exist and be readable
- Image formats: JPEG, PNG, BMP, TIFF (based on PIL support)
- Threshold: >= 0 (if provided)
- Distance metric: one of ["cosine", "euclidean", "euclidean_l2"]
- Normalization: one of ["base", "resnet", "circle"]
- Enforce_detection: boolean

### Output Validation
- verified: boolean
- distance: float >= 0
- threshold: float >= 0
- fusion_score: None or 0.0 <= score <= 1.0
- face_weight + body_weight = 1.0
- facial_areas: dict with "img1" and "img2" keys
- modality_available: dict with "body" and "face" keys

### Error Conditions
- **NoPersonDetected**: No person found in image (if enforce_detection=True)
- **MultiplePersonsDetected**: Multiple persons detected, using primary (warning)
- **FileNotFoundError**: Image file doesn't exist
- **ValueError**: Invalid parameter values
- **ImportError**: Face embedding dependencies not available (graceful fallback)

## Backward Compatibility

**Existing Fields** (Maintained):
- `verified`: Same boolean result
- `distance`: Same body distance value
- `threshold`: Same threshold value
- `distance_metric`: Same metric string
- `model`: Same model name
- `detector_backend`: Same backend string
- `facial_areas`: Same structure with bbox coordinates
- `warnings`: Same warning list format

**New Fields** (Additive):
- All new fields are optional or have default values
- Existing code ignoring new fields continues to work
- Type hints and documentation updated

## Example Structures

### Minimal Verification Result (Body-Only Fallback)

```python
{
    "verified": True,
    "distance": 0.342,
    "threshold": 0.5,
    "distance_metric": "cosine",
    "model": "resnet50_circle_dg",
    "detector_backend": "yolo",
    "facial_areas": {
        "img1": {"x1": 100, "y1": 50, "x2": 300, "y2": 450},
        "img2": {"x1": 120, "y1": 55, "x2": 320, "y2": 455}
    },
    "body_distance": 0.342,
    "face_distance": None,
    "fusion_score": None,
    "face_weight": 0.5,
    "body_weight": 0.5,
    "used_fusion": False,
    "modality_available": {
        "body": True,
        "face": False
    }
}
```

### Full Fusion Verification Result

```python
{
    "verified": True,
    "distance": 0.342,  # body_distance alias
    "threshold": 0.6,
    "distance_metric": "cosine",
    "model": "resnet50_circle_dg",
    "detector_backend": "yolo",
    "facial_areas": {
        "img1": {"x1": 100, "y1": 50, "x2": 300, "y2": 450},
        "img2": {"x1": 120, "y1": 55, "x2": 320, "y2": 455}
    },
    "body_distance": 0.342,
    "face_distance": 0.156,
    "fusion_score": 0.751,
    "face_weight": 0.5,
    "body_weight": 0.5,
    "used_fusion": True,
    "modality_available": {
        "body": True,
        "face": True
    },
    "warnings": []
}
```

## Summary

The enhanced VerificationResult structure:
- **Maintains backward compatibility** with existing fields
- **Adds fusion scoring transparency** with separate body/face distances and fusion score
- **Includes modality availability** to understand what data was used
- **Provides clear fallback indication** via used_fusion boolean
- **Enables fine-grained analysis** of verification decisions
