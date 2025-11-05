# Quick Start: Image Verification API

**Feature**: Image Verification API Enhancement
**Date**: 2025-11-05

## Overview

The Image Verification API determines if two images show the same person using multi-modal fusion scoring (body+face embeddings) with automatic fallback to body-only comparison when face data is unavailable.

## Basic Usage

### Simple Verification

```python
from components.deep_person.api import DeepPerson

# Initialize DeepPerson
dp = DeepPerson(model_name="resnet50_circle_dg")

# Verify two images
result = dp.verify(
    img1_path="person1_photo.jpg",
    img2_path="person1_id_card.jpg"
)

# Check result
if result["verified"]:
    print(f"Same person! Distance: {result['distance']:.4f}")
else:
    print(f"Different people. Distance: {result['distance']:.4f}")

# Access detailed information
print(f"Used fusion scoring: {result['used_fusion']}")
print(f"Face embeddings available: {result['modality_available']['face']}")
if result['used_fusion']:
    print(f"Body distance: {result['body_distance']:.4f}")
    print(f"Face distance: {result['face_distance']:.4f}")
    print(f"Fusion score: {result['fusion_score']:.4f}")
```

### Result Structure

```python
{
    "verified": True,                    # Boolean: same person?
    "distance": 0.342,                   # Body embedding distance
    "threshold": 0.5,                    # Threshold used
    "distance_metric": "cosine",         # Metric used
    "model": "resnet50_circle_dg",       # Model name
    "detector_backend": "yolo",          # Detector used
    "facial_areas": {                    # Bounding boxes
        "img1": {"x1": 100, "y1": 50, "x2": 300, "y2": 450},
        "img2": {"x1": 120, "y1": 55, "x2": 320, "y2": 455}
    },
    # New fields (fusion scoring):
    "body_distance": 0.342,              # Same as 'distance'
    "face_distance": 0.156,              # Face distance (or None)
    "fusion_score": 0.751,               # Combined score (or None)
    "face_weight": 0.5,                  # Face weight (0.5 default)
    "body_weight": 0.5,                  # Body weight (0.5 default)
    "used_fusion": True,                 # Fusion used?
    "modality_available": {              # What data was available
        "body": True,
        "face": True
    },
    "warnings": []                       # Any warnings
}
```

## Advanced Usage

### Custom Distance Metric

```python
# Use Euclidean distance instead of default cosine
result = dp.verify(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    distance_metric="euclidean"
)
```

### Custom Threshold

```python
# Use stricter threshold (lower = more strict for distance)
result = dp.verify(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    threshold=0.3  # More conservative
)

# Or more lenient (higher for distance)
result = dp.verify(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    threshold=0.7  # More lenient
)
```

### Custom Fusion Weights

```python
# Emphasize face embeddings (e.g., for ID verification)
result = dp.verify(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    fusion_weights={
        "body": 0.3,   # 30% body
        "face": 0.7    # 70% face
    }
)

# Emphasize body embeddings (e.g., full-body surveillance)
result = dp.verify(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    fusion_weights={
        "body": 0.8,   # 80% body
        "face": 0.2    # 20% face
    }
)
```

### Different Normalization

```python
# Use Circle normalization
result = dp.verify(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    normalization="circle"
)
```

### Graceful Handling of No Detection

```python
# Don't raise error if no person detected
result = dp.verify(
    img1_path="image1.jpg",
    img2_path="image2.jpg",
    enforce_detection=False  # Returns result with warning instead of error
)

if "warnings" in result:
    for warning in result["warnings"]:
        print(f"Warning: {warning}")
```

## Understanding Fusion Scoring

### When Fusion Scoring is Used

Fusion scoring (body + face) is used when **both images** have face embeddings:

```python
# Both images have face data → fusion scoring
result = dp.verify("photo1.jpg", "photo2.jpg")
print(f"Used fusion: {result['used_fusion']}")  # True
print(f"Face available: {result['modality_available']['face']}")  # True
```

**Fusion Process**:
1. Compute body distance: `body_dist = distance(body_emb1, body_emb2)`
2. Compute face distance: `face_dist = distance(face_emb1, face_emb2)`
3. Convert to similarities: `body_sim = 1 - body_dist`, `face_sim = 1 - face_dist`
4. Combine: `fusion_score = (body_sim * body_weight) + (face_sim * face_weight)`
5. Verify: `verified = fusion_score >= threshold`

### When Body-Only Fallback is Used

Body-only comparison is used when **one or both images** lack face embeddings:

```python
# Image 2 has no face → body-only comparison
result = dp.verify("photo1.jpg", "silhouette.jpg")
print(f"Used fusion: {result['used_fusion']}")  # False
print(f"Face available: {result['modality_available']['face']}")  # False
print(f"Fusion score: {result['fusion_score']}")  # None
```

**Body-Only Process**:
1. Compute body distance: `body_dist = distance(body_emb1, body_emb2)`
2. Verify: `verified = body_dist <= threshold`

### Threshold Interpretation

**For Distance Metrics** (cosine, euclidean, euclidean_l2):
- Lower distance = more similar
- **Verified if**: `distance <= threshold`
- Example: `distance=0.3, threshold=0.5` → **Verified** (same person)

**For Fusion Scores** (similarity):
- Higher score = more similar
- **Verified if**: `fusion_score >= threshold`
- Example: `fusion_score=0.75, threshold=0.6` → **Verified** (same person)

## Common Scenarios

### Scenario 1: ID Card Verification

Use face-emphasized fusion for document verification:

```python
result = dp.verify(
    img1_path="selfie.jpg",
    img2_path="id_card_photo.jpg",
    fusion_weights={"body": 0.2, "face": 0.8},  # Emphasize face
    threshold=0.6  # Conservative threshold
)

if result["verified"]:
    print("ID verified - face match confirmed")
```

### Scenario 2: Surveillance Footage

Use body-emphasized fusion for CCTV analysis:

```python
result = dp.verify(
    img1_path="camera1_frame.jpg",
    img2_path="camera2_frame.jpg",
    fusion_weights={"body": 0.8, "face": 0.2},  # Emphasize body
    distance_metric="euclidean"
)

if result["verified"]:
    print(f"Same person across cameras")
    print(f"Body confidence: {1 - result['body_distance']:.3f}")
```

### Scenario 3: Large Dataset Deduplication

Use efficient body-only for large-scale deduplication:

```python
# Body-only is faster and sufficient for duplicate detection
result = dp.verify(
    img1_path="dataset_image_001.jpg",
    img2_path="dataset_image_002.jpg",
    threshold=0.4  # Stricter threshold
)

if result["verified"] and not result["used_fusion"]:
    print("Potential duplicate found (body match)")
```

### Scenario 4: Handling Low-Quality Images

```python
result = dp.verify(
    img1_path="blurry_image.jpg",
    img2_path="clear_image.jpg",
    enforce_detection=False,  # Graceful handling
    threshold=0.7  # More lenient
)

if result["warnings"]:
    print("Issues detected:")
    for warning in result["warnings"]:
        print(f"  - {warning}")
```

## Performance Tips

### 1. Reuse DeepPerson Instance

```python
# ✓ Good: Reuse instance
dp = DeepPerson()
for pair in image_pairs:
    result = dp.verify(pair[0], pair[1])

# ✗ Bad: Creating new instance each time
for pair in image_pairs:
    dp = DeepPerson()  # Slow initialization
    result = dp.verify(pair[0], pair[1])
```

### 2. Use Body-Only for Speed

```python
# Faster: Disable face embeddings for large-scale processing
# Note: This is automatic - face is only generated if requested
# No parameter needed, it's automatic based on detectability

# Result will have used_fusion=False if face unavailable
result = dp.verify("img1.jpg", "img2.jpg")
```

### 3. Batch Verification

```python
import itertools

images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
pairs = list(itertools.combinations(images, 2))

results = []
for img1, img2 in pairs:
    result = dp.verify(img1, img2)
    results.append({
        "pair": (img1, img2),
        "verified": result["verified"],
        "used_fusion": result["used_fusion"]
    })
```

## Error Handling

### Common Errors and Solutions

**Error: No person detected**
```python
try:
    result = dp.verify("empty_image.jpg", "person_image.jpg")
except ValueError as e:
    print(f"Detection error: {e}")
    # Solution: Use enforce_detection=False or check image quality
```

**Error: Image file not found**
```python
try:
    result = dp.verify("missing.jpg", "exists.jpg")
except FileNotFoundError as e:
    print(f"File error: {e}")
    # Solution: Verify file paths exist
```

**Error: Invalid threshold**
```python
try:
    result = dp.verify("img1.jpg", "img2.jpg", threshold=-0.5)
except ValueError as e:
    print(f"Validation error: {e}")
    # Solution: Use threshold >= 0
```

### Best Practices

1. **Always check warnings**:
   ```python
   result = dp.verify("img1.jpg", "img2.jpg")
   if result.get("warnings"):
       # Handle warnings appropriately
   ```

2. **Verify modality availability**:
   ```python
   if not result["modality_available"]["face"]:
       # Face not available, result based on body only
   ```

3. **Use appropriate thresholds**:
   ```python
   # Conservative (low false positives)
   threshold = 0.3  # cosine distance

   # Balanced (default)
   threshold = 0.5  # cosine distance

   # Lenient (low false negatives)
   threshold = 0.7  # cosine distance
   ```

## Examples

### Complete Example: Identity Verification Pipeline

```python
from components.deep_person.api import DeepPerson

def verify_identity(applicant_photo, id_photo, threshold=0.5):
    """
    Verify if applicant photo matches ID photo.

    Args:
        applicant_photo: Path to applicant's photo
        id_photo: Path to ID card photo
        threshold: Verification threshold

    Returns:
        dict: Verification result with detailed scores
    """
    dp = DeepPerson(model_name="resnet50_circle_dg")

    result = dp.verify(
        img1_path=applicant_photo,
        img2_path=id_photo,
        distance_metric="cosine",
        threshold=threshold,
        fusion_weights={"body": 0.3, "face": 0.7},  # Emphasize face
        enforce_detection=True
    )

    # Prepare detailed response
    response = {
        "verified": result["verified"],
        "confidence": result.get("fusion_score") or (1 - result["distance"]),
        "method": "fusion" if result["used_fusion"] else "body-only",
        "scores": {
            "body": {
                "distance": result["body_distance"],
                "similarity": 1 - result["body_distance"]
            }
        },
        "modality_available": result["modality_available"],
        "warnings": result.get("warnings", [])
    }

    if result.get("face_distance"):
        response["scores"]["face"] = {
            "distance": result["face_distance"],
            "similarity": 1 - result["face_distance"]
        }

    return response

# Usage
result = verify_identity(
    applicant_photo="applicant.jpg",
    id_photo="id_card.jpg",
    threshold=0.6
)

if result["verified"]:
    print(f"Identity verified! Method: {result['method']}")
    print(f"Confidence: {result['confidence']:.3f}")
else:
    print(f"Identity mismatch. Confidence: {result['confidence']:.3f}")
```

## Summary

The Image Verification API provides:
- ✅ **Multi-modal fusion** (body + face) for improved accuracy
- ✅ **Automatic fallback** to body-only when face unavailable
- ✅ **Customizable parameters** (distance metric, threshold, fusion weights)
- ✅ **Detailed results** with distances, scores, and modality information
- ✅ **Graceful error handling** with warnings and flexible detection
- ✅ **Backward compatible** with existing body-only verification

For more information, see:
- [API Specification](./contracts/verify-api.yaml)
- [Data Model](./data-model.md)
- [Implementation Plan](./plan.md)
