# Research: Image Verification API Fusion Scoring

**Date**: 2025-11-05
**Feature**: Image Verification API Enhancement
**Source**: `/specs/001-verify-api/spec.md`

## Research Summary

This document presents findings from researching fusion scoring for multi-modal (body+face) person verification in the DeepPerson API.

**Key Finding**: The codebase already includes a production-ready `FusionScorer` class in `/src/user_gallery/fusion.py` that implements all needed fusion logic. The recommended approach is to extract this to a common location (`src/fusion.py`) and reuse it in the verify() API enhancement.

## Decision 1: Fusion Scoring Algorithm

**Decision**: Implement weighted fusion scoring for body and face embeddings

**Rationale**:
- Different modalities capture different aspects of identity (body shape vs facial features)
- Combining both improves verification accuracy vs single modality
- Weighted combination allows tunable importance of each modality
- Simple weighted average is effective and interpretable

**Implementation Approach**:
```
Use existing FusionScorer class from /src/user_gallery/fusion.py
```

The `FusionScorer.compute_fusion_score()` method already implements:
- Weighted combination: `(body_score * body_weight) + (face_score * face_weight)`
- Distance-to-similarity conversion
- Custom weight support
- Automatic fallback to body-only when face unavailable
- Confidence-based dynamic weighting

**Advantages of Reusing Existing Code**:
- ✅ Proven implementation in production
- ✅ Handles edge cases (low confidence, missing face)
- ✅ Supports custom fusion weights
- ✅ Returns detailed fusion metadata
- ✅ Tested in gallery retrieval context

**Alternatives Considered** (and rejected):
- Weighted product (rejected: too sensitive to low scores)
- Max pooling (rejected: doesn't combine information)
- Learned fusion (rejected: requires training data, too complex)
- Custom implementation (rejected: unnecessary duplication)

## Decision 2: Distance to Similarity Conversion

**Decision**: Convert distances to similarities for fusion scoring

**Rationale**:
- Distance metrics produce values where lower = more similar
- Fusion scoring more intuitive with higher = better
- Conversion: similarity = 1 - normalized_distance
- Maintain consistency with threshold interpretation

**Implementation**:
```python
if distance_metric == "cosine":
    body_similarity = 1.0 - body_distance
    face_similarity = 1.0 - face_distance
else:
    # For euclidean metrics, normalize by theoretical max
    # Cosine: [0, 2] → Similarity: [1, -1]
    # Euclidean: [0, ∞) → Need normalization
    pass
```

## Decision 3: Fallback Strategy

**Decision**: When face embedding unavailable in either image, use body-only comparison

**Rationale**:
- Graceful degradation better than failure
- Body embeddings always generated (primary subject requirement)
- Maintains backward compatibility with existing body-only verification
- Clear behavior for mixed-modality scenarios

**Implementation Logic**:
```python
if has_face_emb1 and has_face_emb2:
    # Full fusion: body + face
    use_fusion_scoring = True
elif has_face_emb1 or has_face_emb2:
    # One face missing: body-only comparison
    # Log warning about mixed modalities
    use_fusion_scoring = False
else:
    # Both missing: body-only comparison (shouldn't happen)
    use_fusion_scoring = False
```

## Decision 4: Threshold Comparison

**Decision**: Compare fusion score against threshold for similarity-based metrics

**Rationale**:
- Consistent with single-modality verification
- Thresholds from registry work for both distance and similarity
- Document threshold interpretation: score >= threshold → verified

**Implementation**:
```python
if use_fusion_scoring:
    # Compare fusion score to threshold
    verified = fusion_score >= threshold
else:
    # Compare body distance to threshold (existing behavior)
    verified = body_distance <= threshold
```

## Existing Implementation Analysis

### Current verify() Method (src/api.py:506-689)

**Workflow**:
1. Accept two image paths and parameters
2. Call `represent()` for each image (body embeddings only)
3. Extract primary subject embedding from each result
4. Compute distance using `compute_distance()`
5. Compare distance to threshold (from registry or parameter)
6. Return verification result

**Current Response Structure**:
```python
{
    "verified": bool,
    "distance": float,
    "threshold": float,
    "distance_metric": str,
    "model": str,
    "detector_backend": str,
    "facial_areas": {"img1": bbox, "img2": bbox},
    "warnings": [str]  # optional
}
```

### Required Enhancements

**Changes to verify() Method**:
1. Enable face embedding generation in `represent()` calls
2. Extract face embeddings when available
3. Implement fusion scoring logic
4. Update response structure to include:
   - `body_distance`: Body embedding distance
   - `face_distance`: Face embedding distance (if available)
   - `fusion_score`: Combined similarity score
   - `face_weight`: Weight used for face modality
   - `body_weight`: Weight used for body modality
   - `used_fusion`: Boolean indicating if fusion was used

## Integration Points

### Reused Components

1. **represent() API** (src/api.py:251-504)
   - Already supports `generate_face_embeddings=True`
   - Returns face embeddings in subject["face_embedding"]
   - No modification needed

2. **compute_distance()** (src/search.py:833-852)
   - Supports cosine, euclidean, euclidean_l2
   - Already used for body embeddings
   - Can reuse for face embeddings

3. **Model Registry** (src/registry.py)
   - Provides verification thresholds
   - Works with existing threshold lookup
   - No modification needed

4. **PersonEmbedding** (src/entities.py)
   - Contains both body and face embeddings
   - `has_face_embedding` property
   - No modification needed

### Component Reuse

**FusionScorer from user_gallery/fusion.py**:
- ✅ Already implements all needed fusion logic
- ✅ Handles edge cases (low confidence, missing face)
- ✅ Supports custom weights
- ✅ Returns fusion metadata

**Proposed Change**:
- Extract `FusionScorer` class to `src/fusion.py` (common location)
- Update `user_gallery/fusion.py` to import from `src.fusion`
- Both modules use the same FusionScorer implementation

**Rationale**: Fusion scoring is a general utility, not gallery-specific. Moving to common location:
- Enables reuse across the codebase
- Avoids circular dependencies
- Improves architecture and maintainability

### New Components (Minimal)

1. **Common Fusion Module** (src/fusion.py)
   - Extract FusionScorer from user_gallery/fusion.py
   - Make available for any component needing fusion

2. **Enhanced Response Structure** (verify API)
   - Additional fields for fusion scoring
   - Backward compatible additions only

## Performance Considerations

**Time Complexity**:
- Body embedding: O(1) - existing
- Face embedding: O(1) - already generated by represent()
- Fusion scoring: O(1) - simple weighted average
- **Total**: O(1) - no significant overhead

**Memory Impact**:
- Additional storage for face embeddings (already in memory)
- Small overhead for fusion score calculation
- **Total**: Negligible

## Testing Strategy

### Unit Tests

1. **Fusion Scoring Logic**
   - Test with both body and face embeddings
   - Test with missing face in one image
   - Test with missing face in both images
   - Test different weight combinations

2. **Distance/Similarity Conversion**
   - Test cosine distance → similarity conversion
   - Test euclidean distance → similarity conversion
   - Test threshold comparison logic

3. **Edge Cases**
   - No face embeddings generated
   - Face detection failed
   - Multiple detections (existing behavior)

### Integration Tests

1. **End-to-End Verification**
   - Two images, same person with face+body
   - Two images, different people
   - Mixed scenarios (one with face, one without)

2. **Parameter Variations**
   - Different distance metrics
   - Custom thresholds
   - Different fusion weights

## Risk Assessment

**Low Risk Changes**:
- All core components already exist
- Backward compatible (body-only still works)
- Simple algorithm (weighted average)
- Well-understood domain (multi-modal fusion)

**Potential Issues**:
- Face detection may fail on some images (handled by fallback)
- Performance impact of face embedding generation (minimal)
- Threshold calibration for fusion scores (uses existing registry)

## Documentation Requirements

1. **API Documentation**
   - Updated verify() method signature and parameters
   - Fusion scoring explanation
   - Threshold interpretation for fusion scores

2. **Usage Examples**
   - Basic verification (same/different person)
   - Fusion scoring with face embeddings
   - Custom fusion weights
   - Fallback behavior explanation

## Conclusion

**Fusion scoring implementation is feasible with minimal changes**:
- Reuse existing `represent()`, `compute_distance()`, and model registry
- Add simple weighted average logic for combining scores
- Enhance response structure for transparency
- Maintain backward compatibility

**Key Success Factors**:
- Clear fallback to body-only when face unavailable
- Transparent response with all scores and weights
- Backward compatible with existing usage
- Proper threshold interpretation

**Next Steps**:
1. Implement fusion scoring logic in verify() method
2. Update response structure
3. Add comprehensive tests
4. Document usage and parameters
