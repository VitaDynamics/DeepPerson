# Feature Specification: Image Verification API

**Feature Branch**: `001-verify-api`
**Created**: 2025-11-05
**Status**: Draft
**Input**: User description: "I want to add a new API in @src/api.py , named verify. This API just simply verify if two image are same subject. It should use the represent API to create two represent. Just use primary subject. And use @src/utils.py and @src/search.py  or @src/utils.py to compare them. YOu may check @src/user_gallery/models.py ?"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Verify Two Images Show Same Person (Priority: P1)

**As a** user of the DeepPerson API,
**I want to** verify whether two images contain the same person,
**So that I can** confirm identity or detect duplicate subjects in a dataset.

**Why this priority**: This is the core use case for image verification - the most fundamental identity checking operation that enables applications to verify if two photo IDs, profile pictures, or surveillance images show the same individual.

**Independent Test**: Can be fully tested by calling `verify(img1, img2)` with two images of the same person and getting `verified=True`, then testing with different people and getting `verified=False`.

**Acceptance Scenarios**:

1. **Given** two images of the same person taken from different angles,
   **When** I call the verify API,
   **Then** I should receive `verified=True` with distance below the threshold and bounding boxes showing detected regions.

2. **Given** two images of different people,
   **When** I call the verify API,
   **Then** I should receive `verified=False` with distance above the threshold.

3. **Given** two images with the same person wearing different clothes,
   **When** I call the verify API,
   **Then** I should still receive `verified=True` since the API uses fusion scoring (body+face when available, body-only fallback).

4. **Given** an image with multiple people detected,
   **When** I call the verify API,
   **Then** the API should use the primary (highest confidence) detection from each image with a warning message.

5. **Given** an image with no person detected,
   **When** I call the verify API with `enforce_detection=True`,
   **Then** I should receive a ValueError explaining no person was found.

6. **Given** an image with no person detected,
   **When** I call the verify API with `enforce_detection=False`,
   **Then** I should receive a result with `verified=False`, infinite distance, and a warning.

---

### User Story 2 - Customize Verification Parameters (Priority: P2)

**As a** application developer,
**I want to** customize verification parameters (distance metric, threshold, normalization),
**So that I can** fine-tune verification sensitivity for my specific use case and data quality.

**Why this priority**: Different applications have different requirements for verification strictness. Some security-critical applications need conservative thresholds, while others prioritize user experience with more lenient settings.

**Independent Test**: Can be fully tested by calling verify with different parameters and observing changes in verification results and returned metadata.

**Acceptance Scenarios**:

1. **Given** two images to verify,
   **When** I specify `distance_metric="euclidean"`,
   **Then** the API should use Euclidean distance instead of the default cosine distance.

2. **Given** two images to verify,
   **When** I provide a custom `threshold=0.5`,
   **Then** the API should use my threshold instead of the model default from the registry.

3. **Given** two images to verify,
   **When** I specify `normalization="circle"`,
   **Then** the API should apply Circle normalization to embeddings before comparison.

4. **Given** I specify all verification parameters,
   **When** the API returns the result,
   **Then** all parameter values should be reflected in the response metadata.

---

### User Story 3 - Retrieve Verification Details (Priority: P3)

**As a** developer or end user,
**I want to** receive detailed verification information beyond just a boolean result,
**So that I can** understand why two images were verified as same/different and debug issues.

**Why this priority**: Detailed feedback helps developers tune parameters, debug false positives/negatives, and understand verification confidence levels for better user experience.

**Independent Test**: Can be fully tested by verifying images and validating that all response fields contain expected information (distance, threshold, model, detector, bounding boxes, warnings).

**Acceptance Scenarios**:

1. **Given** a verification request,
   **When** I receive the result,
   **Then** I should see the computed distance, used threshold, and distance metric in the response.

2. **Given** a verification request,
   **When** I receive the result,
   **Then** I should see bounding boxes for detected faces/bodies in both images.

3. **Given** a verification request with multiple detections or warnings,
   **When** I receive the result,
   **Then** I should see a warnings list describing any non-ideal conditions.

4. **Given** a verification request,
   **When** I receive the result,
   **Then** I should see model and detector backend information for audit purposes.

---

### Edge Cases

- What happens when image files don't exist or are corrupted?
- How does the system handle very low quality or blurry images?
- What occurs when both images have multiple people detected?
- How are extreme lighting conditions or occlusions handled?
- What happens with extremely high or low confidence detections?
- How does the system handle different image formats and sizes?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept two image paths/URLs and generate embeddings for each using the represent API
- **FR-002**: System MUST extract the primary subject (highest confidence detection) from each image
- **FR-003**: System MUST compute fusion score using body and face embeddings when both available, fallback to body-only when face missing
- **FR-004**: System MUST compare fusion score or distance against threshold to determine verification result (same person if score >= threshold for cosine similarity, or distance <= threshold)
- **FR-005**: System MUST return comprehensive result including: verified boolean, distance value, threshold, metric, model info, detector backend, and facial areas
- **FR-006**: System MUST support multiple distance metrics: cosine, euclidean, and euclidean_l2
- **FR-007**: System MUST support custom threshold values or use model default from registry
- **FR-008**: System MUST support multiple normalization methods: base, resnet, and circle
- **FR-009**: System MUST handle missing person detection gracefully with enforce_detection parameter
- **FR-010**: System MUST provide warnings for non-ideal conditions (multiple detections, no detections)
- **FR-011**: System MUST validate image file existence before processing
- **FR-012**: System MUST validate all input parameters and raise appropriate errors for invalid values

### Key Entities

- **Image**: Input image file containing one or more people, with path, format, and quality metadata
- **Person Embedding**: Numerical representation of detected person features, with dimension, normalization, and confidence
- **Verification Result**: Outcome of comparing two embeddings, including verified boolean, distance, threshold, and metadata
- **Distance Metric**: Method for comparing embeddings (cosine, euclidean, euclidean_l2) with different characteristics
- **Threshold**: Boundary value used to determine if distance indicates same person (lower distance = more similar)
- **Bounding Box**: Rectangular coordinates marking detected person location in image for visualization

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can complete verification of two images and receive definitive same/different person result in under 2 seconds
- **SC-002**: System achieves 95%+ accuracy on verification tasks when using appropriate threshold values
- **SC-003**: API response includes all required fields (verified, distance, threshold, metric, model, detector, bounding boxes) 100% of the time
- **SC-004**: System handles edge cases (no detection, multiple detections) with appropriate errors or warnings without crashes
- **SC-005**: Developers can customize all verification parameters (metric, threshold, normalization, detector) to fine-tune for their use case
- **SC-006**: Verification maintains consistent results across multiple runs with same input images (deterministic behavior)

## Assumptions

- Input images contain detectable human subjects (person detection backend will identify them)
- Primary subject selection (highest confidence detection) is appropriate for verification use cases
- System uses fusion scoring with body and face embeddings when both available
- If face embedding unavailable in either image, system falls back to body-only comparison
- At least one subject with body embedding is present for verification
- Model registry contains appropriate threshold values for different model and metric combinations
- GPU acceleration is available when supported, with automatic CPU fallback
- Distance threshold interpretation: lower distance means more similar for euclidean metrics, higher score means more similar for cosine similarity

## Dependencies

- **represent API**: Generates embeddings from images (uses detector + embedding pipeline)
- **compute_distance**: Compares two embeddings using selected distance metric
- **Model Registry**: Provides default verification thresholds by model and metric
- **Person Detector**: YOLO or other backend to detect persons in images
- **Embedding Generator**: ResNet50-Circle-DG or other backbone to create feature embeddings

## Recommended Thresholds

For optimal verification accuracy, the following threshold values are recommended for different model and metric combinations. These should be stored in the model registry and can be overridden by users:

- **Cosine Distance**:
  - Conservative (low false positives): 0.3-0.4
  - Balanced: 0.4-0.5
  - Lenient (low false negatives): 0.5-0.6

- **Euclidean Distance**:
  - Conservative: 0.8-1.0
  - Balanced: 1.0-1.2
  - Lenient: 1.2-1.4

- **Euclidean L2 Distance**:
  - Conservative: 0.4-0.5
  - Balanced: 0.5-0.6
  - Lenient: 0.6-0.7

*Note: Actual threshold values may vary based on dataset characteristics and quality requirements. The model registry should provide calibrated defaults for each model/metric combination.*

## Clarifications

### Session 2025-11-05

- Q: Embedding modality selection for verification → A: Use both body and face embeddings with fusion scoring; if face embedding unavailable, fallback to body embedding only; assume at least one subject with body embedding present
- Q: Mixed modality handling (one image has face, other has only body) → A: Use fusion scoring when possible; fallback to body-only comparison when face data missing in either image
- Q: Recommended threshold values for distance metrics → A: Use model-specific defaults from registry; provide guidance table for common model/metric combinations
