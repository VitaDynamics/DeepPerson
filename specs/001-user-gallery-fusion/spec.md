# Feature Specification: User Gallery Fusion Retrieval

**Feature Branch**: `001-user-gallery-fusion`  
**Created**: 2025-11-04  
**Status**: Draft  
**Input**: User description: "Current System only calculate two image or single image. We should be at the gallery level to introduce the concept of a User. User can bind with Multiple Images and Face also.  \n\nSo for other APIs, I think represent API should support optional use deepface to generate a face embedding https://github.com/serengil/deepface also. This facilitates later dual-way recall retrieval.  \n\nAnd also we should think about An algorithm that converts a retrieval request into comparisons with all embeddings under the username. This requires a fusion algorithm. For example, a user may have different outfits every day, so the facial embeddings should be similar; because of different outfits, they will be bound to multiple images, with each outfit having one image or multiple images. There may also be different angles. Therefore a sample (User) may correspond to multiple entities, each entity with multiple images. There is an algorithm that can retrieve a user from a single image."

## Clarifications

### Session 2025-11-04

- Q: How should the system assign new images to "Person Entity Variant" groupings? → A: System automatically clusters images without operator review.
- Q: What default fusion weighting should combine face and body embeddings? → A: Confidence-weighted fusion that scales each modality per probe.

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Register user gallery with body & face media (Priority: P1)

Security operators create a user profile that links multiple full-body images and at least one face image, ensuring the system recognises a person across outfits and angles.

**Why this priority**: Without persistent user galleries, the system cannot aggregate embeddings or enable multi-image retrieval, blocking downstream functionality.

**Independent Test**: Import a new user with three body images and one face image, confirm the gallery stores metadata and embeddings without touching other features.

**Acceptance Scenarios**:

1. **Given** the operator provides a new user identifier and body images, **When** the gallery registration is submitted, **Then** the system stores each image with associated body embeddings and timestamps.
2. **Given** the operator uploads a face portrait for an existing user, **When** optional face embedding generation is requested, **Then** the system attaches the resulting face embedding set to the user profile.

---

### User Story 2 - Generate representations via API (Priority: P2)

Application developers call the `represent` API with user galleries to refresh embeddings, optionally enabling face embedding generation through an external provider.

**Why this priority**: Controllable representation generation is needed to keep galleries current and to supply both body and face embeddings for retrieval quality.

**Independent Test**: Invoke the API for a user gallery with the face-embedding flag enabled and verify the response returns new embeddings and status without using search flows.

**Acceptance Scenarios**:

1. **Given** a gallery with new body images, **When** the `represent` API is triggered, **Then** the response includes fresh body embeddings for each image and audit metadata for the run.
2. **Given** a gallery with eligible face images and the face flag enabled, **When** the API runs, **Then** the response includes face embeddings generated via the configured face embedding provider and marks images without detectable faces.

---

### User Story 3 - Retrieve user from single probe image (Priority: P3)

Investigators submit one probe image (body or face) and receive top candidate users ranked by a fusion score combining all embeddings in each gallery.

**Why this priority**: Retrieval accuracy on real-world probes validates the end-to-end value of the new gallery model and fusion algorithm.

**Independent Test**: Execute a retrieval query with a probe image not seen during registration and verify the ranked results aggregate per user with explanatory scores.

**Acceptance Scenarios**:

1. **Given** a probe body image, **When** retrieval runs, **Then** the system compares the probe embedding to every gallery image embedding for each user and returns user-level scores with top matches.
2. **Given** a probe face image, **When** retrieval runs, **Then** the system leverages stored face embeddings where available, falls back to body-only scoring otherwise, and reports how the fusion score was computed.

---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

- What happens when a user gallery has body images but no valid face image? The system must skip face embedding generation gracefully and flag the gap for follow-up.
- How does system handle multiple users with highly similar face embeddings (e.g., twins)? The fusion logic must retain per-user evidence and allow thresholds/alerts to review ambiguous matches.
- What if a new outfit image lacks a detectable person or is low quality? The ingestion flow must reject or quarantine the asset with actionable feedback without corrupting existing embeddings.
- How does the system behave when a user gallery exceeds configured image limits? Registration should enforce caps and prompt curation before accepting more images.
- What happens when a retrieval probe has mismatched modalities (e.g., face probe against body-only galleries)? Fusion must degrade gracefully by weighting available modalities and communicating confidence.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST allow operators or automated feeds to create and update user galleries that aggregate unique user identifiers, metadata, and related image assets.
- **FR-002**: System MUST ingest and store multiple body images per user, preserving source metadata (capture time, camera, outfit descriptors) for retrieval context.
- **FR-003**: System MUST support attaching one or more face images per user and associate each with generated face embeddings when requested.
- **FR-004**: `represent` API MUST generate body embeddings for every new or updated body image and return structured results grouped by image and modality.
- **FR-005**: `represent` API MUST, when invoked with the face-embedding option, call the configured face embedding provider (DeepFace or equivalent) and persist resulting embeddings with traceability to the provider and version.
- **FR-006**: Retrieval service MUST compute probe embeddings, compare them against all stored embeddings for each user, and aggregate scores per user via a documented fusion strategy that supports modality weighting.
- **FR-007**: System MUST expose configuration to adjust fusion weights and thresholds per deployment while providing a sensible default profile.
- **FR-008a**: Default fusion MUST use confidence-weighted blending so stronger modality signals dominate per probe while considering both modalities when available.
- **FR-009**: Retrieval responses MUST return top-N user candidates with aggregated confidence scores, contributing evidence (image references, modality scores), and rationale for low-confidence or tie cases.
- **FR-010**: System MUST maintain audit logs for gallery updates, embedding regenerations, and retrieval decisions to support investigations and model monitoring.
- **FR-011**: System MUST automatically cluster incoming user images into appearance variants without requiring operator confirmation while exposing variant assignments for downstream use.

### Key Entities *(include if feature involves data)*

- **User Gallery**: Represents a unique person within the system; contains user identifier, demographic/operational metadata, status flags, and relationships to person entities and embeddings.
- **Person Entity Variant**: Captures a specific appearance cluster for a user (e.g., outfit, time period) and links to the image assets and embeddings representing that visual state.
- **Image Asset**: Stores media inputs (body or face), source descriptors, modality tags, and processing status for embedding generation.
- **Embedding Set**: Encapsulates body and face embedding vectors, provider metadata (model version, modality), quality scores, and timestamps for regeneration.
- **Retrieval Probe**: Represents a query instance containing image input, derived embeddings, requested modalities, and evaluation metadata used during comparison.

### Assumptionsn after regeneration, subject to retention policies managed elsewhere.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: 95% of user gallery registrations complete with valid body embeddings within 2 minutes of asset upload.
- **SC-002**: Optional face embedding generation adds no more than 1.5 seconds median latency per processed face image when run with the default provider.
- **SC-003**: Retrieval on validation probes (mix of face and body) returns the correct user in the top-3 results at least 90% of the time.
- **SC-004**: At least 80% of retrieval responses include modality-level evidence (body and/or face) enabling investigators to justify match decisions without manual log inspection.


- Deployments will provide or approve access to a face embedding provider compatible with DeepFace APIs for optional face processing.
- Galleries cap the number of stored images per user (default 20 body, 5 face) unless configuration increases limits.
- Operators can supply minimal metadata (user identifier and capture source) at registration; deeper demographic data is optional.
- Historical embeddings remain available for auditing eve