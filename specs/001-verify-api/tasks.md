---
description: "Task list for Image Verification API Enhancement"
---

# Tasks: Image Verification API Enhancement

**Input**: Design documents from `/specs/001-verify-api/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Existing Code**: verify() API already exists in src/api.py (lines 506-689)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 [P] Review existing verify() implementation in src/api.py lines 506-689
- [x] T002 [P] Verify existing test directory structure (tests/unit/, tests/integration/)
- [x] T003 [P] Check existing pytest configuration and add test_verify.py stubs

**Checkpoint**: Setup complete - ready for foundational work

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Extract and prepare shared fusion scoring component needed for all verification enhancements

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Create src/fusion.py with FusionScorer class
  - Copy FusionScorer implementation from src/user_gallery/fusion.py
  - Include compute_fusion_score() and compute_batch_fusion_scores() methods
  - Include __init__() with default_face_weight, default_body_weight, min_face_confidence parameters
  - Add comprehensive docstrings

- [x] T005 Refactor src/user_gallery/fusion.py to import from src.fusion
  - Import FusionScorer from src.fusion
  - Remove duplicate FusionScorer class definition
  - Keep FusionRetrievalService as is
  - Verify gallery functionality still works

- [x] T006 [P] Create test for FusionScorer in tests/unit/test_fusion_scorer.py
  - Test compute_fusion_score() with both modalities
  - Test fallback behavior (no face embedding)
  - Test custom weight parameters
  - Test confidence-based weighting
  - Ensure all tests pass

**Checkpoint**: Foundation ready - fusion scoring component extracted and tested

---

## Phase 3: User Story 1 - Verify Two Images Show Same Person (Priority: P1) 🎯 MVP

**Goal**: Enhance verify() API to support fusion scoring (body+face) with body-only fallback

**Independent Test**: Call verify(img1, img2) with two images of the same person and get verified=True, then test with different people and get verified=False

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T007 [P] [US1] Unit test for basic verification (same/different person) in tests/unit/test_verify.py
- [ ] T008 [P] [US1] Unit test for fusion scoring with body+face in tests/unit/test_verify.py
- [ ] T009 [P] [US1] Unit test for body-only fallback in tests/unit/test_verify.py
- [ ] T010 [P] [US1] Integration test for end-to-end verification workflow in tests/integration/test_verify_api.py

### Implementation for User Story 1

- [ ] T011 [US1] Enhance verify() method in src/api.py (lines 506-689)
  - Add call to represent() with generate_face_embeddings=True parameter
  - Extract face_embedding from results[0]["subjects"][0] when available
  - Import and use FusionScorer from src.fusion
  - Convert body_distance and face_distance to similarities (1 - distance)
  - Use FusionScorer.compute_fusion_score() for fusion scoring
  - Handle body-only fallback when face unavailable
  - Update response structure per data-model.md:
    - Add body_distance, face_distance, fusion_score, face_weight, body_weight
    - Add used_fusion, modality_available fields
    - Maintain backward compatibility with existing fields

- [ ] T012 [US1] Add parameter validation in verify() method
  - Validate fusion_weights if provided (body + face must equal 1.0)
  - Add proper error messages for invalid parameters

- [ ] T013 [US1] Add logging for verification operations
  - Log when fusion scoring is used vs body-only
  - Log fusion scores and weights
  - Log detection counts and warnings

**Checkpoint**: At this point, User Story 1 should be fully functional - verify() supports fusion scoring with body-only fallback

---

## Phase 4: User Story 2 - Customize Verification Parameters (Priority: P2)

**Goal**: Support customization of distance metrics, thresholds, and fusion weights

**Independent Test**: Call verify with different parameters and observe changes in verification results

### Tests for User Story 2

- [ ] T014 [P] [US2] Unit test for different distance metrics in tests/unit/test_verify.py
  - Test cosine (default)
  - Test euclidean
  - Test euclidean_l2
  - Verify correct metric is used and distance values are computed correctly

- [ ] T015 [P] [US2] Unit test for custom thresholds in tests/unit/test_verify.py
  - Test with threshold lower than default (stricter)
  - Test with threshold higher than default (more lenient)
  - Verify threshold from registry is used when not provided

- [ ] T016 [P] [US2] Unit test for fusion weight customization in tests/unit/test_verify.py
  - Test with body_emphasized weights (e.g., 0.8 body, 0.2 face)
  - Test with face_emphasized weights (e.g., 0.3 body, 0.7 face)
  - Verify fusion scoring uses custom weights

### Implementation for User Story 2

- [ ] T017 [US2] Add fusion_weights parameter to verify() method
  - Accept dict with 'body' and 'face' keys
  - Validate weights sum to 1.0
  - Pass custom weights to FusionScorer.compute_fusion_score()

- [ ] T018 [US2] Verify existing distance metric support
  - Confirm cosine, euclidean, euclidean_l2 all work
  - Ensure compute_distance() is called with correct metric
  - Update response to include distance_metric value

- [ ] T019 [US2] Verify threshold handling
  - Use custom threshold if provided
  - Fall back to registry threshold if not provided
  - Pass threshold to FusionScorer and threshold comparison logic

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently with full parameter customization

---

## Phase 5: User Story 3 - Retrieve Verification Details (Priority: P3)

**Goal**: Return comprehensive verification information including all distances, scores, and metadata

**Independent Test**: Verify images and validate that all response fields contain expected information

### Tests for User Story 3

- [ ] T020 [P] [US3] Unit test for response structure completeness in tests/unit/test_verify.py
  - Verify all required fields present (verified, distance, threshold, etc.)
  - Verify fusion-specific fields when used (body_distance, face_distance, fusion_score, etc.)
  - Verify modality_available reflects actual availability

- [ ] T021 [P] [US3] Unit test for bounding box reporting in tests/unit/test_verify.py
  - Verify facial_areas.img1 and facial_areas.img2 are included
  - Verify bounding boxes have correct format (x1, y1, x2, y2)

- [ ] T022 [P] [US3] Unit test for warnings in tests/unit/test_verify.py
  - Test warning when multiple persons detected
  - Test warning when no face embedding available
  - Test warning when face detection confidence low

### Implementation for User Story 3

- [ ] T023 [US3] Enhance verify() response to include all fusion metadata
  - Ensure body_distance (alias of distance), face_distance included
  - Include fusion_score when used
  - Include face_weight, body_weight (both default and custom)
  - Include used_fusion boolean
  - Include modality_available dict with body/face availability

- [ ] T024 [US3] Ensure bounding boxes are always included
  - Extract from results["subjects"][0]["metadata"]["bbox"]
  - Format as {"img1": bbox, "img2": bbox}
  - Handle None case gracefully

- [ ] T025 [US3] Implement comprehensive warning system
  - Add warning when multiple persons detected (use primary)
  - Add warning when face embedding unavailable
  - Add warning when face detection confidence below threshold
  - Return warnings as list in response

**Checkpoint**: All user stories should now be independently functional with comprehensive verification details

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements and validation

- [ ] T026 [P] Add performance benchmark in tests/integration/test_verify_performance.py
  - Verify <2s verification time (SC-001)
  - Test with and without fusion scoring
  - Test CPU vs GPU execution

- [ ] T027 [P] Add accuracy validation test in tests/integration/test_verify_accuracy.py
  - Test with known same-person pairs (should verify=True)
  - Test with known different-person pairs (should verify=False)
  - Aim for 95%+ accuracy (SC-002)

- [ ] T028 [P] Update quickstart.md examples to reflect fusion scoring
  - Add examples with fusion scoring
  - Add examples with body-only fallback
  - Add examples with custom parameters

- [ ] T029 Run all tests to ensure backward compatibility
  - Ensure existing body-only verification still works
  - Verify all response fields present 100% of time (SC-003)
  - Check deterministic results (SC-006)

- [ ] T030 [P] Code review and cleanup
  - Ensure consistent code style
  - Add type hints where missing
  - Update docstrings for verify() method

**Checkpoint**: All enhancements complete and validated

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Phase 6)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Integrates with US1 but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Integrates with US1/US2 but independently testable

### Within Each User Story

- Tests (if included) MUST be written and FAIL before implementation
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel
- All tests for a user story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test for basic verification (same/different person) in tests/unit/test_verify.py"
Task: "Unit test for fusion scoring with body+face in tests/unit/test_verify.py"
Task: "Unit test for body-only fallback in tests/unit/test_verify.py"
Task: "Integration test for end-to-end verification workflow in tests/integration/test_verify_api.py"

# Launch implementation:
Task: "Enhance verify() method in src/api.py (lines 506-689) to support fusion scoring"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Success Criteria Validation

- [ ] verify() supports fusion scoring (body+face when available)
- [ ] Backward compatible with existing body-only usage
- [ ] <2s verification time (SC-001)
- [ ] 95%+ accuracy with appropriate thresholds (SC-002)
- [ ] All response fields present 100% of time (SC-003)
- [ ] Handles edge cases gracefully (SC-004)
- [ ] Customizable parameters work correctly (SC-005)
- [ ] Deterministic results (SC-006)

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
