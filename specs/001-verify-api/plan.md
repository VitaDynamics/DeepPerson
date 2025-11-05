# Implementation Plan: Image Verification API Enhancement

**Branch**: `001-verify-api` | **Date**: 2025-11-05 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-verify-api/spec.md`

**Note**: The verify API already exists in `/home/heng.li/repo/DeepPerson/src/api.py` (lines 506-689). This plan focuses on minimal enhancements to support multi-modal fusion scoring while reusing existing components.

## Summary

**Primary Requirement**: Enhance existing verify API to support fusion scoring (body+face embeddings) with body-only fallback, using existing core APIs (represent, compute_distance, model registry).

**Technical Approach**:
- Leverage existing `represent()` API with `generate_face_embeddings=True`
- Check for face embeddings in results
- **Reuse existing FusionScorer** from `/src/user_gallery/fusion.py` for fusion scoring
- Fallback to body-only comparison when face data unavailable
- Use existing model registry for threshold management

## Technical Context

**Language/Version**: Python 3.12+ (existing codebase standard)
**Primary Dependencies**: torch, torchvision, numpy, scikit-learn (existing)
**Storage**: N/A (stateless API)
**Testing**: pytest (existing framework)
**Target Platform**: Linux server with CUDA GPU support (existing)
**Project Type**: Python library (existing deep-person component)
**Performance Goals**: <2s verification time (from spec SC-001)
**Constraints**: Use existing APIs; minimal code additions; backward compatible
**Scale/Scope**: Two image comparison per call

## Constitution Check

*No constitution constraints found - using existing codebase patterns*

## Project Structure

### Documentation (this feature)

```text
specs/001-verify-api/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/
├── api.py              # MODIFY: Enhance existing verify() method
├── search.py           # REUSE: compute_distance() function
├── utils.py            # REUSE: Existing utilities
├── registry.py         # REUSE: Model registry for thresholds
├── embeddings.py       # REUSE: BodyEmbeddingGenerator
├── detectors.py        # REUSE: PersonDetector implementations
└── entities.py         # REUSE: PersonEmbedding data structures

tests/
├── unit/
│   └── test_verify.py  # NEW: Unit tests for verify API
└── integration/
    └── test_verify_api.py  # NEW: Integration tests
```

**Structure Decision**: Reuse existing single-library structure; modify api.py to enhance verify method

## Complexity Tracking

**Current Complexity**: Low - verify API exists with basic body-only verification
**Enhancement Complexity**: Minimal - Add fusion scoring support using existing FusionScorer component

| Change | Why Needed | Simpler Alternative Rejected Because |
|--------|------------|--------------------------------------|
| Fusion scoring support | Spec requires body+face verification | Body-only insufficient per clarification |
| Reuse FusionScorer | Avoid code duplication, leverage existing implementation | New fusion logic would be redundant |
| Multi-modal embedding generation | Needed for fusion scoring | Single modality doesn't meet requirements |
| Fallback logic | Handle missing face data gracefully | Error on missing face too restrictive |

## Architecture Decision: FusionScorer Location

**Option A**: Use existing FusionScorer from user_gallery/fusion.py
- ✅ Simpler: Just import and use
- ❌ Creates dependency from core API to user_gallery module
- ❌ Semantically odd: verify API shouldn't depend on gallery functionality

**Option B**: Extract FusionScorer to common location (e.g., src/fusion.py)
- ✅ Proper separation of concerns
- ✅ Available for any component needing fusion
- ✅ Clean dependency structure
- ❌ Requires refactoring user_gallery/fusion.py

**Decision**: Option B - Extract to common location
**Rationale**: Better long-term architecture. Fusion scoring is a general utility, not gallery-specific.

**Implementation**:
1. Create `src/fusion.py` with FusionScorer class
2. Update `src/user_gallery/fusion.py` to import from `src.fusion`
3. Update verify() to import from `src.fusion`

## Phase 0: Research & Planning

### Research Required

1. **Fusion Scoring Implementation**
   - Research best practices for combining body and face embedding distances
   - Determine optimal weighting strategies (50/50 default from spec)
   - Document in research.md

2. **Existing Component Analysis**
   - Map existing verify API workflow (lines 506-689 in api.py)
   - Identify integration points for face embeddings
   - Document dependency graph in research.md

### Research Tasks

- [ ] Document current verify() implementation details
- [ ] Research fusion scoring algorithms for multi-modal verification
- [ ] Identify required changes to support face embeddings
- [ ] Define minimal modification strategy

## Phase 1: Design & Contracts

### Design Tasks

1. **Data Model Updates**
   - Document updated verification result structure (face scores, fusion weights)
   - Create data-model.md with entity changes

2. **API Contract Enhancements**
   - Document enhanced verify() method signature (already exists)
   - Add fusion scoring parameters to contracts/
   - Create quickstart.md with usage examples

3. **Agent Context Update**
   - Update claude-specific context file with verification logic patterns

### Deliverables

- data-model.md: Enhanced result structure
- /contracts/verify-api.yaml: OpenAPI specification
- quickstart.md: Usage guide
- Agent-specific context updates

## Phase 2: Implementation Tasks

*Generated by `/speckit.tasks` command - NOT created by `/speckit.plan`*

### Expected Tasks

1. **Extract FusionScorer to Common Location**
   - Create `src/fusion.py` with FusionScorer class
   - Copy FusionScorer implementation from user_gallery/fusion.py
   - Update user_gallery/fusion.py to import from src.fusion
   - Ensure both modules use the same FusionScorer

2. **Enhance verify() Method** (src/api.py:506-689)
   - Add face embedding generation using existing represent() API
   - **Reuse FusionScorer from src.fusion** for fusion scoring
   - Add fallback to body-only comparison
   - Update response structure to include fusion metadata

2. **Add Unit Tests** (tests/unit/test_verify.py)
   - Test fusion scoring with both modalities
   - Test fallback behavior with missing face data
   - Test threshold comparison logic
   - Test edge cases

3. **Add Integration Tests** (tests/integration/test_verify_api.py)
   - End-to-end verification workflow
   - Multiple distance metrics
   - Custom threshold handling

### Minimal Modification Strategy

**Core Principle**: Reuse existing APIs and components

**Modifications**:
1. Modify verify() to call represent() with `generate_face_embeddings=True`
2. **Reuse FusionScorer** from `/src/user_gallery/fusion.py` for fusion scoring
3. Update response to include face scores and fusion weights

**Reused Components**:
- `represent()` - Embedding generation with face embeddings
- **`FusionScorer`** - Fusion scoring logic (already in user_gallery/fusion.py)
- `compute_distance()` - Distance calculations
- Model Registry - Threshold lookup
- PersonDetector - Person detection
- BodyEmbeddingGenerator - Feature extraction

**Key Benefit**: FusionScorer already handles all fusion logic including:
- Body-only fallback when face unavailable
- Custom weight support
- Confidence-based weighting
- Detailed fusion metadata

## Risk Assessment

**Low Risk**: Existing API structure allows backward-compatible enhancements
**Dependencies**: None - all required components already exist
**Performance Impact**: Minimal - single additional API call for face embeddings when needed

## Next Steps

1. Execute Phase 0 research tasks
2. Generate Phase 1 design artifacts
3. Run `/speckit.tasks` to create implementation tasks
4. Implement minimal enhancements to verify() method
5. Add comprehensive tests

## Success Metrics

- [ ] verify() supports fusion scoring (body+face when available)
- [ ] Backward compatible with existing body-only usage
- [ ] <2s verification time (SC-001)
- [ ] 95%+ accuracy with appropriate thresholds (SC-002)
- [ ] All response fields present 100% of time (SC-003)
- [ ] Handles edge cases gracefully (SC-004)
- [ ] Customizable parameters work correctly (SC-005)
- [ ] Deterministic results (SC-006)
