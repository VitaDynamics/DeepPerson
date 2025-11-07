<!--
Sync Impact Report
==================

Version Change: 1.0.0 → 2.0.0

Modified Principles:
  - II. Pipeline Pattern: Detection → Embedding → Search → Detection → Embedding → Verification/Fusion
  - III. Hardware Optimization: Removed FAISS GPU references
  - V. Production Readiness: Updated performance metrics to remove gallery search metrics

Added Sections:
  - Multi-modal Fusion Scoring emphasis in Principle II

Removed Sections:
  - IV. Gallery System with Multi-Modal Support (PRINCIPLE REMOVED - stateless API focus)
  - Gallery storage format references
  - FAISS dependency requirements

Templates Status:
  ✅ .specify/templates/plan-template.md - Constitution Check references updated (Gallery tag removed)
  ✅ .specify/templates/spec-template.md - No updates needed (constitution compliant)
  ✅ .specify/templates/tasks-template.md - Gallery tag removed from constitution principle tags

Deferred Items:
  - N/A (all placeholders filled)

Version Bump Rationale:
  MAJOR version bump (1.0.0 → 2.0.0) due to removal of Principle IV (Gallery System).
  This is a backward incompatible change as the gallery API has been completely removed,
  shifting the library focus to stateless embedding generation and fusion scoring.
  The library is now purely functional with no persistent state management.

-->

# DeepPerson Constitution

## Core Principles

### I. Registry Pattern for Model Management (NON-NEGOTIABLE)
All model profiles, configurations, and cache management MUST use the Registry pattern with thread-safe access. The `ModelRegistry` class in `src/registry.py` is the single source of truth for model profiles. All model loading, caching, and profile management MUST route through the registry. New models require ModelProfile registration before use. Model profiles MUST include: identifier, backbone_path, feature_dim, preprocessing config, and hardware requirements. Thread safety using `threading.RLock` MUST be enforced for all registry operations. Face embedding models MUST be cached via registry's `load_face_model()` method.

**Rationale**: Centralized model management ensures consistent state across concurrent operations, prevents duplicate model loading, and provides a single point for model profile validation and updates. This applies to both body embedding models (ResNet-50) and face embedding models (DeepFace).

### II. Pipeline Pattern for Embedding Generation and Verification (NON-NEGOTIABLE)
The processing pipeline MUST follow the pattern: Person Detection (detectors.py) → Feature Extraction (embeddings.py, face_embeddings.py) → Distance Computation (distance.py) → Fusion Scoring (fusion.py for multi-modal). Each stage MUST be independently testable and composable. The `DeepPerson` façade class orchestrates this pipeline but each component MUST expose clean interfaces for unit testing. The pipeline MUST handle: YOLO person detection with configurable thresholds, ResNet-50 Circle DG body embedding generation (512-dim), optional DeepFace face embedding generation (128-512-dim), distance metric computation (cosine, euclidean, euclidean_l2), and confidence-weighted fusion scoring for multi-modal verification. Pipeline failures at any stage MUST provide clear error propagation without partial state corruption. The `EmbeddingGenerator` interface MUST be implemented by both body and face generators for polymorphic usage.

**Rationale**: The pipeline pattern enables independent testing of each stage, clear separation of concerns, and flexibility to swap components (e.g., different detectors or embedding models) without affecting the overall workflow. Multi-modal fusion scoring significantly improves verification accuracy by combining body shape and facial features with confidence weighting.

### III. Hardware Optimization with CPU/GPU Fallback (NON-NEGOTIABLE)
All operations MUST support automatic hardware detection with graceful CPU fallback. The system MUST detect CUDA availability and use GPU acceleration when present without requiring manual configuration. Device selection MUST be explicit in all class constructors and method signatures. GPU memory management MUST include batch size controls and cleanup mechanisms. Models MUST checkpoint state appropriately when switching devices. CPU MUST be fully functional as a complete fallback - no feature degradation allowed. Performance considerations MUST be documented: GPU provides ~10-50x speedup for body embedding generation, face embeddings are typically CPU-based (DeepFace), batch processing provides linear speedup up to GPU memory limits.

**Rationale**: Hardware optimization ensures the component works across diverse deployment environments while maximizing performance on capable hardware. Automatic fallback maintains accessibility and reliability across CPU-only servers, local development machines, and GPU-enabled production environments.

### IV. Multi-Modal Fusion Scoring (NON-NEGOTIABLE)
Identity verification MUST support multi-modal fusion scoring that combines body and face embeddings with confidence weighting. The `FusionScorer` class in `src/fusion.py` MUST implement late fusion strategy with dynamic weighting based on embedding quality and detection confidence. Fusion scoring MUST support: configurable default weights (face/body), dynamic confidence-based weighting, minimum confidence thresholds for face embedding usage, and batch processing for multiple candidates. Fusion metadata MUST include: applied weights, individual modality scores, confidence values, and face usage indicator. Distance metrics (cosine, euclidean, euclidean_l2) MUST be converted to similarity scores (0.0-1.0 range) before fusion. Verification decisions MUST use fusion scores when both modalities available, fallback to body-only when face unavailable or low confidence.

**Rationale**: Multi-modal fusion significantly improves verification accuracy by leveraging complementary information from body shape and facial features. Confidence-based weighting ensures robust handling of partial detections and varying image quality. This principle supports the library's core value proposition of accurate person re-identification.

### V. Production Readiness with Comprehensive Observability (NON-NEGOTIABLE)
All operations MUST include structured logging, error handling, and performance metrics. Logging MUST use appropriate levels (DEBUG, INFO, WARNING, ERROR) with contextual information (model profiles, hardware info, timing). Error handling MUST be specific and actionable - no generic exceptions. Performance metrics MUST track: body embedding generation time, face embedding generation time (when used), detection confidence, distance computation time, fusion scoring time, and overall verification latency. Code coverage MUST meet project standards with unit, integration, and contract tests. Type hints REQUIRED for all public interfaces (mypy strict compliance). Thread safety validation REQUIRED for all shared state operations (registry, model cache). Production readiness checklist MUST pass before deployment: error handling verified, logging configured, tests passing, type checking clean, performance benchmarks documented.

**Rationale**: Production systems require comprehensive observability for debugging, monitoring, and incident response. Structured logging and metrics enable operational visibility, while type hints and tests ensure code quality and reliability. Performance tracking is critical for SLA compliance and optimization decisions.

## Technology & Quality Standards

All code MUST follow Python 3.11+ standards with strict type checking. Dependencies MUST use `pyproject.toml` with proper core and optional extras. Core dependencies: torch>=2.0.0, torchvision>=0.15.0, numpy>=1.24.0, pillow>=9.0.0, scikit-learn>=1.2.0, gdown>=5.0.0, ultralytics>=8.3.224, tf-keras>=2.18.0. Optional dependencies: deepface (required for face embeddings), pytest>=7.0.0, pytest-cov>=4.0.0, ruff>=0.8.0, mypy>=1.0.0. Code formatting and linting MUST use `ruff check --fix` and `ruff format` with 88-character line length. All public classes and methods MUST have type hints and documentation. Import organization MUST follow stdlib → third-party → local with Ruff auto-formatting.

Testing requirements: Unit tests for individual components (detectors, embeddings, fusion, distance), integration tests for full pipeline (represent, verify), contract tests for public API boundaries. All tests MUST use pytest with appropriate markers (@pytest.mark.unit, @pytest.mark.integration, @pytest.mark.slow). Test coverage targets: 80% minimum for core components, 90% for critical paths (fusion scoring, distance metrics, verification pipeline).

Performance standards: Body embedding generation should complete within 100ms per person on GPU (batch_size=1), face embedding generation within 200-500ms per person on CPU (DeepFace), verification (body+face) within 1 second for two images. Memory usage MUST be monitored and cleanup provided for long-running operations. Batch processing MUST provide linear speedup up to GPU memory limits.

Model management standards: All model weights MUST be automatically downloaded on first use via `ModelManager`, cached locally for subsequent runs, and verified with checksum validation. Body embedding models MUST support hot-swapping without service restart. Face embedding models MUST be cached by the registry per (model_name, detector_backend) combination. Backward compatibility MUST be maintained for model profile changes with migration utilities.

## Development Workflow

All changes MUST follow the feature branch workflow with corresponding PRs. Feature branches MUST use the format `###-feature-name` (e.g., `001-fusion-scoring`). PRs MUST include: clear description of changes, link to related specifications, test coverage verification, and type checking results. Code review MUST verify: compliance with all constitution principles, test coverage adequacy, performance impact assessment, and backward compatibility considerations.

Pre-commit validation REQUIRED: `ruff check --fix` and `ruff format` for code quality, `mypy src/` for type safety, `pytest tests/` for test verification. PRs MUST NOT merge without passing all checks. Complexity additions MUST be justified - simpler alternatives considered and documented if rejected. Breaking changes MUST include migration strategy and deprecation timeline.

Dependency management: New dependencies MUST be evaluated for necessity, security, and maintenance burden. Optional dependencies MUST be documented with use cases. Version pinning MUST use conservative ranges (e.g., `>=2.0.0` for major frameworks). Security updates MUST be tracked and applied within reasonable timeframes. Removal of dependencies (e.g., FAISS) MUST be accompanied by deprecation notices and migration documentation.

Documentation requirements: All public APIs MUST have docstrings with examples, README updates for user-facing changes, API reference documentation for new endpoints. Architecture decisions MUST be documented in relevant files (CLAUDE.md, constitution.md). Migration guides REQUIRED for breaking changes. Examples MUST demonstrate both single-modal (body-only) and multi-modal (body+face) usage patterns.

## Governance

**Constitution Supremacy**: This constitution supersedes all other development practices and guidelines. In conflicts between this constitution and other documentation, this constitution takes precedence.

**Amendment Process**: Amendments require:
1. Draft proposal with rationale, impact analysis, and backward compatibility assessment
2. Review period with stakeholder feedback collection
3. Version bump following semantic versioning rules:
   - MAJOR: Backward incompatible changes to principles or architecture (e.g., principle removal)
   - MINOR: New principles, sections, or materially expanded guidance
   - PATCH: Clarifications, wording improvements, non-semantic refinements
4. Migration strategy documentation if breaking changes
5. Approval from project maintainers

**Compliance Verification**: All PRs MUST verify constitution compliance through:
- Self-check against each applicable principle
- Automated tooling where possible (type checking, test coverage, linting)
- Manual review for complex architectural decisions
- Performance benchmarking for hardware optimization changes
- Fusion scoring validation for multi-modal changes

**Version Control**: Constitution changes MUST be tracked in git with clear commit messages indicating version bumps and rationale. The `LAST_AMENDED_DATE` MUST be updated for all changes (including PATCH level). Historical versions MUST be preserved for audit trail.

**Exception Process**: Exceptions to principles require documented justification with:
- Specific business or technical reason for deviation
- Alternative approaches considered and rejected
- Mitigation strategy for risks introduced
- Sunset date for temporary exceptions
- Approval from project maintainers

**Guidance Integration**: Development guidance in `CLAUDE.md` MUST align with constitution principles. When principles are updated, all guidance documents MUST be reviewed and updated accordingly. Template files in `.specify/templates/` MUST reference current principles only (Registry, Pipeline, Hardware, Fusion, Observability).

**Version**: 2.0.0 | **Ratified**: 2025-11-04 | **Last Amended**: 2025-11-07
