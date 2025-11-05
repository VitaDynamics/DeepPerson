<!--
Sync Impact Report
==================

Version Change: Initial Creation → 1.0.0

Modified Principles:
  - N/A (initial creation)

Added Sections:
  - I. Registry Pattern for Model Management (NON-NEGOTIABLE)
  - II. Pipeline Pattern for Detection → Embedding → Search (NON-NEGOTIABLE)
  - III. Hardware Optimization with CPU/GPU Fallback (NON-NEGOTIABLE)
  - IV. Gallery System with Multi-Modal Support (NON-NEGOTIABLE)
  - V. Production Readiness with Comprehensive Observability (NON-NEGOTIABLE)
  - Technology & Quality Standards (new section)
  - Development Workflow (new section)
  - Governance (expanded from template with detailed procedures)

Removed Sections:
  - N/A (initial creation)

Templates Status:
  ✅ .specify/templates/plan-template.md - Constitution Check section already present
  ✅ .specify/templates/spec-template.md - No updates needed (constitution compliant)
  ✅ .specify/templates/tasks-template.md - Already references constitution principles (Registry, Pipeline, Hardware, Gallery, Observability)

Deferred Items:
  - N/A (all placeholders filled)

Version Bump Rationale:
  N/A (initial creation)

-->

# DeepPerson Constitution

## Core Principles

### I. Registry Pattern for Model Management (NON-NEGOTIABLE)
All model profiles, configurations, and cache management MUST use the Registry pattern with thread-safe access. The `ModelRegistry` class in `src/registry.py` is the single source of truth for model profiles. All model loading, caching, and profile management MUST route through the registry. New models require ModelProfile registration before use. Model profiles MUST include: identifier, backbone_path, feature_dim, preprocessing config, and hardware requirements. Thread safety using `threading.RLock` MUST be enforced for all registry operations.

**Rationale**: Centralized model management ensures consistent state across concurrent operations, prevents duplicate model loading, and provides a single point for model profile validation and updates.

### II. Pipeline Pattern for Detection → Embedding → Search (NON-NEGOTIABLE)
The processing pipeline MUST follow the sequential pattern: Person Detection (detectors.py) → Feature Extraction (embeddings.py) → Similarity Search (search.py). Each stage MUST be independently testable and composable. The `DeepPerson` façade class orchestrates this pipeline but each component MUST expose clean interfaces for unit testing. The pipeline MUST handle: YOLO person detection with configurable thresholds, ResNet-50 Circle DG embedding generation with normalization options, and FAISS/sklearn similarity search with multiple distance metrics. Pipeline failures at any stage MUST provide clear error propagation without partial state corruption.

**Rationale**: The pipeline pattern enables independent testing of each stage, clear separation of concerns, and flexibility to swap components (e.g., different detectors or search backends) without affecting the overall workflow.

### III. Hardware Optimization with CPU/GPU Fallback (NON-NEGOTIABLE)
All operations MUST support automatic hardware detection with graceful CPU fallback. The system MUST detect CUDA availability and use GPU acceleration when present without requiring manual configuration. Device selection MUST be explicit in all class constructors and method signatures. GPU memory management MUST include batch size controls and cleanup mechanisms. Models MUST checkpoint state appropriately when switching devices. CPU MUST be fully functional as a complete fallback - no feature degradation allowed. Performance considerations MUST be documented: GPU provides ~10-50x speedup for embedding generation, FAISS GPU acceleration for large galleries (>10K embeddings).

**Rationale**: Hardware optimization ensures the component works across diverse deployment environments while maximizing performance on capable hardware. Automatic fallback maintains accessibility and reliability.

### IV. Gallery System with Multi-Modal Support (NON-NEGOTIABLE)
User Gallery operations MUST support multi-modal embeddings (body + face) with fusion-based retrieval. Gallery storage MUST follow the standardized format: `{gallery_name}_embeddings.npy`, `{gallery_name}_ids.npy`, `{gallery_name}_metadata.pkl`, `{gallery_name}_config.json`. Galleries MUST support both BODY and FACE modalities with separate embedding spaces and fusion weights. Gallery operations (create, update, add_images, retrieve) MUST be idempotent and thread-safe. Gallery serialization/deserialization MUST handle version compatibility for model upgrades. Clustering algorithms MUST support appearance variant grouping for robust person matching across different poses/lighting.

**Rationale**: Multi-modal support significantly improves retrieval accuracy by combining body shape and facial features. Standardized storage ensures interoperability, and thread safety enables concurrent gallery operations.

### V. Production Readiness with Comprehensive Observability (NON-NEGOTIABLE)
All operations MUST include structured logging, error handling, and performance metrics. Logging MUST use appropriate levels (DEBUG, INFO, WARNING, ERROR) with contextual information (model profiles, hardware info, timing). Error handling MUST be specific and actionable - no generic exceptions. Performance metrics MUST track: embedding generation time, detection confidence, search latency, gallery retrieval scores. Code coverage MUST meet project standards with unit, integration, and contract tests. Type hints REQUIRED for all public interfaces (mypy strict compliance). Thread safety validation REQUIRED for all shared state operations. Production readiness checklist MUST pass before deployment: error handling verified, logging configured, tests passing, type checking clean.

**Rationale**: Production systems require comprehensive observability for debugging, monitoring, and incident response. Structured logging and metrics enable operational visibility, while type hints and tests ensure code quality and reliability.

## Technology & Quality Standards

All code MUST follow Python 3.12+ standards with strict type checking. Dependencies MUST use `pyproject.toml` with proper optional extras (faiss-gpu, faiss-cpu, dev tools). Code formatting and linting MUST use `ruff check --fix` and `ruff format` with 88-character line length. All public classes and methods MUST have type hints and documentation. Import organization MUST follow stdlib → third-party → local with Ruff auto-formatting.

Testing requirements: Unit tests for individual components (detectors, embeddings, search), integration tests for full pipeline, contract tests for public API boundaries. All tests MUST use pytest with appropriate markers (@pytest.mark.unit, @pytest.mark.integration, @pytest.mark.slow). Test coverage targets: 80% minimum for core components, 90% for critical paths (detection → embedding → search pipeline).

Performance standards: Embedding generation should complete within 100ms per person on GPU (batch_size=1), gallery search should return top-10 results within 50ms for galleries up to 10K embeddings. Memory usage MUST be monitored and cleanup provided for long-running operations.

Model management standards: All model weights MUST be automatically downloaded on first use via `ModelManager`, cached locally for subsequent runs, and verified with checksum validation. Models MUST support hot-swapping without service restart. Backward compatibility MUST be maintained for model profile changes with migration utilities.

## Development Workflow

All changes MUST follow the feature branch workflow with corresponding PRs. Feature branches MUST use the format `###-feature-name` (e.g., `001-verify-api`). PRs MUST include: clear description of changes, link to related specifications, test coverage verification, and type checking results. Code review MUST verify: compliance with all constitution principles, test coverage adequacy, performance impact assessment, and backward compatibility considerations.

Pre-commit validation REQUIRED: `ruff check --fix` and `ruff format` for code quality, `mypy src/` for type safety, `pytest tests/` for test verification. PRs MUST NOT merge without passing all checks. Complexity additions MUST be justified - simpler alternatives considered and documented if rejected. Breaking changes MUST include migration strategy and deprecation timeline.

Dependency management: New dependencies MUST be evaluated for necessity, security, and maintenance burden. Optional dependencies MUST be documented with use cases. Version pinning MUST use conservative ranges (e.g., `>=2.0.0` for major frameworks). Security updates MUST be tracked and applied within reasonable timeframes.

Documentation requirements: All public APIs MUST have docstrings with examples, README updates for user-facing changes, API reference documentation for new endpoints. Architecture decisions MUST be documented in relevant files. Migration guides REQUIRED for breaking changes.

## Governance

**Constitution Supremacy**: This constitution supersedes all other development practices and guidelines. In conflicts between this constitution and other documentation, this constitution takes precedence.

**Amendment Process**: Amendments require:
1. Draft proposal with rationale, impact analysis, and backward compatibility assessment
2. Review period with stakeholder feedback collection
3. Version bump following semantic versioning rules:
   - MAJOR: Backward incompatible changes to principles or architecture
   - MINOR: New principles, sections, or materially expanded guidance
   - PATCH: Clarifications, wording improvements, non-semantic refinements
4. Migration strategy documentation if breaking changes
5. Approval from project maintainers

**Compliance Verification**: All PRs MUST verify constitution compliance through:
- Self-check against each applicable principle
- Automated tooling where possible (type checking, test coverage, linting)
- Manual review for complex architectural decisions
- Performance benchmarking for hardware optimization changes

**Version Control**: Constitution changes MUST be tracked in git with clear commit messages indicating version bumps and rationale. The `LAST_AMENDED_DATE` MUST be updated for all changes (including PATCH level). Historical versions MUST be preserved for audit trail.

**Exception Process**: Exceptions to principles require documented justification with:
- Specific business or technical reason for deviation
- Alternative approaches considered and rejected
- Mitigation strategy for risks introduced
- Sunset date for temporary exceptions
- Approval from project maintainers

**Guidance Integration**: Development guidance in `CLAUDE.md` MUST align with constitution principles. When principles are updated, all guidance documents MUST be reviewed and updated accordingly.

**Version**: 1.0.0 | **Ratified**: 2025-11-04 | **Last Amended**: 2025-11-05
