# Implementation Plan: FastAPI Stateless Service for DeepPerson

**Branch**: `001-fastapi-service` | **Date**: 2025-11-05 | **Spec**: [link](/home/heng.li/repo/DeepPerson/specs/001-fastapi-service/spec.md)
**Input**: Feature specification from `/specs/001-fastapi-service/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Transform DeepPerson library into a stateless FastAPI service with two API categories:
1. **Core API**: `/represent` and `/verify` endpoints for person recognition and embedding generation
2. **Health & Documentation API**: `/health`, `/docs`, and `/openapi.json` endpoints for monitoring and API documentation

Service will be purely stateless (no persistent storage), accept multipart/form-data and base64 images, provide automatic OpenAPI documentation, and support GPU/CPU with automatic fallback.

**Note**: Gallery management functionality will be implemented separately with OceanBase Dataset integration in future releases.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.12+
**Primary Dependencies**: FastAPI, Uvicorn, Pydantic, DeepPerson (local), Python-multipart
**Storage**: Purely stateless - no persistent storage required
**Testing**: pytest with unit, integration, and contract test markers
**Target Platform**: Linux server/container, cloud deployment ready
**Project Type**: REST API service
**Performance Goals**: <2s response time for core API, 100+ concurrent requests, <100ms health checks
**Constraints**: 4GB RAM minimum, 10MB max image size, no authentication (deferred), stateless design
**Scale/Scope**: Support batch processing, concurrent requests, no gallery persistence (deferred to OceanBase)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Principle I - Registry Pattern**: ✅ COMPLIANT
- Service uses existing DeepPerson library which implements ModelRegistry
- All model loading routes through registry with thread-safe access
- No new model management introduced, service is stateless consumer

**Principle II - Pipeline Pattern**: ✅ COMPLIANT
- Service leverages existing Detection → Embedding → Search pipeline via DeepPerson façade
- Each component remains independently testable
- No modifications to pipeline structure, only HTTP interface added

**Principle III - Hardware Optimization**: ✅ COMPLIANT
- Service implements automatic GPU/CPU detection (FR-013)
- CPU fallback fully supported (stateless, can run on any instance)
- Device selection explicit in initialization

**Principle IV - Gallery System**: ✅ DEFERRED
- Gallery functionality will be implemented with OceanBase Dataset integration in future releases
- Current MVP has no persistent storage requirements
- Will comply with gallery system principles when implemented

**Principle V - Production Readiness**: ✅ COMPLIANT
- Comprehensive logging required (FR-010)
- Error handling with standardized format (FR-004)
- Health checks for monitoring (FR-005)
- Testing requirements specified (pytest with markers)

**GATE STATUS**: ✅ PASS - All constitutional requirements satisfied (Gallery system deferred)

---

## Post-Design Constitution Re-Check

*After Phase 1 design completion*

**Principle I - Registry Pattern**: ✅ COMPLIANT
- Service integrates with existing ModelRegistry without modification
- All model operations route through registry as required
- Thread-safe cache management preserved

**Principle II - Pipeline Pattern**: ✅ COMPLIANT
- Service façade wraps existing pipeline without changing internals
- Detection → Embedding → Search flow maintained
- Each stage remains independently testable

**Principle III - Hardware Optimization**: ✅ COMPLIANT
- Automatic GPU/CPU detection implemented in service initialization
- Environment-based configuration supports diverse deployment scenarios
- Performance metrics exposed via health endpoint

**Principle IV - Gallery System**: ✅ DEFERRED
- Gallery API removed from current implementation
- Will be implemented with OceanBase Dataset integration in future releases
- When implemented, will follow multi-modal support and standardized storage principles

**Principle V - Production Readiness**: ✅ COMPLIANT
- Comprehensive logging with request tracing
- Standardized error handling with meaningful messages
- Health checks at multiple levels (liveness, readiness, performance)
- Test strategy defined (unit, integration, contract)

**RE-EVALUATION STATUS**: ✅ PASS - Design maintains full constitutional compliance (Gallery system intentionally deferred)

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── api.py                      # DeepPerson library (existing)
├── detectors.py                # DeepPerson library (existing)
├── embeddings.py               # DeepPerson library (existing)
└── ... (other existing DeepPerson files)

fastapi_service/                # FastAPI service implementation
├── __init__.py
├── app.py                      # FastAPI application factory
├── main.py                     # Uvicorn entry point
├── server.py                   # Server configuration
├── routes/                     # API route handlers
│   ├── __init__.py
│   ├── core.py                 # /represent, /verify endpoints
│   └── health.py               # /health, /docs endpoints
├── schemas/                    # Pydantic models
│   ├── __init__.py
│   ├── represent.py            # Represent request/response schemas
│   ├── verify.py               # Verify request/response schemas
│   ├── health.py               # Health check schemas
│   └── errors.py               # Error response schemas
├── middleware/                 # FastAPI middleware
│   ├── __init__.py
│   ├── logging.py              # Request/response logging
│   ├── cors.py                 # CORS headers
│   └── errors.py               # Global error handling
├── dependencies.py             # FastAPI dependencies (DeepPerson instance)
├── config.py                   # Service configuration (env vars)
└── utils/
    ├── __init__.py
    ├── image_io.py             # Image validation and processing
    ├── response.py             # Response formatting
    └── validators.py           # Custom validation functions

tests/
├── contract/                   # API contract tests
│   ├── test_api_contracts.py   # Validate JSON schemas
│   └── test_openapi_spec.py    # OpenAPI specification tests
├── integration/                # End-to-end API tests
│   ├── test_represent_api.py   # Represent endpoint tests
│   └── test_verify_api.py      # Verify endpoint tests
├── unit/                       # Component tests
│   ├── test_schemas.py         # Pydantic schema tests
│   ├── test_middleware.py      # Middleware tests
│   └── test_config.py          # Configuration tests
└── conftest.py                 # pytest configuration

pyproject.toml                  # Dependencies (FastAPI, uvicorn, etc.)
Dockerfile                      # Container definition
docker-compose.yml              # Local development setup
.env.example                    # Environment variables template
```

**Structure Details**:

**Public API (Exposed Interface)**:
- `src/api.py` - DeepPerson library public façade (existing, unchanged)
- `fastapi_service/app.py` - FastAPI app factory (public endpoint)

**FastAPI Service Module (`fastapi_service/`) Internal Structure**:
- `app.py` - FastAPI application factory that wraps DeepPerson API
- `main.py` - Uvicorn entry point
- `server.py` - Server startup/shutdown logic
- `routes/` - Organized by functional area (Core, Health)
- `schemas/` - Pydantic models for all API contracts
- `middleware/` - Request logging, CORS, error handling
- `dependencies.py` - FastAPI dependency injection for DeepPerson instance
- `config.py` - Environment-based configuration management
- `utils/` - Image processing and response formatting

**Integration Architecture**:
```
Client Request
    ↓
FastAPI Router (fastapi_service/routes/)
    ↓
Pydantic Validation (fastapi_service/schemas/)
    ↓
DeepPerson API (src/api.py methods)
    ↓
Response Formatting (fastapi_service/utils/response.py)
    ↓
Client Response
```

**Key Design Decisions**:
1. **Clean Separation**: FastAPI service code isolated in `fastapi_service/` directory
2. **Public Interface**: Only `src/api.py` exposed from core library
3. **No Pollution**: FastAPI code doesn't clutter the main `src/` directory
4. **Modular Routes**: Routes organized by functionality (core/health)
5. **Pure Statelessness**: No persistent storage, all processing in-memory
6. **Dependency Injection**: FastAPI dependencies manage DeepPerson lifecycle

**Test Structure**:
- `tests/contract/` - Validates API responses match JSON contracts
- `tests/integration/` - Full API testing with real images
- `tests/unit/` - Component tests for schemas, middleware, config

**Service Infrastructure**:
- `fastapi_service/main.py` - Uvicorn service entry point
- `pyproject.toml` - Dependencies: fastapi, uvicorn, python-multipart
- `Dockerfile` - Container with service layer
- `docker-compose.yml` - Dev environment

**Storage**:
- **None** - Purely stateless design, no persistent storage
- Gallery functionality deferred to OceanBase Dataset integration

**Gallery API Note**: Gallery routes and schemas removed. Will be implemented in future release with OceanBase Dataset.

## Generated Artifacts

### Phase 0 - Research
- ✅ **research.md** - Technical decisions, best practices, deployment architecture

### Phase 1 - Design & Contracts
- ✅ **data-model.md** - Complete entity model with 10 core entities, relationships, validation rules
- ✅ **quickstart.md** - Comprehensive usage guide with examples
- ✅ **contracts/** - API contract schemas:
  - `represent-request.json` - Embedding generation request schema
  - `represent-response.json` - Embedding response schema
  - `verify-request.json` - Identity verification request schema
  - `verify-response.json` - Verification response schema
  - `gallery-create-request.json` - Gallery creation schema
  - `gallery-search-request.json` - Gallery search schema
  - `health-response.json` - Health check response schema
  - `error-response.json` - Standardized error format schema

### Agent Context
- ✅ **CLAUDE.md** - Updated with FastAPI service technology stack

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |

## Summary

**Implementation Plan Complete** ✅

The FastAPI stateless service for DeepPerson has been fully designed with:
- **13 functional requirements** mapped to specific implementation tasks
- **5 API contracts** with detailed JSON schemas (represent, verify, health, error)
- **Core entities** focused on request/response and embeddings
- **Comprehensive documentation** including research, data model, and quickstart guide
- **Full constitutional compliance** verified at both design and post-design stages

**Gallery API Status**: Deferred to future release with OceanBase Dataset integration

**Next Step**: Proceed to `/speckit.tasks` to generate actionable implementation tasks

**Key Technology Decisions**:
- FastAPI framework for high-performance async API
- Pydantic for schema validation and OpenAPI generation
- Pure stateless design with no persistent storage
- Multi-modal support (body + face embeddings) via DeepPerson library
- GPU/CPU auto-detection with graceful fallback

