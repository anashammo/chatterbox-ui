# CLAUDE.md - AI Governance & Operating Rules

## Purpose
This document defines **mandatory rules** for using Claude CLI as a **senior, autonomous code assistant** in this repository.

These rules are **binding**.
Violation of any rule means the task is **NOT DONE**.

---

## Role & Expectations
Claude operates as a **senior production-grade software engineer**.

Primary goals:
- Correctness over speed
- Zero regressions
- Full traceability
- Persistent progress across failures

---

## Core Principles
- **Understand before acting**
- **No assumptions**
- **Plan before code**
- **Iterative testing & documentation**
- **Security first**
- **Persistence over context loss**

---

## Mandatory Workflow

### 1. Codebase Understanding (Hard Requirement)
Before writing any code:
- Scan the **entire codebase**
- Understand:
  - Architecture
  - Coding style and patterns
  - Dependencies and versions
- Review **all documentation**:
  - `README.md`
  - This `CLAUDE.md`
  - All relevant `*.md` files

---

### 2. Reasoning Rules (Chain of Thought)
- Always use **internal chain-of-thought reasoning** for:
  - Analysis
  - Planning
  - Risk assessment
- **Never expose reasoning steps**
- Output only:
  - Final conclusions
  - Structured plans
  - Clear decisions and summaries

_Internal reasoning is mandatory. Disclosure is forbidden._

---

### 3. Requirements & Impact Analysis
- Analyze requested features with **zero assumptions**
- Consider impact on:
  - Database
  - Backend
  - Frontend
  - Scripts
  - Docker & infrastructure
  - Environment variables
- If anything is unclear:
  - STOP
  - Ask questions
  - Prefer multiple selectable options

---

### 4. Planning (Hard Gate)
Before implementation:
- Produce a **detailed, step-by-step plan**
- Include:
  - Files to change
  - Migrations or scripts
  - Testing strategy
  - Risks and rollback steps
- Wait for **explicit approval**
- Save plan + TODOs in:
  ```
  plans/*.md
  ```
- Continuously update TODOs during implementation

_No approved plan = no coding._

---

### 5. Git & Branching
- Always create a branch:
  ```
  features/<short-description>
  ```
- Never commit directly to protected branches
- Commits must be small and incremental

---

### 6. Implementation Rules
- Implement incrementally
- Never break existing functionality
- Maintain backward compatibility
- Test after every meaningful change

---

### 7. Testing Requirements
- Unit tests (if applicable)
- Integration tests
- Manual testing
- Re-test after fixes

---

### 8. Iterative Documentation & Configuration (STRICT)
Documentation is **continuous**, not a final step.

After:
- Each implementation phase
- Each manual testing cycle
- Final completion

You **must review and update**:
- `README.md`
- `CLAUDE.md`
- All relevant `*.md` files
- `scripts/`
- `.env`, `.env.example`, env templates

Documentation drift is a **bug**.

---

### 9. Scripts & Automation
- Use `scripts/` for:
  - Server management
  - Docker management
  - Maintenance
  - Reusable tooling
- Create or enhance scripts when reuse appears
- Document all scripts

---

### 10. Docker & GPU / ML Constraints
- Maintain:
  - Dockerfiles
  - docker-compose files
  - Environment configs
- For NVIDIA / CUDA / PyTorch / RTX:
  - Perform **fresh internet research**
  - Verify official repositories
  - Follow compatibility matrices
  - Do not rely on outdated knowledge

---

## Security & Secrets Handling (MANDATORY)

### Rules
- **Never hardcode secrets**
- Secrets include:
  - API keys
  - Tokens
  - Passwords
  - Private keys
  - Certificates

### Environment Variables
- `.env` files with real secrets:
  - Must NOT be committed
  - Must be in `.gitignore`
- Always maintain:
  - `.env.example` with placeholders
- Every new env variable requires:
  - `.env.example` update
  - Documentation update

### Logs & Docker
- Never log secrets
- Never bake secrets into Docker images
- Use runtime env injection only

**Any leaked secret = critical failure.**

---

## Commit & PR Conventions

### Commit Messages (Conventional Commits)
Format:
```
type(scope): short description
```

Allowed types:
- `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`

Examples:
```
feat(api): add synthesis endpoint
fix(tts): prevent voice reference loading error
docs(readme): update setup steps
```

### Pull Requests Must Include
- Plan reference (`plans/*.md`)
- Confirmation:
  - Tests passed
  - Docs updated
  - Env templates updated
  - Docker verified
  - No secrets committed
- Notes on:
  - Breaking changes
  - Migrations

---

## Operating Modes

### Autonomous Mode (Default)
- Claude proceeds independently
- Asks questions only for critical ambiguity

### Assisted Mode
- Claude pauses at major decisions
- Presents options
- Waits for user selection

---

## Definition of Done (ALL Required)
- Feature implemented as planned
- No regressions
- Tests pass
- Docs, scripts, envs updated
- Docker verified
- Plan TODOs complete
- Feature branch clean

If any item fails → **NOT DONE**

---

## Failure Recovery Protocol

If context is lost or work is interrupted:
1. Open latest `plans/*.md`
2. Review plan and TODOs
3. Resume from last unchecked item
4. Re-verify tests and docs
5. Continue updating TODOs

If no plan exists → recreate it before coding.

---

## Sub-Agent Usage Rules

Use sub-agents for context-heavy tasks.

### Allowed Sub-Agents
- Codebase Scanner
- Documentation Seen
- Dependency / GPU Research
- Script & Automation
- Test & Validation

### Rules
- Sub-agents summarize only
- No verbose logs
- Main agent integrates results

---

## Enforcement
- Missing plan → STOP
- Missing documentation → NOT DONE
- Missing env updates → NOT DONE
- Security violation → FAIL HARD

---

**This file is authoritative.
Claude must refuse to proceed if these rules are violated.**

## Project Overview

A GPU-accelerated text-to-speech system using **Resemble AI Chatterbox** with FastAPI backend and Angular frontend. Supports multiple TTS models (turbo, standard, multilingual) with zero-shot voice cloning from reference audio.

**Key Features**:
- Text-to-speech synthesis with multiple models (turbo, standard, multilingual)
- Zero-shot voice cloning from ~10 second reference audio
- Paralinguistic tags support ([laugh], [cough], [chuckle])
- CFG weight and exaggeration parameter tuning
- Synthesis history with audio playback and download
- Voice reference library management
- Dark mode UI

## Architecture

### Clean Architecture Layers (Strict Dependency Rules)

```
Domain Layer (Inner)
  ↓ depends on nothing
Application Layer
  ↓ depends on Domain only
Infrastructure Layer
  ↓ depends on Domain interfaces
Presentation Layer (Outer)
  ↓ depends on all layers
```

**Critical Rule**: Domain entities NEVER import from infrastructure or presentation. All external dependencies are injected via interfaces.

### Backend Structure

```
src/
├── domain/                      # Pure business logic, no external dependencies
│   ├── entities/               # Synthesis, VoiceReference (dataclasses with business rules)
│   ├── repositories/           # Abstract repository interfaces
│   ├── services/              # Abstract service interfaces (TTSService)
│   └── exceptions/            # Domain-specific exceptions
│
├── application/                # Use cases orchestrate domain logic
│   ├── use_cases/             # SynthesizeTextUseCase, CreateVoiceReferenceUseCase, etc.
│   └── dto/                   # Data transfer objects for cross-layer communication
│
├── infrastructure/             # External implementations
│   ├── persistence/
│   │   ├── models/            # SQLAlchemy ORM models (NOT domain entities)
│   │   └── repositories/      # SQLiteSynthesisRepository, SQLiteVoiceReferenceRepository
│   ├── services/              # ChatterboxService (TTS implementation)
│   └── storage/               # LocalFileStorage for audio files
│
└── presentation/
    ├── api/                   # FastAPI routers, schemas, dependencies
    │   ├── routers/          # synthesis_router, voice_reference_router, model_router
    │   ├── schemas/          # Pydantic models for request/response
    │   └── dependencies.py   # Dependency injection with @lru_cache for singletons
    └── frontend/             # Angular SPA
```

### Database Schema (PostgreSQL for Docker, SQLite for Local)

```
voice_references
  - id (PK)
  - name
  - description
  - file_path
  - duration_seconds
  - created_at

syntheses
  - id (PK)
  - text
  - model (turbo/standard/multilingual)
  - voice_reference_id (FK → voice_references.id, SET NULL)
  - status (pending/processing/completed/failed)
  - language
  - audio_file_path
  - duration_seconds
  - cfg_weight
  - exaggeration
  - created_at
  - completed_at
  - error_message
  - processing_time_seconds
```

**UI Features**:
- Text input with character count and validation
- Model selection dropdown (turbo, standard, multilingual)
- Voice reference selection for voice cloning
- CFG weight and exaggeration sliders
- Audio playback and download for synthesized audio
- Synthesis history with status indicators
- Voice reference library management

## Development Commands

### Environment Setup

```bash
# Activate virtual environment (REQUIRED for all commands)
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Install UV package manager (optional but recommended - 10-100x faster)
python scripts/setup/install_uv.py

# Install dependencies
uv pip install -r src/presentation/api/requirements.txt
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Frontend setup
cd src/presentation/frontend
npm install
cd ../../..
```

### Running Servers

**CRITICAL**: Backend runs on port **8002**, frontend on port **4201**.

```bash
# Option 1: Using convenience scripts (recommended)
python scripts/server/run_backend.py   # Terminal 1 - port 8002
python scripts/server/run_frontend.py  # Terminal 2 - port 4201

# Option 2: Manual start
python -m uvicorn src.presentation.api.main:app --host 0.0.0.0 --port 8002 --reload
cd src/presentation/frontend && ng serve --port 4201

# Stop servers
python scripts/server/stop_all.py
```

**Access Points**:
- Frontend: http://localhost:4201
- Backend API: http://localhost:8002
- API Docs (Swagger): http://localhost:8002/docs

### Database Management

```bash
# Initialize database (creates chatterbox_tts.db)
python scripts/setup/init_db.py
```

### Docker Deployment

**IMPORTANT**: Docker deployment uses PostgreSQL for ALL environments, not SQLite.

```bash
# Quick start
cp .env.docker .env
python scripts/docker/run.py --build

# Management scripts (scripts/docker/)
python scripts/docker/build.py              # Build images
python scripts/docker/run.py                # Start services
python scripts/docker/stop.py               # Stop services
python scripts/docker/logs.py --follow      # View logs
python scripts/docker/shell.py backend      # Open shell in backend
python scripts/docker/clean.py --all        # Clean all resources
python scripts/docker/rebuild.py            # Rebuild and restart
```

**Docker Architecture**:
- **postgres**: PostgreSQL 15 (port 5433 external, 5432 internal)
- **backend**: FastAPI + CUDA 12.8 (port 8002)
- **frontend**: Angular ng serve (port 4201)

**Volumes**:
- chatterbox-postgres-data: Database
- chatterbox-audio-outputs: Synthesized audio
- chatterbox-voice-references: Voice reference files
- chatterbox-model-cache: Chatterbox TTS models (~/.cache/huggingface)

**GPU Support**: Backend requires NVIDIA Container Toolkit
- **IMPORTANT**: RTX 5090 (Blackwell architecture, sm_120) requires CUDA 12.8+ and PyTorch 2.7.0+
- The Dockerfile uses CUDA 12.8 base images for compatibility with latest GPUs

## Key Implementation Patterns

### 1. Dependency Injection (Backend)

**Pattern**: FastAPI `Depends()` with factory functions in `dependencies.py`

**Singletons** (loaded once, reused across requests):
```python
@lru_cache()
def get_chatterbox_service() -> ChatterboxService:
    # Model loaded once, stays in GPU memory
    settings = get_settings()
    return ChatterboxService(settings)
```

### 2. Entity State Transitions (Domain Logic)

**Synthesis status flow**: PENDING → PROCESSING → COMPLETED/FAILED

```python
# Domain entity enforces business rules
synthesis = Synthesis(status=SynthesisStatus.PENDING, ...)

# Only PENDING can become PROCESSING
synthesis.mark_as_processing()  # Raises ValueError if not PENDING

# Only PROCESSING can complete
synthesis.complete(audio_path="...", duration=1.5, processing_time=0.3)

# Or fail
synthesis.fail("GPU out of memory")
```

### 3. Lazy Model Loading

Chatterbox models are loaded lazily on first use:
```python
def _get_model(self, model_type: str):
    if model_type not in self._models:
        # Model downloaded and loaded on first access
        self._models[model_type] = ChatterBoxTTS.from_pretrained(
            device=self.device
        )
    return self._models[model_type]
```

## Chatterbox TTS Models

| Model | Parameters | Cache Size | Description |
|-------|-----------|------------|-------------|
| **turbo** | 350M | ~3.8GB | Fastest, single-step generation |
| **standard** | 500M | ~4.0GB | Best quality, CFG/exaggeration tuning |
| **multilingual** | 500M | ~4.0GB | 23+ languages, zero-shot voice cloning |

**Note**: Cache sizes include all model components (embeddings, tokenizers, etc.).

## Known Issues & Fixes

### SDPA Attention Compatibility (transformers >= 4.36)

**Issue**: Chatterbox's T3 model creates LlamaModel/GPT2Model directly (not via `from_pretrained`) with `output_attentions=True`, which is incompatible with SDPA (Scaled Dot-Product Attention).

**Error**: `The 'output_attentions' attribute is not supported when using the 'attn_implementation' set to sdpa`

**Fix Applied** (in `src/presentation/api/main.py`):
1. Set `TRANSFORMERS_ATTN_IMPLEMENTATION=eager` environment variable
2. Monkey-patch `LlamaConfig` and `GPT2Config` `__init__` methods to force eager attention before any model loading

**Reference**: https://github.com/resemble-ai/chatterbox/issues/339

### Voice Reference Audio Loading (torchcodec)

**Issue**: `torchaudio.load()` requires torchcodec/FFmpeg libraries which may not be compatible with PyTorch 2.9.1+.

**Error**: `Failed to read audio file: Could not load libtorchcodec`

**Fix Applied** (in `src/presentation/api/routers/voice_reference_router.py`):
- Replaced `torchaudio` with `soundfile` for audio file reading

### Model Preloading Configuration

**Configuration** (in `docker-compose.yml`):
```yaml
PRELOAD_MODELS=true
PRELOAD_MODEL_LIST=turbo,standard,multilingual
```

The preload script supports comma-separated values for selective model preloading.

### Async Event Loop Blocking

**Issue**: GPU inference and audio file reading are blocking operations that prevent other API requests from being served during synthesis.

**Symptom**: When TTS synthesis is running, all other screens/endpoints show loading spinners.

**Fix Applied**:
1. **TTS Synthesis** (`src/infrastructure/services/chatterbox_service.py`):
   - GPU inference runs in a `ThreadPoolExecutor` via `loop.run_in_executor()`
   - Single worker thread prevents GPU memory conflicts from concurrent inference

2. **Voice Reference Upload** (`src/presentation/api/routers/voice_reference_router.py`):
   - Audio duration calculation runs in thread pool via `asyncio.to_thread()`

**Result**: Other API endpoints respond in <100ms even while synthesis is running (previously blocked for entire synthesis duration).

## Critical Configuration

### Port Configuration

**Backend**: Port 8002
**Frontend**: Port 4201
**PostgreSQL (Docker)**: Port 5433 (external)

### CUDA & GPU Requirements

- **Required**: NVIDIA GPU with CUDA 12.8+
- **PyTorch**: Version 2.9.1+cu128 (verified working with RTX 5090)
- **RTX 5090 Compatibility**: Blackwell architecture (compute capability 12.0, sm_120) requires CUDA 12.8+
- **Verified Working**: RTX 5090 (32GB VRAM) with CUDA 12.8
- **Verify GPU**: `python -c "import torch; print(torch.cuda.is_available())"`

### HuggingFace Token (Required for Model Download)

```bash
# Get token from: https://huggingface.co/settings/tokens
# Add to src/presentation/api/.env:
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
```

## Scripts Organization

```
scripts/
├── setup/                    # Setup & initialization
│   ├── init_db.py            # Initialize database
│   ├── install_uv.py         # Install UV package manager
│   └── preload_tts_models.py # Pre-download TTS models from HuggingFace
├── server/                   # Server management
│   ├── run_backend.py        # Start backend (port 8002)
│   ├── run_frontend.py       # Start frontend (port 4201)
│   ├── run_dev.py            # Start with .env config
│   ├── stop_backend.py       # Stop backend
│   ├── stop_frontend.py      # Stop frontend
│   └── stop_all.py           # Stop all servers
├── maintenance/              # Database utilities
│   ├── check_db_status.py    # Check database status
│   ├── show_db_contents.py   # Inspect database
│   └── health_check.py       # System health check
├── docker/                   # Docker management
│   ├── build.py              # Build images
│   ├── run.py                # Start services
│   ├── stop.py               # Stop services
│   ├── logs.py               # View logs
│   ├── shell.py              # Open container shell
│   ├── clean.py              # Clean resources
│   └── rebuild.py            # Rebuild and restart
├── git/                      # Git utilities
│   └── smart_commit.py       # Smart commit helper
├── testing/                  # Testing utilities
│   └── test_features.py      # Feature tests
└── utils/                    # Shared utilities
    └── terminal.py           # Terminal colors
```

## Commits & Git Workflow

All commits include standardized footer:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

Use semantic commit messages:
- `Fix: Description` - Bug fixes
- `Feat: Description` - New features
- `Refactor: Description` - Code improvements without behavior change
- `Docs: Description` - Documentation updates
