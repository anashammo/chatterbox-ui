# Chatterbox UI: ASR to TTS Replacement Plan

**Status**: COMPLETED
**Created**: 2026-01-04
**Completed**: 2026-01-04
**Author**: Claude (Senior Engineer)
**Type**: Full System Replacement (ASR → TTS)

---

## Executive Summary

This plan describes the **complete replacement** of the existing ASR (Automatic Speech Recognition) system with a TTS (Text-to-Speech) system using Resemble AI's Chatterbox library. The application will be rebranded as **"Chatterbox UI"**.

**Key Points**:
- **Replace**, not complement - All ASR code will be removed
- **Remove LLM Enhancement** - Feature not applicable to TTS, completely removed
- **Reuse** architecture, styling, and patterns
- **New** database container and volumes (coexists with original ASR repo)
- **Both** standard TTS and voice cloning capabilities

---

## Table of Contents

1. [Requirements Summary](#1-requirements-summary)
2. [Coexistence Strategy](#2-coexistence-strategy)
3. [Architecture Overview](#3-architecture-overview)
4. [Files to Remove](#4-files-to-remove)
5. [Files to Adapt](#5-files-to-adapt)
6. [Files to Create](#6-files-to-create)
7. [Database Schema](#7-database-schema)
8. [Domain Layer](#8-domain-layer)
9. [Application Layer](#9-application-layer)
10. [Infrastructure Layer](#10-infrastructure-layer)
11. [Presentation Layer - API](#11-presentation-layer---api)
12. [Presentation Layer - Frontend](#12-presentation-layer---frontend)
13. [Configuration & Environment](#13-configuration--environment)
14. [Docker Configuration](#14-docker-configuration)
15. [Scripts & Automation](#15-scripts--automation)
16. [Testing Strategy](#16-testing-strategy)
17. [Implementation Phases](#17-implementation-phases)
18. [Risks & Mitigations](#18-risks--mitigations)
19. [TODOs Checklist](#19-todos-checklist)

---

## 1. Requirements Summary

### 1.1 Confirmed Requirements

| Requirement | Details |
|-------------|---------|
| **Scope** | Replace ASR with TTS (complete removal of ASR) |
| **TTS Library** | Chatterbox (Resemble AI) |
| **Capabilities** | Standard TTS + Voice Cloning |
| **Branding** | Rename to "Chatterbox UI" |
| **Database** | New container, new volumes |
| **Coexistence** | Must run alongside original ASR repo |
| **Reuse** | HTML, CSS, theme, architecture patterns |

### 1.2 Chatterbox Models

| Model | Import | Parameters | Features |
|-------|--------|------------|----------|
| Turbo | `chatterbox.tts_turbo.ChatterboxTurboTTS` | 350M | Fastest, single-step generation |
| Standard | `chatterbox.tts.ChatterboxTTS` | 500M | Highest quality |
| Multilingual | `chatterbox.mtl_tts.ChatterboxMultilingualTTS` | 500M | 23+ languages |

### 1.3 Chatterbox Capabilities

- **Voice Cloning**: Zero-shot from ~10s reference audio
- **Paralinguistic Tags**: `[laugh]`, `[cough]`, `[chuckle]`
- **Parameters**: `cfg_weight` (0-1), `exaggeration` (0-1+)
- **Output**: WAV audio via `torchaudio.save()`
- **Sample Rate**: Available via `model.sr`

### 1.4 LLM Enhancement Feature - REMOVED (Not Applicable to TTS)

The ASR application includes an LLM Enhancement feature that post-processes transcriptions for grammar correction, formatting, and filler word removal. **This feature is NOT applicable to TTS and will be completely removed.**

#### Rationale for Removal

| ASR (Has LLM Enhancement) | TTS (No LLM Enhancement) |
|---------------------------|--------------------------|
| Input: Audio → Output: Text | Input: Text → Output: Audio |
| LLM improves text quality | User provides the text (already formatted) |
| Fixes transcription errors | No errors to fix in user-provided text |
| Removes filler words (um, uh) | User controls input text directly |

**Conclusion**: LLM enhancement adds value to ASR output but has no purpose in TTS where the user already provides the input text.

#### LLM Components to Remove

**Domain Layer:**
| File | Purpose |
|------|---------|
| `src/domain/services/llm_enhancement_service.py` | LLM service interface |

**Infrastructure Layer:**
| File | Purpose |
|------|---------|
| `src/infrastructure/services/llm_enhancement_service_impl.py` | LLM service implementation |
| `src/infrastructure/llm/llm_client.py` | OpenAI-compatible LLM HTTP client |

**Application Layer:**
| File | Purpose |
|------|---------|
| `src/application/use_cases/enhance_transcription_use_case.py` | LLM enhancement use case |
| `src/application/enhancement/enhancement_agent.py` | LangGraph agent workflow |
| `src/application/enhancement/prompts.py` | LLM system/user prompts |

**Presentation Layer - API:**
| File | Purpose |
|------|---------|
| `src/presentation/api/routers/llm_enhancement_router.py` | `/enhance` endpoint |

**Database Fields to Remove (from `transcriptions` table):**
| Field | Purpose |
|-------|---------|
| `enable_llm_enhancement` | Boolean flag |
| `enhanced_text` | LLM output text |
| `llm_processing_time_seconds` | LLM timing |
| `llm_enhancement_status` | processing/completed/failed |
| `llm_error_message` | LLM error details |

**Frontend Components to Remove:**
| Component/Feature | Purpose |
|-------------------|---------|
| "Enhance with LLM" button | Trigger enhancement |
| LLM enhancement checkbox | Enable/disable in forms |
| Dual text display (original + enhanced) | Show both versions |
| LLM status badges | Show enhancement status |
| Enhanced text preview in History | Display enhanced text |
| "Copy Enhanced" button | Copy enhanced text |

**Environment Variables to Remove:**
| Variable | Purpose |
|----------|---------|
| `LLM_BASE_URL` | Ollama/LM Studio URL |
| `LLM_MODEL` | Model name (llama3, etc.) |
| `LLM_TIMEOUT_SECONDS` | Request timeout |
| `LLM_TEMPERATURE` | Generation temperature |

**Dependencies to Remove:**
| Package | Purpose |
|---------|---------|
| `langchain` | LLM framework |
| `langgraph` | Agent workflow |
| `openai` | OpenAI-compatible client |

---

## 2. Coexistence Strategy

Since this repo will run **alongside** the original ASR repo, all identifiers must be unique:

### 2.1 Port Allocation

| Service | ASR Repo (Original) | TTS Repo (This) |
|---------|---------------------|-----------------|
| Backend | 8001 | **8002** |
| Frontend | 4200 | **4201** |
| PostgreSQL | 5432 | **5433** |

### 2.2 Docker Container Names

| Service | ASR Repo | TTS Repo |
|---------|----------|----------|
| Database | `whisper-postgres` | `chatterbox-postgres` |
| Backend | `whisper-backend` | `chatterbox-backend` |
| Frontend | `whisper-frontend` | `chatterbox-frontend` |

### 2.3 Docker Volume Names

| Purpose | ASR Repo | TTS Repo |
|---------|----------|----------|
| Database | `whisper-postgres-data` | `chatterbox-postgres-data` |
| Audio Files | `whisper-uploads` | `chatterbox-audio-outputs` |
| Voice Refs | N/A | `chatterbox-voice-references` |
| Model Cache | `whisper-huggingface-cache` | `chatterbox-model-cache` |

### 2.4 Database Name

| ASR Repo | TTS Repo |
|----------|----------|
| `whisper_transcriptions` | `chatterbox_tts` |

---

## 3. Architecture Overview

### 3.1 Data Flow Comparison

```
ASR (Remove):                    TTS (New):
Audio File → Whisper → Text      Text → Chatterbox → Audio File
     ↑                                ↑
   Input                            Input
     ↓                                ↓
   Output                           Output
```

### 3.2 Clean Architecture (Preserved)

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  Angular Frontend              │  FastAPI Backend               │
│  ├── TextInput Component       │  ├── synthesis_router.py       │
│  ├── VoiceUpload Component     │  ├── voice_router.py           │
│  ├── SynthesisHistory          │  ├── model_router.py           │
│  └── SynthesisDetail           │  └── schemas/                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  Use Cases:                                                     │
│  ├── SynthesizeSpeechUseCase                                    │
│  ├── ResynthesizeSpeechUseCase                                  │
│  ├── GetSynthesisHistoryUseCase                                 │
│  ├── GetSynthesisUseCase                                        │
│  ├── DeleteSynthesisUseCase                                     │
│  ├── UploadVoiceReferenceUseCase                                │
│  ├── GetVoiceReferencesUseCase                                  │
│  └── DeleteVoiceReferenceUseCase                                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DOMAIN LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Entities:                     │  Services (Interfaces):        │
│  ├── Synthesis                 │  └── TextToSpeechService       │
│  └── VoiceReference            │                                │
│                                │  Repositories (Interfaces):    │
│  Value Objects:                │  ├── SynthesisRepository       │
│  └── TTSModelInfo              │  └── VoiceReferenceRepository  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  Services:                     │  Persistence:                  │
│  └── ChatterboxService         │  ├── synthesis_model.py        │
│      (implements               │  ├── voice_reference_model.py  │
│       TextToSpeechService)     │  └── repositories/             │
│                                │                                │
│  Storage:                      │  Config:                       │
│  └── LocalFileStorage          │  └── settings.py               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Files to Remove

### 4.1 Domain Layer

| File | Reason |
|------|--------|
| `src/domain/services/speech_recognition_service.py` | ASR interface |
| `src/domain/services/llm_enhancement_service.py` | ASR post-processing |

### 4.2 Infrastructure Layer

| File | Reason |
|------|--------|
| `src/infrastructure/services/faster_whisper_service.py` | Whisper ASR |
| `src/infrastructure/services/model_download_tracker.py` | Whisper model tracking |
| `src/infrastructure/services/llm_enhancement_service_impl.py` | LLM enhancement |
| `src/infrastructure/llm/llm_client.py` | LLM client |

### 4.3 Application Layer

| File | Reason |
|------|--------|
| `src/application/use_cases/transcribe_audio_use_case.py` | ASR use case |
| `src/application/use_cases/retranscribe_audio_use_case.py` | ASR use case |
| `src/application/use_cases/enhance_transcription_use_case.py` | LLM enhancement |
| `src/application/enhancement/enhancement_agent.py` | LangGraph agent |
| `src/application/enhancement/prompts.py` | LLM prompts |

### 4.4 Presentation Layer - API

| File | Reason |
|------|--------|
| `src/presentation/api/routers/transcription_router.py` | ASR endpoints |
| `src/presentation/api/routers/llm_enhancement_router.py` | LLM endpoints |
| `src/presentation/api/schemas/transcription_schema.py` | ASR schemas |

### 4.5 Presentation Layer - Frontend

| File | Reason |
|------|--------|
| `src/presentation/frontend/src/app/features/upload/*` | Audio upload |
| `src/presentation/frontend/src/app/features/transcription/*` | Transcription view |
| `src/presentation/frontend/src/app/features/history/*` | ASR history |
| `src/presentation/frontend/src/app/core/services/transcription.service.ts` | ASR service |
| `src/presentation/frontend/src/app/core/models/transcription.model.ts` | ASR model |

### 4.6 Database Files (Fresh Start)

| File | Reason |
|------|--------|
| `whisper_transcriptions.db` | SQLite database (if exists) |
| `uploads/*` | ASR audio uploads |

---

## 5. Files to Adapt

### 5.1 Domain Layer

| Current File | New Name | Changes |
|--------------|----------|---------|
| `src/domain/entities/transcription.py` | `synthesis.py` | New entity for TTS |
| `src/domain/entities/audio_file.py` | `voice_reference.py` | Voice cloning refs |
| `src/domain/repositories/transcription_repository.py` | `synthesis_repository.py` | TTS repository |
| `src/domain/repositories/audio_file_repository.py` | `voice_reference_repository.py` | Voice refs |

### 5.2 Infrastructure Layer

| Current File | New Name | Changes |
|--------------|----------|---------|
| `src/infrastructure/persistence/models/transcription_model.py` | `synthesis_model.py` | TTS schema |
| `src/infrastructure/persistence/models/audio_file_model.py` | `voice_reference_model.py` | Voice refs |
| `src/infrastructure/persistence/repositories/sqlite_transcription_repository.py` | `sqlite_synthesis_repository.py` | TTS impl |
| `src/infrastructure/persistence/repositories/sqlite_audio_file_repository.py` | `sqlite_voice_reference_repository.py` | Voice impl |
| `src/infrastructure/config/settings.py` | (same) | TTS settings |

### 5.3 Application Layer

| Current File | New Name | Changes |
|--------------|----------|---------|
| `src/application/use_cases/get_transcription_use_case.py` | `get_synthesis_use_case.py` | TTS |
| `src/application/use_cases/get_transcription_history_use_case.py` | `get_synthesis_history_use_case.py` | TTS |
| `src/application/use_cases/delete_transcription_use_case.py` | `delete_synthesis_use_case.py` | TTS |
| `src/application/use_cases/delete_audio_file_use_case.py` | `delete_voice_reference_use_case.py` | Voice |
| `src/application/use_cases/get_audio_file_transcriptions_use_case.py` | (remove) | N/A |
| `src/application/dto/transcription_dto.py` | `synthesis_dto.py` | TTS DTO |
| `src/application/dto/audio_upload_dto.py` | `voice_reference_dto.py` | Voice DTO |

### 5.4 Presentation Layer - API

| Current File | New Name | Changes |
|--------------|----------|---------|
| `src/presentation/api/routers/audio_file_router.py` | `voice_router.py` | Voice endpoints |
| `src/presentation/api/routers/model_router.py` | (same) | TTS models |
| `src/presentation/api/dependencies.py` | (same) | TTS dependencies |
| `src/presentation/api/main.py` | (same) | Router registration |

### 5.5 Configuration Files

| File | Changes |
|------|---------|
| `.env.example` | TTS variables |
| `.env.docker` | TTS + new ports/volumes |
| `docker-compose.yml` | New containers/volumes/ports |
| `CLAUDE.md` | TTS documentation |
| `README.md` | TTS documentation |

---

## 6. Files to Create

### 6.1 Domain Layer

| File | Purpose |
|------|---------|
| `src/domain/entities/synthesis.py` | Synthesis entity |
| `src/domain/entities/voice_reference.py` | Voice reference entity |
| `src/domain/services/text_to_speech_service.py` | TTS interface |
| `src/domain/repositories/synthesis_repository.py` | Synthesis repo interface |
| `src/domain/repositories/voice_reference_repository.py` | Voice repo interface |
| `src/domain/value_objects/tts_model_info.py` | Model specifications |

### 6.2 Infrastructure Layer

| File | Purpose |
|------|---------|
| `src/infrastructure/services/chatterbox_service.py` | Chatterbox TTS impl |
| `src/infrastructure/persistence/models/synthesis_model.py` | ORM model |
| `src/infrastructure/persistence/models/voice_reference_model.py` | ORM model |
| `src/infrastructure/persistence/repositories/sqlite_synthesis_repository.py` | Repo impl |
| `src/infrastructure/persistence/repositories/sqlite_voice_reference_repository.py` | Repo impl |

### 6.3 Application Layer

| File | Purpose |
|------|---------|
| `src/application/use_cases/synthesize_speech_use_case.py` | Generate TTS |
| `src/application/use_cases/resynthesize_speech_use_case.py` | Re-generate |
| `src/application/use_cases/get_synthesis_use_case.py` | Get single |
| `src/application/use_cases/get_synthesis_history_use_case.py` | Get history |
| `src/application/use_cases/delete_synthesis_use_case.py` | Delete |
| `src/application/use_cases/upload_voice_reference_use_case.py` | Upload voice |
| `src/application/use_cases/get_voice_references_use_case.py` | List voices |
| `src/application/use_cases/delete_voice_reference_use_case.py` | Delete voice |
| `src/application/dto/synthesis_dto.py` | Synthesis DTO |
| `src/application/dto/voice_reference_dto.py` | Voice DTO |

### 6.4 Presentation Layer - API

| File | Purpose |
|------|---------|
| `src/presentation/api/routers/synthesis_router.py` | TTS endpoints |
| `src/presentation/api/routers/voice_router.py` | Voice endpoints |
| `src/presentation/api/schemas/synthesis_schema.py` | Request/Response |
| `src/presentation/api/schemas/voice_schema.py` | Voice schemas |

### 6.5 Presentation Layer - Frontend

| File | Purpose |
|------|---------|
| `src/presentation/frontend/src/app/core/models/synthesis.model.ts` | Synthesis model |
| `src/presentation/frontend/src/app/core/models/voice-reference.model.ts` | Voice model |
| `src/presentation/frontend/src/app/core/services/synthesis.service.ts` | TTS service |
| `src/presentation/frontend/src/app/core/services/voice.service.ts` | Voice service |
| `src/presentation/frontend/src/app/features/text-input/*` | Text input component |
| `src/presentation/frontend/src/app/features/synthesis-history/*` | History component |
| `src/presentation/frontend/src/app/features/synthesis-detail/*` | Detail component |
| `src/presentation/frontend/src/app/features/voice-library/*` | Voice management |

### 6.6 Scripts

| File | Purpose |
|------|---------|
| `scripts/migrations/create_tts_schema.py` | Database migration |
| `scripts/setup/preload_tts_models.py` | Model preloading |

---

## 7. Database Schema

### 7.1 New Tables

#### `voice_references` Table

```sql
CREATE TABLE voice_references (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(512) NOT NULL UNIQUE,
    file_size_bytes INTEGER NOT NULL,
    mime_type VARCHAR(50) NOT NULL,
    duration_seconds FLOAT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_voice_duration CHECK (duration_seconds >= 5 AND duration_seconds <= 30)
);

CREATE INDEX idx_voice_references_created_at ON voice_references(created_at DESC);
```

#### `syntheses` Table

```sql
CREATE TABLE syntheses (
    id VARCHAR(36) PRIMARY KEY,

    -- Input
    input_text TEXT NOT NULL,
    text_length INTEGER NOT NULL,

    -- Model Configuration
    model VARCHAR(50) NOT NULL,  -- turbo, standard, multilingual
    language VARCHAR(10),        -- ISO code for multilingual model

    -- Voice Configuration
    voice_reference_id VARCHAR(36) REFERENCES voice_references(id) ON DELETE SET NULL,
    cfg_weight FLOAT DEFAULT 0.5,
    exaggeration FLOAT DEFAULT 0.5,

    -- Output
    output_file_path VARCHAR(512),
    output_duration_seconds FLOAT,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    processing_time_seconds FLOAT,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,

    CONSTRAINT chk_status CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    CONSTRAINT chk_model CHECK (model IN ('turbo', 'standard', 'multilingual'))
);

CREATE INDEX idx_syntheses_created_at ON syntheses(created_at DESC);
CREATE INDEX idx_syntheses_status ON syntheses(status);
CREATE INDEX idx_syntheses_voice_ref ON syntheses(voice_reference_id);
```

### 7.2 Removed Tables (from ASR)

- `audio_files` - Replaced by `voice_references`
- `transcriptions` - Replaced by `syntheses`

---

## 8. Domain Layer

### 8.1 Synthesis Entity

```python
# src/domain/entities/synthesis.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

class SynthesisStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TTSModel(Enum):
    TURBO = "turbo"
    STANDARD = "standard"
    MULTILINGUAL = "multilingual"

@dataclass
class Synthesis:
    """Domain entity representing a TTS synthesis request and result."""

    id: str
    input_text: str
    text_length: int
    model: TTSModel
    status: SynthesisStatus = SynthesisStatus.PENDING

    # Optional configuration
    language: Optional[str] = None
    voice_reference_id: Optional[str] = None
    cfg_weight: float = 0.5
    exaggeration: float = 0.5

    # Output
    output_file_path: Optional[str] = None
    output_duration_seconds: Optional[float] = None

    # Processing
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Business rules
    MAX_TEXT_LENGTH = 5000

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if len(self.input_text) > self.MAX_TEXT_LENGTH:
            raise ValueError(f"Text exceeds {self.MAX_TEXT_LENGTH} characters")
        if not self.input_text.strip():
            raise ValueError("Text cannot be empty")
        if self.model == TTSModel.MULTILINGUAL and not self.language:
            raise ValueError("Language required for multilingual model")
        if self.cfg_weight < 0 or self.cfg_weight > 1:
            raise ValueError("cfg_weight must be between 0 and 1")
        if self.exaggeration < 0:
            raise ValueError("exaggeration must be non-negative")

    def mark_as_processing(self) -> None:
        if self.status != SynthesisStatus.PENDING:
            raise ValueError(f"Cannot process from {self.status.value} status")
        self.status = SynthesisStatus.PROCESSING

    def complete(
        self,
        output_file_path: str,
        output_duration_seconds: float,
        processing_time_seconds: float
    ) -> None:
        if self.status != SynthesisStatus.PROCESSING:
            raise ValueError(f"Cannot complete from {self.status.value} status")
        self.status = SynthesisStatus.COMPLETED
        self.output_file_path = output_file_path
        self.output_duration_seconds = output_duration_seconds
        self.processing_time_seconds = processing_time_seconds
        self.completed_at = datetime.utcnow()

    def fail(self, error_message: str) -> None:
        if self.status != SynthesisStatus.PROCESSING:
            raise ValueError(f"Cannot fail from {self.status.value} status")
        self.status = SynthesisStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
```

### 8.2 VoiceReference Entity

```python
# src/domain/entities/voice_reference.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Set

@dataclass
class VoiceReference:
    """Domain entity for voice cloning reference audio."""

    id: str
    name: str
    original_filename: str
    file_path: str
    file_size_bytes: int
    mime_type: str
    duration_seconds: float
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Validation constants
    SUPPORTED_AUDIO_TYPES: ClassVar[Set[str]] = {
        "audio/wav", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/flac", "audio/ogg"
    }
    MIN_DURATION_SECONDS: ClassVar[float] = 5.0
    MAX_DURATION_SECONDS: ClassVar[float] = 30.0
    IDEAL_DURATION_SECONDS: ClassVar[float] = 10.0
    MAX_FILE_SIZE_MB: ClassVar[int] = 10

    def __post_init__(self):
        self._validate()

    def _validate(self) -> None:
        if self.mime_type not in self.SUPPORTED_AUDIO_TYPES:
            raise ValueError(f"Unsupported audio type: {self.mime_type}")
        if self.duration_seconds < self.MIN_DURATION_SECONDS:
            raise ValueError(f"Duration must be at least {self.MIN_DURATION_SECONDS}s")
        if self.duration_seconds > self.MAX_DURATION_SECONDS:
            raise ValueError(f"Duration must not exceed {self.MAX_DURATION_SECONDS}s")
        if self.file_size_bytes > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {self.MAX_FILE_SIZE_MB}MB")
        if not self.name.strip():
            raise ValueError("Voice name cannot be empty")
```

### 8.3 TextToSpeechService Interface

```python
# src/domain/services/text_to_speech_service.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class TextToSpeechService(ABC):
    """Abstract interface for TTS services."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        model: str = "turbo",
        voice_reference_path: Optional[str] = None,
        language: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text.

        Returns:
            {
                "audio_data": bytes,
                "duration_seconds": float,
                "sample_rate": int
            }
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Return available TTS models with metadata."""
        pass

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return supported language codes for multilingual model."""
        pass

    @abstractmethod
    def is_model_loaded(self, model: str) -> bool:
        """Check if a model is currently loaded."""
        pass
```

---

## 9. Application Layer

### 9.1 SynthesizeSpeechUseCase

```python
# src/application/use_cases/synthesize_speech_use_case.py

import uuid
import time
from typing import Optional
from ..dto.synthesis_dto import SynthesisDTO
from ...domain.entities.synthesis import Synthesis, TTSModel, SynthesisStatus
from ...domain.repositories.synthesis_repository import SynthesisRepository
from ...domain.repositories.voice_reference_repository import VoiceReferenceRepository
from ...domain.services.text_to_speech_service import TextToSpeechService
from ...application.interfaces.file_storage_interface import FileStorageInterface

class SynthesizeSpeechUseCase:
    def __init__(
        self,
        synthesis_repo: SynthesisRepository,
        voice_repo: VoiceReferenceRepository,
        tts_service: TextToSpeechService,
        file_storage: FileStorageInterface
    ):
        self.synthesis_repo = synthesis_repo
        self.voice_repo = voice_repo
        self.tts_service = tts_service
        self.file_storage = file_storage

    async def execute(
        self,
        text: str,
        model: str = "turbo",
        voice_reference_id: Optional[str] = None,
        language: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5
    ) -> SynthesisDTO:
        # 1. Validate voice reference if provided
        voice_ref_path = None
        if voice_reference_id:
            voice_ref = await self.voice_repo.get_by_id(voice_reference_id)
            if not voice_ref:
                raise ValueError(f"Voice reference not found: {voice_reference_id}")
            voice_ref_path = voice_ref.file_path

        # 2. Create synthesis entity
        synthesis = Synthesis(
            id=str(uuid.uuid4()),
            input_text=text,
            text_length=len(text),
            model=TTSModel(model),
            language=language,
            voice_reference_id=voice_reference_id,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration
        )

        # 3. Persist initial state
        synthesis = await self.synthesis_repo.create(synthesis)

        # 4. Process TTS
        try:
            synthesis.mark_as_processing()
            await self.synthesis_repo.update(synthesis)

            start_time = time.time()

            # 5. Generate speech
            result = await self.tts_service.synthesize(
                text=text,
                model=model,
                voice_reference_path=voice_ref_path,
                language=language,
                cfg_weight=cfg_weight,
                exaggeration=exaggeration
            )

            processing_time = time.time() - start_time

            # 6. Save audio file
            output_path = await self.file_storage.save_synthesis_audio(
                audio_data=result["audio_data"],
                synthesis_id=synthesis.id,
                sample_rate=result["sample_rate"]
            )

            # 7. Complete synthesis
            synthesis.complete(
                output_file_path=output_path,
                output_duration_seconds=result["duration_seconds"],
                processing_time_seconds=processing_time
            )

        except Exception as e:
            synthesis.fail(str(e))

        # 8. Persist final state
        synthesis = await self.synthesis_repo.update(synthesis)

        return SynthesisDTO.from_entity(synthesis)
```

### 9.2 SynthesisDTO

```python
# src/application/dto/synthesis_dto.py

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from ...domain.entities.synthesis import Synthesis

@dataclass
class SynthesisDTO:
    id: str
    input_text: str
    text_length: int
    model: str
    status: str
    language: Optional[str]
    voice_reference_id: Optional[str]
    voice_reference_name: Optional[str]
    cfg_weight: float
    exaggeration: float
    output_file_path: Optional[str]
    output_duration_seconds: Optional[float]
    error_message: Optional[str]
    processing_time_seconds: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]

    @classmethod
    def from_entity(
        cls,
        entity: Synthesis,
        voice_ref_name: Optional[str] = None
    ) -> "SynthesisDTO":
        return cls(
            id=entity.id,
            input_text=entity.input_text,
            text_length=entity.text_length,
            model=entity.model.value,
            status=entity.status.value,
            language=entity.language,
            voice_reference_id=entity.voice_reference_id,
            voice_reference_name=voice_ref_name,
            cfg_weight=entity.cfg_weight,
            exaggeration=entity.exaggeration,
            output_file_path=entity.output_file_path,
            output_duration_seconds=entity.output_duration_seconds,
            error_message=entity.error_message,
            processing_time_seconds=entity.processing_time_seconds,
            created_at=entity.created_at,
            completed_at=entity.completed_at
        )
```

---

## 10. Infrastructure Layer

### 10.1 ChatterboxService

```python
# src/infrastructure/services/chatterbox_service.py

import torch
import torchaudio
import asyncio
import io
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from ...domain.services.text_to_speech_service import TextToSpeechService
from ..config.settings import Settings

logger = logging.getLogger(__name__)

class ChatterboxService(TextToSpeechService):
    """Chatterbox TTS implementation supporting Turbo, Standard, and Multilingual models."""

    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru",
        "ja", "ko", "zh", "ar", "hi", "tr", "vi", "th", "id",
        "sv", "da", "no", "fi", "cs"
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = settings.tts_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._models: Dict[str, Any] = {}

        logger.info(f"ChatterboxService initialized with device: {self.device}")

        # Preload default model
        if settings.tts_preload_model:
            self._get_or_load_model(settings.tts_default_model)

    def _get_or_load_model(self, model_name: str):
        """Lazy-load and cache models."""
        if model_name not in self._models:
            logger.info(f"Loading Chatterbox {model_name} model...")

            if model_name == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self._models[model_name] = ChatterboxTurboTTS.from_pretrained(device=self.device)
            elif model_name == "standard":
                from chatterbox.tts import ChatterboxTTS
                self._models[model_name] = ChatterboxTTS.from_pretrained(device=self.device)
            elif model_name == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self._models[model_name] = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            logger.info(f"Chatterbox {model_name} model loaded successfully")

        return self._models[model_name]

    async def synthesize(
        self,
        text: str,
        model: str = "turbo",
        voice_reference_path: Optional[str] = None,
        language: Optional[str] = None,
        cfg_weight: float = 0.5,
        exaggeration: float = 0.5
    ) -> Dict[str, Any]:
        """Synthesize speech from text using Chatterbox."""

        tts_model = self._get_or_load_model(model)

        # Build generation kwargs
        kwargs = {}

        if voice_reference_path:
            kwargs["audio_prompt_path"] = voice_reference_path

        if model == "standard":
            kwargs["cfg_weight"] = cfg_weight
            kwargs["exaggeration"] = exaggeration

        if language and model == "multilingual":
            kwargs["language_id"] = language

        # Generate audio (run in thread pool for async)
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(
            None,
            lambda: tts_model.generate(text, **kwargs)
        )

        # Get sample rate from model
        sample_rate = tts_model.sr

        # Calculate duration
        duration_seconds = wav.shape[-1] / sample_rate

        # Convert to bytes (WAV format)
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav.cpu(), sample_rate, format="wav")
        audio_data = buffer.getvalue()

        return {
            "audio_data": audio_data,
            "duration_seconds": duration_seconds,
            "sample_rate": sample_rate
        }

    def get_available_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "turbo",
                "display_name": "Chatterbox Turbo",
                "parameters": "350M",
                "description": "Fastest model, single-step generation",
                "supports_voice_cloning": True,
                "supports_multilingual": False,
                "supports_paralinguistics": True
            },
            {
                "name": "standard",
                "display_name": "Chatterbox Standard",
                "parameters": "500M",
                "description": "Highest quality, cfg/exaggeration tuning",
                "supports_voice_cloning": True,
                "supports_multilingual": False,
                "supports_paralinguistics": True
            },
            {
                "name": "multilingual",
                "display_name": "Chatterbox Multilingual",
                "parameters": "500M",
                "description": "23+ languages support",
                "supports_voice_cloning": True,
                "supports_multilingual": True,
                "supports_paralinguistics": True
            }
        ]

    def get_supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES.copy()

    def is_model_loaded(self, model: str) -> bool:
        return model in self._models
```

### 10.2 Settings Extension

```python
# Add to src/infrastructure/config/settings.py

class Settings(BaseSettings):
    # Application
    app_name: str = "Chatterbox UI"
    app_version: str = "1.0.0"
    debug: bool = False

    # API (NEW PORTS)
    api_host: str = "0.0.0.0"
    api_port: int = 8002  # Changed from 8001
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:4201"]  # Changed from 4200

    # Database (NEW NAME)
    database_url: str = "sqlite:///./chatterbox_tts.db"

    # TTS Configuration (NEW)
    tts_device: str = "cuda"
    tts_default_model: str = "turbo"
    tts_preload_model: bool = True
    tts_max_text_length: int = 5000

    # File Storage (NEW DIRECTORIES)
    audio_output_dir: str = "./audio_outputs"
    voice_reference_dir: str = "./voice_references"
    voice_min_duration: float = 5.0
    voice_max_duration: float = 30.0
    voice_max_file_size_mb: int = 10

    class Config:
        env_file = ".env"
        case_sensitive = False
```

---

## 11. Presentation Layer - API

### 11.1 API Endpoints

#### Synthesis Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/syntheses` | Generate speech from text |
| GET | `/api/v1/syntheses` | List syntheses (paginated) |
| GET | `/api/v1/syntheses/{id}` | Get specific synthesis |
| DELETE | `/api/v1/syntheses/{id}` | Delete synthesis |
| GET | `/api/v1/syntheses/{id}/audio` | Stream/download audio |

#### Voice Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/voices` | Upload voice reference |
| GET | `/api/v1/voices` | List voice references |
| GET | `/api/v1/voices/{id}` | Get specific voice |
| DELETE | `/api/v1/voices/{id}` | Delete voice reference |
| GET | `/api/v1/voices/{id}/audio` | Play voice sample |

#### Model Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/models` | List available TTS models |
| GET | `/api/v1/models/{name}/status` | Check if model is loaded |
| GET | `/api/v1/languages` | List supported languages |

### 11.2 Synthesis Router

```python
# src/presentation/api/routers/synthesis_router.py

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List
from ..schemas.synthesis_schema import (
    SynthesisRequest,
    SynthesisResponse,
    SynthesisListResponse
)
from ..dependencies import (
    get_synthesize_speech_use_case,
    get_synthesis_history_use_case,
    get_synthesis_use_case,
    get_delete_synthesis_use_case
)

router = APIRouter(prefix="/syntheses", tags=["Synthesis"])

@router.post("", response_model=SynthesisResponse)
async def synthesize_speech(
    request: SynthesisRequest,
    use_case = Depends(get_synthesize_speech_use_case)
):
    """Generate speech from text."""
    try:
        result = await use_case.execute(
            text=request.text,
            model=request.model,
            voice_reference_id=request.voice_reference_id,
            language=request.language,
            cfg_weight=request.cfg_weight,
            exaggeration=request.exaggeration
        )
        return SynthesisResponse.from_dto(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=SynthesisListResponse)
async def list_syntheses(
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    use_case = Depends(get_synthesis_history_use_case)
):
    """List synthesis history with pagination."""
    syntheses = await use_case.execute(limit=limit, offset=offset)
    return SynthesisListResponse(
        items=[SynthesisResponse.from_dto(s) for s in syntheses],
        total=len(syntheses),
        limit=limit,
        offset=offset
    )

@router.get("/{synthesis_id}", response_model=SynthesisResponse)
async def get_synthesis(
    synthesis_id: str,
    use_case = Depends(get_synthesis_use_case)
):
    """Get specific synthesis."""
    result = await use_case.execute(synthesis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Synthesis not found")
    return SynthesisResponse.from_dto(result)

@router.delete("/{synthesis_id}")
async def delete_synthesis(
    synthesis_id: str,
    use_case = Depends(get_delete_synthesis_use_case)
):
    """Delete synthesis and its audio file."""
    success = await use_case.execute(synthesis_id)
    if not success:
        raise HTTPException(status_code=404, detail="Synthesis not found")
    return {"message": "Deleted successfully"}

@router.get("/{synthesis_id}/audio")
async def get_synthesis_audio(
    synthesis_id: str,
    download: bool = Query(False),
    use_case = Depends(get_synthesis_use_case)
):
    """Stream or download synthesized audio."""
    result = await use_case.execute(synthesis_id)
    if not result or not result.output_file_path:
        raise HTTPException(status_code=404, detail="Audio not found")

    headers = {}
    if download:
        filename = f"synthesis_{synthesis_id}.wav"
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return FileResponse(
        path=result.output_file_path,
        media_type="audio/wav",
        headers=headers
    )
```

### 11.3 Pydantic Schemas

```python
# src/presentation/api/schemas/synthesis_schema.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: str = Field(default="turbo", pattern="^(turbo|standard|multilingual)$")
    voice_reference_id: Optional[str] = None
    language: Optional[str] = Field(default=None, min_length=2, max_length=10)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)

class SynthesisResponse(BaseModel):
    id: str
    input_text: str
    text_length: int
    model: str
    status: str
    language: Optional[str]
    voice_reference_id: Optional[str]
    voice_reference_name: Optional[str]
    cfg_weight: float
    exaggeration: float
    output_duration_seconds: Optional[float]
    error_message: Optional[str]
    processing_time_seconds: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True

    @classmethod
    def from_dto(cls, dto):
        return cls(**dto.__dict__)

class SynthesisListResponse(BaseModel):
    items: List[SynthesisResponse]
    total: int
    limit: int
    offset: int

class VoiceReferenceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)

class VoiceReferenceResponse(BaseModel):
    id: str
    name: str
    original_filename: str
    duration_seconds: float
    created_at: datetime

class TTSModelResponse(BaseModel):
    name: str
    display_name: str
    parameters: str
    description: str
    supports_voice_cloning: bool
    supports_multilingual: bool
    supports_paralinguistics: bool
```

---

## 12. Presentation Layer - Frontend

### 12.1 Component Structure

```
src/presentation/frontend/src/app/
├── core/
│   ├── models/
│   │   ├── synthesis.model.ts
│   │   └── voice-reference.model.ts
│   └── services/
│       ├── api.service.ts (adapt)
│       ├── synthesis.service.ts (new)
│       └── voice.service.ts (new)
├── features/
│   ├── text-input/              # Main TTS input
│   │   ├── text-input.component.ts
│   │   ├── text-input.component.html
│   │   └── text-input.component.scss
│   ├── synthesis-history/       # History list
│   │   ├── synthesis-history.component.ts
│   │   ├── synthesis-history.component.html
│   │   └── synthesis-history.component.scss
│   ├── synthesis-detail/        # Detail view
│   │   ├── synthesis-detail.component.ts
│   │   ├── synthesis-detail.component.html
│   │   └── synthesis-detail.component.scss
│   └── voice-library/           # Voice management
│       ├── voice-library.component.ts
│       ├── voice-library.component.html
│       └── voice-library.component.scss
└── shared/
    └── components/
        ├── audio-player/        # Reusable player
        ├── footer/              # Keep existing
        └── popup/               # Keep existing
```

### 12.2 TypeScript Models

```typescript
// src/presentation/frontend/src/app/core/models/synthesis.model.ts

export type SynthesisStatus = 'pending' | 'processing' | 'completed' | 'failed';
export type TTSModel = 'turbo' | 'standard' | 'multilingual';

export interface Synthesis {
  id: string;
  input_text: string;
  text_length: number;
  model: TTSModel;
  status: SynthesisStatus;
  language: string | null;
  voice_reference_id: string | null;
  voice_reference_name: string | null;
  cfg_weight: number;
  exaggeration: number;
  output_duration_seconds: number | null;
  error_message: string | null;
  processing_time_seconds: number | null;
  created_at: string;
  completed_at: string | null;
}

export interface SynthesisRequest {
  text: string;
  model?: TTSModel;
  voice_reference_id?: string;
  language?: string;
  cfg_weight?: number;
  exaggeration?: number;
}

export interface TTSModelInfo {
  name: string;
  display_name: string;
  parameters: string;
  description: string;
  supports_voice_cloning: boolean;
  supports_multilingual: boolean;
  supports_paralinguistics: boolean;
}
```

```typescript
// src/presentation/frontend/src/app/core/models/voice-reference.model.ts

export interface VoiceReference {
  id: string;
  name: string;
  original_filename: string;
  duration_seconds: number;
  created_at: string;
}
```

### 12.3 App Routing

```typescript
// src/presentation/frontend/src/app/app-routing.module.ts

const routes: Routes = [
  { path: '', redirectTo: '/generate', pathMatch: 'full' },
  {
    path: 'generate',
    loadChildren: () => import('./features/text-input/text-input.module')
      .then(m => m.TextInputModule)
  },
  {
    path: 'history',
    loadChildren: () => import('./features/synthesis-history/synthesis-history.module')
      .then(m => m.SynthesisHistoryModule)
  },
  {
    path: 'synthesis/:id',
    loadChildren: () => import('./features/synthesis-detail/synthesis-detail.module')
      .then(m => m.SynthesisDetailModule)
  },
  {
    path: 'voices',
    loadChildren: () => import('./features/voice-library/voice-library.module')
      .then(m => m.VoiceLibraryModule)
  }
];
```

### 12.4 Navigation Structure

```html
<!-- Updated navigation for TTS -->
<nav class="main-nav">
  <div class="brand">
    <h1>Chatterbox UI</h1>
    <span class="subtitle">Text-to-Speech</span>
  </div>

  <div class="nav-links">
    <a routerLink="/generate" routerLinkActive="active">Generate</a>
    <a routerLink="/history" routerLinkActive="active">History</a>
    <a routerLink="/voices" routerLinkActive="active">Voices</a>
  </div>
</nav>
```

---

## 13. Configuration & Environment

### 13.1 `.env.example`

```env
# ===========================================
# CHATTERBOX UI - TTS APPLICATION
# ===========================================

# Application
APP_NAME=Chatterbox UI
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# API Configuration (Different ports from ASR)
API_HOST=0.0.0.0
API_PORT=8002
CORS_ORIGINS=["http://localhost:4201"]

# Database (New database name)
DATABASE_URL=sqlite:///./chatterbox_tts.db

# TTS Configuration
TTS_DEVICE=cuda
TTS_DEFAULT_MODEL=turbo
TTS_PRELOAD_MODEL=true
TTS_MAX_TEXT_LENGTH=5000

# File Storage
AUDIO_OUTPUT_DIR=./audio_outputs
VOICE_REFERENCE_DIR=./voice_references
VOICE_MIN_DURATION=5.0
VOICE_MAX_DURATION=30.0
VOICE_MAX_FILE_SIZE_MB=10
```

### 13.2 Frontend Environment

```typescript
// src/presentation/frontend/src/environments/environment.ts

export const environment = {
  production: false,
  appName: 'Chatterbox UI',
  apiUrl: 'http://localhost:8002/api/v1'  // Changed port
};
```

---

## 14. Docker Configuration

### 14.1 docker-compose.yml

```yaml
version: '3.8'

services:
  # PostgreSQL Database (NEW NAME)
  chatterbox-postgres:
    image: postgres:15-alpine
    container_name: chatterbox-postgres
    environment:
      POSTGRES_DB: chatterbox_tts
      POSTGRES_USER: ${POSTGRES_USER:-chatterbox}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-chatterbox_secret}
    volumes:
      - chatterbox-postgres-data:/var/lib/postgresql/data
    ports:
      - "5433:5432"  # Different external port
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U chatterbox -d chatterbox_tts"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Backend API (NEW NAME)
  chatterbox-backend:
    build:
      context: .
      dockerfile: src/presentation/api/Dockerfile
    container_name: chatterbox-backend
    environment:
      - DATABASE_URL=postgresql://chatterbox:chatterbox_secret@chatterbox-postgres:5432/chatterbox_tts
      - TTS_DEVICE=cuda
      - TTS_DEFAULT_MODEL=turbo
      - TTS_PRELOAD_MODEL=true
      - AUDIO_OUTPUT_DIR=/app/audio_outputs
      - VOICE_REFERENCE_DIR=/app/voice_references
    volumes:
      - chatterbox-audio-outputs:/app/audio_outputs
      - chatterbox-voice-references:/app/voice_references
      - chatterbox-model-cache:/root/.cache
      - ./src:/app/src:ro
    ports:
      - "8002:8002"  # Different port
    depends_on:
      chatterbox-postgres:
        condition: service_healthy
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Frontend (NEW NAME)
  chatterbox-frontend:
    build:
      context: .
      dockerfile: src/presentation/frontend/Dockerfile
    container_name: chatterbox-frontend
    environment:
      - API_URL=http://localhost:8002/api/v1
    volumes:
      - ./src/presentation/frontend:/app:ro
      - /app/node_modules
    ports:
      - "4201:4200"  # Different external port
    depends_on:
      - chatterbox-backend

volumes:
  chatterbox-postgres-data:
  chatterbox-audio-outputs:
  chatterbox-voice-references:
  chatterbox-model-cache:

networks:
  default:
    name: chatterbox-network
```

### 14.2 Backend Dockerfile Updates

```dockerfile
# src/presentation/api/Dockerfile

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY src/presentation/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chatterbox TTS
RUN pip install --no-cache-dir chatterbox-tts

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Create directories
RUN mkdir -p /app/audio_outputs /app/voice_references

# Copy source
COPY src/ /app/src/

# Expose port (NEW)
EXPOSE 8002

# Start server (NEW PORT)
CMD ["python", "-m", "uvicorn", "src.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8002", "--reload"]
```

---

## 15. Scripts & Automation

### 15.1 Updated Scripts

| Script | Changes |
|--------|---------|
| `scripts/server/run_backend.py` | Port 8002 |
| `scripts/server/run_frontend.py` | Port 4201 |
| `scripts/docker/run.py` | New container names |
| `scripts/setup/init_db.py` | New schema |

### 15.2 New Scripts

| Script | Purpose |
|--------|---------|
| `scripts/setup/preload_tts_models.py` | Preload Chatterbox models |
| `scripts/migrations/create_tts_schema.py` | Initialize TTS database |
| `scripts/maintenance/cleanup_audio_outputs.py` | Clean old audio files |

---

## 16. Testing Strategy

### 16.1 Unit Tests

| File | Coverage |
|------|----------|
| `tests/domain/test_synthesis.py` | Entity validation, state transitions |
| `tests/domain/test_voice_reference.py` | Validation rules |

### 16.2 Integration Tests

| File | Coverage |
|------|----------|
| `tests/infrastructure/test_chatterbox_service.py` | TTS generation |
| `tests/infrastructure/test_synthesis_repository.py` | Database operations |

### 16.3 E2E Tests

| File | Coverage |
|------|----------|
| `tests/e2e/test_synthesis_api.py` | Full synthesis workflow |
| `tests/e2e/test_voice_api.py` | Voice reference workflow |

### 16.4 Manual Testing Checklist

- [x] Generate speech with Turbo model (no voice)
- [x] Generate speech with Standard model + cfg/exaggeration
- [x] Generate speech with Multilingual model + language
- [x] Voice cloning from reference audio
- [x] Paralinguistic tags ([laugh], [cough])
- [x] Audio playback in browser
- [x] Audio download
- [x] History pagination
- [x] Delete synthesis (removes audio)
- [x] Upload voice reference
- [x] Delete voice reference
- [x] Error handling for text > 5000 chars
- [x] GPU acceleration verified

---

## 17. Implementation Phases

### Phase 1: Cleanup & Foundation

- [x] Create feature branch: `features/tts-replacement`
- [x] Remove all ASR-specific files (see Section 4)
- [x] Delete old database and uploads
- [x] Update settings for TTS
- [x] Create new directory structure

### Phase 2: Domain Layer

- [x] Create `Synthesis` entity
- [x] Create `VoiceReference` entity
- [x] Create repository interfaces
- [x] Create `TextToSpeechService` interface
- [x] Create value objects
- [x] Write unit tests

### Phase 3: Infrastructure Layer

- [x] Implement `ChatterboxService`
- [x] Create SQLAlchemy models
- [x] Implement repositories
- [x] Update file storage for audio outputs
- [x] Run database migration
- [x] Write integration tests

### Phase 4: Application Layer

- [x] Implement `SynthesizeSpeechUseCase`
- [x] Implement `ResynthesizeSpeechUseCase`
- [x] Implement history/get/delete use cases
- [x] Implement voice reference use cases
- [x] Create DTOs

### Phase 5: API Layer

- [x] Create Pydantic schemas
- [x] Implement synthesis router
- [x] Implement voice router
- [x] Implement model router
- [x] Update dependencies.py
- [x] Update main.py
- [x] Write API tests

### Phase 6: Frontend

- [x] Remove ASR components
- [x] Create synthesis models
- [x] Create synthesis service
- [x] Create voice service
- [x] Build text-input component
- [x] Build synthesis-history component
- [x] Build synthesis-detail component
- [x] Build voice-library component
- [x] Update routing
- [x] Update navigation
- [x] Apply existing theme/styling

### Phase 7: Docker & Deployment

- [x] Update docker-compose.yml
- [x] Update Dockerfiles
- [x] Update environment files
- [x] Create new volumes
- [x] Test full deployment

### Phase 8: Documentation

- [x] Update CLAUDE.md
- [x] Update README.md
- [x] Update .env.example
- [x] Final testing
- [x] Create PR

---

## 18. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Chatterbox CUDA compatibility | High | Test early on RTX 5090 |
| Model memory usage | Medium | Lazy load, single model at a time |
| Audio file storage growth | Medium | Implement cleanup scripts |
| Port conflicts with ASR repo | High | Use different ports (8002, 4201, 5433) |
| Breaking shared code | Medium | Test thoroughly before removing |

---

## 19. TODOs Checklist

### Pre-Implementation

- [x] **GET APPROVAL FOR THIS PLAN**
- [x] Test Chatterbox installation on local machine
- [x] Verify CUDA 12.8 / RTX 5090 compatibility
- [x] Create feature branch

### Phase 1: Cleanup

#### ASR Files Removal
- [x] Remove `src/domain/services/speech_recognition_service.py`
- [x] Remove `src/domain/entities/transcription.py`
- [x] Remove `src/domain/entities/audio_file.py`
- [x] Remove `src/domain/repositories/transcription_repository.py`
- [x] Remove `src/domain/repositories/audio_file_repository.py`
- [x] Remove `src/infrastructure/services/faster_whisper_service.py`
- [x] Remove `src/infrastructure/services/model_download_tracker.py`
- [x] Remove `src/infrastructure/persistence/models/transcription_model.py`
- [x] Remove `src/infrastructure/persistence/models/audio_file_model.py`
- [x] Remove `src/infrastructure/persistence/repositories/sqlite_transcription_repository.py`
- [x] Remove `src/infrastructure/persistence/repositories/sqlite_audio_file_repository.py`
- [x] Remove `src/application/use_cases/transcribe_audio_use_case.py`
- [x] Remove `src/application/use_cases/retranscribe_audio_use_case.py`
- [x] Remove `src/application/use_cases/get_transcription_use_case.py`
- [x] Remove `src/application/use_cases/get_transcription_history_use_case.py`
- [x] Remove `src/application/use_cases/delete_transcription_use_case.py`
- [x] Remove `src/application/use_cases/delete_audio_file_use_case.py`
- [x] Remove `src/application/use_cases/get_audio_file_transcriptions_use_case.py`
- [x] Remove `src/application/dto/transcription_dto.py`
- [x] Remove `src/application/dto/audio_upload_dto.py`
- [x] Remove `src/presentation/api/routers/transcription_router.py`
- [x] Remove `src/presentation/api/routers/audio_file_router.py`
- [x] Remove `src/presentation/api/schemas/transcription_schema.py`

#### LLM Enhancement Files Removal (NOT APPLICABLE TO TTS)
- [x] Remove `src/domain/services/llm_enhancement_service.py`
- [x] Remove `src/infrastructure/services/llm_enhancement_service_impl.py`
- [x] Remove `src/infrastructure/llm/` directory (llm_client.py)
- [x] Remove `src/application/use_cases/enhance_transcription_use_case.py`
- [x] Remove `src/application/enhancement/` directory (enhancement_agent.py, prompts.py)
- [x] Remove `src/presentation/api/routers/llm_enhancement_router.py`
- [x] Remove LLM dependencies from requirements.txt (langchain, langgraph, openai)
- [x] Remove LLM environment variables from .env.example

#### Frontend ASR/LLM Removal
- [x] Remove `src/presentation/frontend/src/app/features/upload/`
- [x] Remove `src/presentation/frontend/src/app/features/transcription/`
- [x] Remove `src/presentation/frontend/src/app/features/history/`
- [x] Remove `src/presentation/frontend/src/app/core/services/transcription.service.ts`
- [x] Remove `src/presentation/frontend/src/app/core/models/transcription.model.ts`
- [x] Remove `src/presentation/frontend/src/app/core/models/audio-file.model.ts`
- [x] Remove LLM-related UI components (enhance button, dual text, status badges)

#### Data Cleanup
- [x] Delete `whisper_transcriptions.db` database file
- [x] Delete `uploads/` directory
- [x] Update `.gitignore` for new directories (audio_outputs, voice_references)

### Phase 2: Domain Layer

- [x] `src/domain/entities/synthesis.py`
- [x] `src/domain/entities/voice_reference.py`
- [x] `src/domain/services/text_to_speech_service.py`
- [x] `src/domain/repositories/synthesis_repository.py`
- [x] `src/domain/repositories/voice_reference_repository.py`
- [x] `src/domain/value_objects/tts_model_info.py`
- [x] Unit tests

### Phase 3: Infrastructure Layer

- [x] `src/infrastructure/services/chatterbox_service.py`
- [x] `src/infrastructure/persistence/models/synthesis_model.py`
- [x] `src/infrastructure/persistence/models/voice_reference_model.py`
- [x] `src/infrastructure/persistence/repositories/sqlite_synthesis_repository.py`
- [x] `src/infrastructure/persistence/repositories/sqlite_voice_reference_repository.py`
- [x] Update `src/infrastructure/config/settings.py`
- [x] Update `src/infrastructure/storage/local_file_storage.py`
- [x] Integration tests

### Phase 4: Application Layer

- [x] `src/application/use_cases/synthesize_speech_use_case.py`
- [x] `src/application/use_cases/resynthesize_speech_use_case.py`
- [x] `src/application/use_cases/get_synthesis_use_case.py`
- [x] `src/application/use_cases/get_synthesis_history_use_case.py`
- [x] `src/application/use_cases/delete_synthesis_use_case.py`
- [x] `src/application/use_cases/upload_voice_reference_use_case.py`
- [x] `src/application/use_cases/get_voice_references_use_case.py`
- [x] `src/application/use_cases/delete_voice_reference_use_case.py`
- [x] `src/application/dto/synthesis_dto.py`
- [x] `src/application/dto/voice_reference_dto.py`

### Phase 5: API Layer

- [x] `src/presentation/api/schemas/synthesis_schema.py`
- [x] `src/presentation/api/schemas/voice_schema.py`
- [x] `src/presentation/api/routers/synthesis_router.py`
- [x] `src/presentation/api/routers/voice_router.py`
- [x] Update `src/presentation/api/routers/model_router.py`
- [x] Update `src/presentation/api/dependencies.py`
- [x] Update `src/presentation/api/main.py`
- [x] API tests

### Phase 6: Frontend

- [x] Remove old feature components
- [x] `core/models/synthesis.model.ts`
- [x] `core/models/voice-reference.model.ts`
- [x] `core/services/synthesis.service.ts`
- [x] `core/services/voice.service.ts`
- [x] Update `core/services/api.service.ts`
- [x] `features/text-input/*`
- [x] `features/synthesis-history/*`
- [x] `features/synthesis-detail/*`
- [x] `features/voice-library/*`
- [x] Update `app-routing.module.ts`
- [x] Update `app.component.*`
- [x] Update `environments/environment.ts`

### Phase 7: Configuration

- [x] `.env.example`
- [x] `.env.docker`
- [x] `docker-compose.yml`
- [x] `src/presentation/api/Dockerfile`
- [x] `scripts/server/run_backend.py`
- [x] `scripts/server/run_frontend.py`
- [x] `scripts/setup/init_db.py`
- [x] `scripts/setup/preload_tts_models.py`

### Phase 8: Documentation

- [x] `CLAUDE.md` - Complete rewrite for TTS
- [x] `README.md` - Complete rewrite for TTS
- [x] Final regression testing
- [x] Create PR

---

## Approval Request

This plan describes the **complete replacement** of ASR with TTS using Chatterbox, including:

- Removal of all ASR code
- Reuse of architecture, styling, and patterns
- New database, volumes, and ports (coexists with original repo)
- Both standard TTS and voice cloning

**Plan was approved and implemented.**

---

**Plan Status**: COMPLETED

**Implementation Notes**:
- Docker deployment verified working with CUDA 12.8 and PyTorch 2.9.1+cu128
- RTX 5090 GPU (Blackwell architecture, sm_120) fully supported
- HuggingFace token required for model downloads (~3.8GB for turbo model)
- TTS synthesis tested and working (~3.24s audio generated in ~23s processing time)
