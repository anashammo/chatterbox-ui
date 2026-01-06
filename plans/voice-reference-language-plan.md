# Plan: Add Language Field to Voice References

## Overview
Add an optional language field to voice references to help users identify the appropriate voice for their text-to-speech synthesis. This improves UX when users have multiple voice references in different languages.

## Impact Analysis

### Layers Affected
1. **Domain Layer** - Entity definition
2. **Infrastructure Layer** - Database model, repository
3. **Application Layer** - DTOs, use cases
4. **Presentation Layer (Backend)** - API schemas, router
5. **Presentation Layer (Frontend)** - Models, services, components
6. **Documentation** - CLAUDE.md, README.md
7. **Scripts** - Database display utilities

### Database Migration Required
- Add `language` column to `voice_references` table
- Column should be nullable (optional field)
- No data migration needed for existing records

---

## Implementation Plan

### Phase 1: Backend - Domain & Infrastructure

#### 1.1 Update Domain Entity
**File:** `src/domain/entities/voice_reference.py`
- Add `language: Optional[str] = None` field to `VoiceReference` dataclass
- Add after `duration_seconds` field

#### 1.2 Update Database Model
**File:** `src/infrastructure/persistence/models/voice_reference_model.py`
- Add `language = Column(String(10), nullable=True)` column
- Position after `duration_seconds`

#### 1.3 Update Repository Implementation
**File:** `src/infrastructure/persistence/repositories/sqlite_voice_reference_repository.py`
- Update `_to_entity()` to map `language` field
- Update `_to_model()` to map `language` field

### Phase 2: Backend - Application Layer

#### 2.1 Update DTOs
**File:** `src/application/dto/voice_reference_dto.py`

**VoiceReferenceDTO:**
- Add `language: Optional[str] = None` field
- Update `from_entity()` to include language
- Update `to_dict()` to include language

**VoiceReferenceCreateDTO:**
- Add `language: Optional[str] = None` field
- Add validation: if provided, must be 2-5 character language code

### Phase 3: Backend - Presentation Layer

#### 3.1 Update API Schemas
**File:** `src/presentation/api/schemas/voice_reference_schema.py`

**VoiceReferenceResponse:**
- Add `language: Optional[str] = None`

**VoiceReferenceUploadResponse:**
- Add `language: Optional[str] = None`

#### 3.2 Update API Router
**File:** `src/presentation/api/routers/voice_reference_router.py`
- Update upload endpoint to accept `language` form field (optional)
- Pass language to `VoiceReferenceCreateDTO`
- Include language in response

### Phase 4: Frontend - Models & Services

#### 4.1 Update Frontend Model
**File:** `src/presentation/frontend/src/app/core/models/voice-reference.model.ts`
- Add `language?: string` to `VoiceReference` interface
- Add `language?: string` to `VoiceReferenceUploadResponse` interface

#### 4.2 Update Synthesis Service
**File:** `src/presentation/frontend/src/app/core/services/synthesis.service.ts`
- Update `uploadVoiceReference()` to accept optional language parameter
- Update the newly created `VoiceReference` object to include language

#### 4.3 Update API Service
**File:** `src/presentation/frontend/src/app/core/services/api.service.ts`
- Update `uploadVoiceReference()` to accept optional language parameter
- Append language to FormData if provided

### Phase 5: Frontend - Voice References Component

#### 5.1 Update Upload Form
**File:** `src/presentation/frontend/src/app/features/voice-references/voice-references.component.html`
- Add language dropdown/select in upload form
- Options: Common language codes (en, es, ar, fr, de, zh, ja, etc.)
- Include "Not specified" option as default

#### 5.2 Update Component TypeScript
**File:** `src/presentation/frontend/src/app/features/voice-references/voice-references.component.ts`
- Add `selectedLanguage: string = ''` property
- Add `availableLanguages` array with language options
- Update `uploadVoiceReference()` to pass language
- Update `removeSelectedFile()` to reset language
- Display language in voice reference cards

#### 5.3 Update Component CSS
**File:** `src/presentation/frontend/src/app/features/voice-references/voice-references.component.css`
- Style language dropdown to match form design
- Add language badge styling for voice cards

### Phase 6: Frontend - Synthesize Component

#### 6.1 Update Voice Reference Dropdown
**File:** `src/presentation/frontend/src/app/features/synthesize/synthesize.component.html`
- Show language in voice reference dropdown options
- Format: `{{ ref.name }} ({{ ref.language || 'No lang' }}) - {{ ref.duration_seconds }}s`

### Phase 7: Documentation & Scripts

#### 7.1 Update CLAUDE.md
**File:** `CLAUDE.md`
- Update Database Schema section to include `language` column in `voice_references` table
- Add documentation for language field feature under Key Features

#### 7.2 Update README.md (if exists)
**File:** `README.md`
- Add language field to voice reference feature description
- Update API documentation for upload endpoint (if documented)

#### 7.3 Update .env.example (if new env vars needed)
**File:** `src/presentation/api/.env.example`
- No new environment variables expected for this feature

#### 7.4 Review Scripts
**Location:** `scripts/`
- Review `scripts/setup/init_db.py` - may need to acknowledge new column
- Review `scripts/maintenance/show_db_contents.py` - should display new field

---

## Language Options

Recommended language codes to support (matching Chatterbox TTS multilingual model):
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ar` - Arabic
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean
- `ru` - Russian
- `hi` - Hindi
- `Other` - Custom/unspecified

---

## File Changes Summary

| File | Change Type |
|------|-------------|
| `src/domain/entities/voice_reference.py` | Add field |
| `src/infrastructure/persistence/models/voice_reference_model.py` | Add column |
| `src/infrastructure/persistence/repositories/sqlite_voice_reference_repository.py` | Update mapping |
| `src/application/dto/voice_reference_dto.py` | Add field to DTOs |
| `src/presentation/api/schemas/voice_reference_schema.py` | Add field to schemas |
| `src/presentation/api/routers/voice_reference_router.py` | Accept language param |
| `src/presentation/frontend/.../voice-reference.model.ts` | Add field |
| `src/presentation/frontend/.../api.service.ts` | Update upload method |
| `src/presentation/frontend/.../synthesis.service.ts` | Update upload method |
| `src/presentation/frontend/.../voice-references.component.html` | Add language select |
| `src/presentation/frontend/.../voice-references.component.ts` | Add language logic |
| `src/presentation/frontend/.../voice-references.component.css` | Style language UI |
| `src/presentation/frontend/.../synthesize.component.html` | Show language in dropdown |
| `CLAUDE.md` | Update database schema docs |
| `README.md` | Update feature documentation |
| `scripts/maintenance/show_db_contents.py` | Display language field |

---

## Testing Plan

1. **Backend:**
   - Upload voice reference with language
   - Upload voice reference without language (should work)
   - List voice references shows language
   - Get single voice reference shows language

2. **Frontend:**
   - Language dropdown appears in upload form
   - Language is sent on upload
   - Language displays in voice reference cards
   - Language shows in synthesize dropdown
   - Existing voice references without language still work

---

## Risks & Rollback

### Risks
- Database schema change (mitigated: nullable column, backward compatible)
- Existing voice references without language (mitigated: optional field with "Not specified" display)

### Rollback Steps
1. Revert code changes
2. Remove `language` column from database (if needed)
3. Existing data remains intact due to nullable column

---

## Notes

- Language field is **optional** - backward compatible with existing voice references
- No database migration script needed for SQLite (column added with nullable=True)
- Docker PostgreSQL may need Alembic migration if using strict schema management

---

## TODOs

- [x] Phase 1: Backend - Domain & Infrastructure
  - [x] 1.1 Update Domain Entity
  - [x] 1.2 Update Database Model
  - [x] 1.3 Update Repository Implementation
- [x] Phase 2: Backend - Application Layer
  - [x] 2.1 Update DTOs
- [x] Phase 3: Backend - Presentation Layer
  - [x] 3.1 Update API Schemas
  - [x] 3.2 Update API Router
- [x] Phase 4: Frontend - Models & Services
  - [x] 4.1 Update Frontend Model
  - [x] 4.2 Update Synthesis Service
  - [x] 4.3 Update API Service
- [x] Phase 5: Frontend - Voice References Component
  - [x] 5.1 Update Upload Form
  - [x] 5.2 Update Component TypeScript
  - [x] 5.3 Update Component CSS
- [x] Phase 6: Frontend - Synthesize Component
  - [x] 6.1 Update Voice Reference Dropdown
- [x] Phase 7: Documentation & Scripts
  - [x] 7.1 Update CLAUDE.md
  - [x] 7.2 Update README.md (N/A - no separate README)
  - [x] 7.4 Review Scripts (show_db_contents.py updated)
- [ ] Testing
  - [ ] Backend testing
  - [ ] Frontend testing
