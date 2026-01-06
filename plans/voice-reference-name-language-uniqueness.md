# Voice Reference Name + Language Uniqueness

## Problem Statement

When uploading a voice reference with the same name but different language, the system incorrectly rejects it with:
```
Voice reference with name 'Anas' already exists
```

The system should allow the same name with different languages (e.g., "Anas" in "en" and "Anas" in "ar").

## Solution

Change the uniqueness constraint from `name` alone to a composite constraint on `(name, language)`.

---

## Implementation Summary

### Files Modified

| File | Change |
|------|--------|
| `src/infrastructure/persistence/models/voice_reference_model.py` | Composite unique constraint |
| `src/domain/repositories/voice_reference_repository.py` | Added `get_by_name_and_language()` |
| `src/infrastructure/persistence/repositories/sqlite_voice_reference_repository.py` | Implemented method |
| `src/application/use_cases/create_voice_reference_use_case.py` | Use new method |
| `scripts/maintenance/migrate_voice_reference_constraint.py` | Migration script |

---

## Detailed Changes

### 1. Database Model

**File**: `src/infrastructure/persistence/models/voice_reference_model.py`

```python
from sqlalchemy import UniqueConstraint

class VoiceReferenceModel(Base):
    __tablename__ = "voice_references"

    # Composite unique constraint
    __table_args__ = (
        UniqueConstraint('name', 'language', name='uq_voice_reference_name_language'),
    )

    name = Column(String(255), nullable=False)  # No longer unique alone
    language = Column(String(10), nullable=True)
```

### 2. Repository Interface

**File**: `src/domain/repositories/voice_reference_repository.py`

```python
@abstractmethod
async def get_by_name_and_language(
    self,
    name: str,
    language: Optional[str]
) -> Optional[VoiceReference]:
    """Retrieve a voice reference by name and language combination."""
    pass
```

### 3. Repository Implementation

**File**: `src/infrastructure/persistence/repositories/sqlite_voice_reference_repository.py`

```python
async def get_by_name_and_language(
    self,
    name: str,
    language: Optional[str]
) -> Optional[VoiceReference]:
    model = self.db.query(VoiceReferenceModel).filter(
        VoiceReferenceModel.name == name,
        VoiceReferenceModel.language == language
    ).first()
    return self._to_entity(model) if model else None
```

### 4. Use Case Update

**File**: `src/application/use_cases/create_voice_reference_use_case.py`

```python
# Check if name + language combination already exists
existing = await self.voice_ref_repo.get_by_name_and_language(
    create_dto.name, create_dto.language
)
if existing:
    lang_info = f" ({create_dto.language})" if create_dto.language else ""
    raise ValueError(
        f"Voice reference with name '{create_dto.name}'{lang_info} already exists"
    )
```

---

## Migration

### Run Migration Script

```bash
# Run inside Docker container
docker exec chatterbox-backend python scripts/maintenance/migrate_voice_reference_constraint.py
```

The script:
1. Drops old constraint: `voice_references_name_key`
2. Adds new constraint: `uq_voice_reference_name_language`

---

## Testing

### Test 1: Upload Same Name Different Language
```bash
# Upload "TestVoice" with language "en"
curl -X POST http://localhost:8002/api/v1/voice-references \
  -F "file=@test.wav" -F "name=TestVoice" -F "language=en"
# Expected: Success

# Upload "TestVoice" with language "ar"
curl -X POST http://localhost:8002/api/v1/voice-references \
  -F "file=@test.wav" -F "name=TestVoice" -F "language=ar"
# Expected: Success (different language allowed)
```

### Test 2: Upload Duplicate Name + Language
```bash
# Upload "TestVoice" with language "en" again
curl -X POST http://localhost:8002/api/v1/voice-references \
  -F "file=@test.wav" -F "name=TestVoice" -F "language=en"
# Expected: Error - "Voice reference with name 'TestVoice' (en) already exists"
```

### Test 3: Verify List API
```bash
curl http://localhost:8002/api/v1/voice-references
# Expected: Both "TestVoice" entries with their respective languages
```

### Test 4: Verify Database Constraint
```bash
docker exec chatterbox-postgres psql -U chatterbox -d chatterbox_tts -c "\d voice_references"
# Should show: uq_voice_reference_name_language UNIQUE (name, language)
```

---

## Rollback (if needed)

```sql
-- Revert to name-only constraint
ALTER TABLE voice_references DROP CONSTRAINT IF EXISTS uq_voice_reference_name_language;
ALTER TABLE voice_references ADD CONSTRAINT voice_references_name_key UNIQUE (name);
```

Note: Rollback may fail if duplicate names with different languages exist.
